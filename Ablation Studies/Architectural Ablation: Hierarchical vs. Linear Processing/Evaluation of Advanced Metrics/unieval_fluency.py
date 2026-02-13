#!/usr/bin/env python3
import json
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'UniEval'))
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

results_dir = Path("results_metrics")
results_dir.mkdir(exist_ok=True)

def find_datasets():
    base_dir = Path("..")
    files = {
        'flat_1024': base_dir / 'Flat 1024' / 'results_flat_1024' / 'summaries_flat_1024.json',
        'flat_overlap': base_dir / 'Flat Overlap ' / 'results_flat_overlap' / 'summaries_flat_overlap.json',
        'treesum_pt1': base_dir / 'Treesum' / 'treesum_part1_results' / 'summaries_treesum_pt1_first_500.json',
        'treesum_pt2': base_dir / 'Treesum' / 'treesum_part2_results' / 'summaries_treesum_pt2_last_500.json'
    }
    
    datasets = {}
    for name, file_path in files.items():
        if file_path.exists():
            with open(file_path, 'r') as f:
                datasets[name] = json.load(f)
            print(f"✅ {file_path.name}")
        else:
            raise FileNotFoundError(f"❌ Missing {file_path}")
    return datasets

def create_matched_samples(datasets):
    treesum_combined = datasets['treesum_pt1'] + datasets['treesum_pt2']
    flat_1024_map = {item['sample_id']: item for item in datasets['flat_1024']}
    flat_overlap_map = {item['sample_id']: item for item in datasets['flat_overlap']}
    treesum_map = {item['sample_id']: item for item in treesum_combined}
    
    common_ids = set(flat_1024_map.keys()) & set(flat_overlap_map.keys()) & set(treesum_map.keys())
    
    matched_samples = []
    for sample_id in sorted(common_ids):
        matched_samples.append({
            'sample_id': sample_id,
            'flat_1024': flat_1024_map[sample_id],
            'flat_overlap': flat_overlap_map[sample_id],
            'treesum': treesum_map[sample_id]
        })
    
    print(f"✅ {len(matched_samples)} matched samples")
    return matched_samples

def evaluate_unieval(evaluator, summaries):
    dimension = 'fluency'
    scores = []
    batch_size = 128
    
    for i in tqdm(range(0, len(summaries), batch_size), desc="unieval fluency"):
        batch_sums = summaries[i:i + batch_size]
        current_batch_size = len(batch_sums)
        success = False
        
        while current_batch_size >= 1 and not success:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                actual_batch = batch_sums[:current_batch_size]
                filtered_sums = [s if s.strip() else "Empty summary" for s in actual_batch]
                data = convert_to_json(output_list=filtered_sums)
                
                eval_scores = evaluator.evaluate(data, dims=[dimension], overall=False, print_result=False)
                
                if isinstance(eval_scores, list) and len(eval_scores) > 0:
                    batch_res = [s.get(dimension, 0.0) for s in eval_scores]
                    scores.extend(batch_res)
                else:
                    scores.extend([0.0] * len(filtered_sums))
                
                success = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    current_batch_size //= 2
                    if current_batch_size < 1:
                        scores.extend([0.0] * len(batch_sums))
                        success = True
                else:
                    scores.extend([0.0] * current_batch_size)
                    success = True
            except Exception:
                scores.extend([0.0] * current_batch_size)
                success = True
                
    return scores

def main():
    datasets = find_datasets()
    matched_samples = create_matched_samples(datasets)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    evaluator = get_evaluator('summarization', device='cuda' if torch.cuda.is_available() else 'cpu', max_length=16384)
    
    datasets_info = {
        'flat_1024': [s['flat_1024']['generated_summary'] for s in matched_samples],
        'flat_overlap': [s['flat_overlap']['generated_summary'] for s in matched_samples],
        'treesum': [s['treesum']['generated_summary'] for s in matched_samples]
    }
    
    results = {}
    for name, summaries in datasets_info.items():
        print(f"\nevaluating {name}...")
        results[name] = evaluate_unieval(evaluator, summaries)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = []
    for i, sample in enumerate(matched_samples):
        results_data.append({
            'sample_id': sample['sample_id'],
            'flat_1024_fluency': results['flat_1024'][i],
            'flat_overlap_fluency': results['flat_overlap'][i],
            'treesum_fluency': results['treesum'][i]
        })
    
    df = pd.DataFrame(results_data)
    df.to_csv(results_dir / f"unieval_fluency_detailed_{timestamp}.csv", index=False)
    
    summary_stats = {}
    for name in results.keys():
        scores = results[name]
        summary_stats[name] = {
            'fluency_mean': np.mean(scores),
            'fluency_std': np.std(scores),
            'fluency_min': np.min(scores),
            'fluency_max': np.max(scores),
            'fluency_median': np.median(scores)
        }
    
    pd.DataFrame(summary_stats).T.to_csv(results_dir / f"unieval_fluency_summary_{timestamp}.csv")
    print(f"\n✅ saved to {results_dir}")

if __name__ == "__main__":
    main()
