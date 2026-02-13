#!/usr/bin/env python3
import json
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from scipy.stats import wilcoxon
import random
from datasets import load_dataset
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'BARTScore'))
try:
    from bart_score import BARTScorer
except ImportError:
    print("⚠️ BARTScore not found. Ensure 'BARTScore' folder exists.")

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

def get_source_documents(num_samples=1000, seed=42):
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_samples)
    samples = dataset.select(indices)
    return {indices[i]: samples[i]['document'] for i in range(len(indices))}

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

def evaluate_bartscore(bart_scorer, documents, summaries, direction="doc2sum"):
    scores = []
    batch_size = 32
    
    for i in tqdm(range(0, len(documents), batch_size), desc=f"bartscore {direction}"):
        batch_docs = documents[i:i + batch_size]
        batch_sums = summaries[i:i + batch_size]
        current_batch_size = len(batch_docs)
        success = False
        
        while current_batch_size >= 1 and not success:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                actual_docs = batch_docs[:current_batch_size]
                actual_sums = batch_sums[:current_batch_size]
                
                valid_pairs = [(d, s) for d, s in zip(actual_docs, actual_sums) 
                             if isinstance(d, str) and isinstance(s, str) and d.strip() and s.strip()]
                
                if not valid_pairs:
                    scores.extend([-10.0] * current_batch_size)
                    success = True
                    continue
                
                v_docs, v_sums = zip(*valid_pairs)
                if direction == "doc2sum":
                    batch_res = bart_scorer.score(list(v_docs), list(v_sums), batch_size=len(v_docs))
                else:
                    batch_res = bart_scorer.score(list(v_sums), list(v_docs), batch_size=len(v_docs))
                
                while len(batch_res) < current_batch_size:
                    batch_res.append(-10.0)
                
                scores.extend(batch_res)
                success = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    current_batch_size //= 2
                else:
                    scores.extend([-10.0] * current_batch_size)
                    success = True
            except Exception:
                scores.extend([-10.0] * current_batch_size)
                success = True
    return scores

def calculate_significance(df):
    print("\n🔬 statistical significance (wilcoxon)")
    for baseline in ['flat_1024', 'flat_overlap']:
        for metric in ['doc2sum', 'sum2doc']:
            col_tree = f'treesum_{metric}'
            col_base = f'{baseline}_{metric}'
            try:
                _, p = wilcoxon(df[col_tree], df[col_base])
                diff = df[col_tree].mean() - df[col_base].mean()
                print(f"   {baseline} {metric}: diff={diff:.4f}, p={p:.4e} {'✅' if p < 0.05 else '❌'}")
            except Exception: pass

def main():
    datasets = find_datasets()
    matched_samples = create_matched_samples(datasets)
    doc_map = get_source_documents()
    
    bart_scorer = BARTScorer(device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint='facebook/bart-large-cnn')
    
    datasets_info = {
        'flat_1024': {
            'docs': [doc_map[s['sample_id']] for s in matched_samples],
            'generated': [s['flat_1024']['generated_summary'] for s in matched_samples]
        },
        'flat_overlap': {
            'docs': [doc_map[s['sample_id']] for s in matched_samples],
            'generated': [s['flat_overlap']['generated_summary'] for s in matched_samples]
        },
        'treesum': {
            'docs': [doc_map[s['sample_id']] for s in matched_samples],
            'generated': [s['treesum']['generated_summary'] for s in matched_samples]
        }
    }
    
    results = {}
    for name, data in datasets_info.items():
        print(f"\nevaluating {name}...")
        results[name] = {
            'doc2sum': evaluate_bartscore(bart_scorer, data['docs'], data['generated'], "doc2sum"),
            'sum2doc': evaluate_bartscore(bart_scorer, data['docs'], data['generated'], "sum2doc")
        }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = []
    for i, sample in enumerate(matched_samples):
        row = {'sample_id': sample['sample_id']}
        for name in results.keys():
            row[f'{name}_doc2sum'] = results[name]['doc2sum'][i]
            row[f'{name}_sum2doc'] = results[name]['sum2doc'][i]
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    df.to_csv(results_dir / f"bartscore_detailed_{timestamp}.csv", index=False)
    calculate_significance(df)
    
    summary_stats = {}
    for name in results.keys():
        for metric in ['doc2sum', 'sum2doc']:
            scores = results[name][metric]
            summary_stats[f"{name}_{metric}"] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            }
    
    pd.DataFrame(summary_stats).T.to_csv(results_dir / f"bartscore_summary_{timestamp}.csv")
    print(f"\n✅ saved to {results_dir}")

if __name__ == "__main__":
    main()