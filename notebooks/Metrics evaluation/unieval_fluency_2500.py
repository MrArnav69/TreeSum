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

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Ablation Studies/Architectural Ablation: Hierarchical vs. Linear Processing/Evaluation of Advanced Metrics/UniEval'))
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

results_dir = Path("results_metrics")
results_dir.mkdir(exist_ok=True)

def find_datasets():
    base_dir = Path("../../")
    files = {
        'PRIMERA': base_dir / 'results' / 'PRIMERA-multinews' / 'vanilla_primera_summaries_2500.json',
        'PEGASUS': base_dir / 'results' / 'PEGASUS-multi_news' / 'vanilla_pegasus_summaries_2500.json',
        'BART': base_dir / 'results' / 'BART-large-cnn' / 'vanilla_bart_summaries_2500.json',
        'TreeSum_p1': base_dir / 'results' / 'TreeSum' / 'treesum_summaries_p1.json',
        'TreeSum_p2': base_dir / 'results' / 'TreeSum' / 'treesum_summaries_p2.json',
        'TreeSum_p3': base_dir / 'results' / 'TreeSum' / 'treesum_summaries_p3.json',
        'TreeSum_p4': base_dir / 'results' / 'TreeSum' / 'treesum_summaries_p4.json',
        'TreeSum_p5': base_dir / 'results' / 'TreeSum' / 'treesum_summaries_p5.json'
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
    # Combine TreeSum parts
    treesum_combined = (datasets['TreeSum_p1'] + datasets['TreeSum_p2'] + 
                       datasets['TreeSum_p3'] + datasets['TreeSum_p4'] + datasets['TreeSum_p5'])
    
    primera_map = {item['sample_id']: item for item in datasets['PRIMERA']}
    pegasus_map = {item['sample_id']: item for item in datasets['PEGASUS']}
    bart_map = {item['sample_id']: item for item in datasets['BART']}
    treesum_map = {item['sample_id']: item for item in treesum_combined}
    
    # Find common sample IDs across all models
    common_ids = (set(primera_map.keys()) & set(pegasus_map.keys()) & 
                 set(bart_map.keys()) & set(treesum_map.keys()))
    
    matched_samples = []
    for sample_id in sorted(common_ids):
        matched_samples.append({
            'sample_id': sample_id,
            'PRIMERA': primera_map[sample_id],
            'PEGASUS': pegasus_map[sample_id],
            'BART': bart_map[sample_id],
            'TreeSum': treesum_map[sample_id]
        })
    
    print(f"Found {len(matched_samples)} matching samples across all models")
    return matched_samples

def evaluate_fluency(samples, model_name):
    """Evaluate fluency using UniEval"""
    print(f"Evaluating fluency for {model_name}...")
    
    # Prepare data for UniEval
    data_list = []
    for sample in samples:
        data_list.append({
            'src': sample[model_name]['document'],
            'hyp': sample[model_name]['generated_summary']
        })
    
    # Convert to UniEval format
    eval_data = convert_to_json(data_list, task='sum', ref_key=None)
    
    # Get evaluator
    evaluator = get_evaluator(task='sum', metrics=['fluency'])
    
    # Evaluate
    scores = evaluator.evaluate(eval_data, print_result=False)
    
    return scores

def main():
    print("=== UniEval Fluency Evaluation for 2500 samples ===")
    
    # Set device - optimize for A40 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimize for A40 GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Load datasets
    datasets = find_datasets()
    
    # Create matched samples
    matched_samples = create_matched_samples(datasets)
    
    # Evaluate each model
    results = {}
    detailed_results = []
    
    models = ['PRIMERA', 'PEGASUS', 'BART', 'TreeSum']
    
    for model in models:
        scores = evaluate_fluency(matched_samples, model)
        results[model] = scores
        
        # Add to detailed results
        for i, sample in enumerate(matched_samples):
            if i == 0:
                detailed_results.append({
                    'sample_id': sample['sample_id'],
                    f'{model}_fluency': scores[i]
                })
            else:
                detailed_results[-1][f'{model}_fluency'] = scores[i]
    
    # Calculate summary statistics
    summary_data = []
    for model in models:
        model_scores = results[model]
        summary_data.append({
            'model': model,
            'fluency_mean': np.mean(model_scores),
            'fluency_std': np.std(model_scores),
            'fluency_min': np.min(model_scores),
            'fluency_max': np.max(model_scores),
            'fluency_median': np.median(model_scores)
        })
    
    # Save summary results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('model')
    summary_file = results_dir / f"unieval_fluency_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file)
    print(f"✅ Summary saved to {summary_file}")
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df = detailed_df.set_index('sample_id')
    detailed_file = results_dir / f"unieval_fluency_detailed_{timestamp}.csv"
    detailed_df.to_csv(detailed_file)
    print(f"✅ Detailed results saved to {detailed_file}")
    
    # Print summary
    print("\n=== Fluency Summary ===")
    print(summary_df)

if __name__ == "__main__":
    main()
