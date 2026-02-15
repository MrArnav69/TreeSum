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
try:
    from utils import convert_to_json
    from metric.evaluator import SumEvaluator
except ImportError as e:
    print(f"⚠️ UniEval import error: {e}")
    print("Ensure UniEval folder exists and has proper structure.")

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
    src_list = []
    output_list = []
    ref_list = []
    
    for sample in samples:
        src_list.append(sample[model_name]['document'])
        output_list.append(sample[model_name]['generated_summary'])
        ref_list.append(sample[model_name]['reference_summary'])
    
    # Convert to UniEval format
    eval_data = convert_to_json(output_list=output_list, src_list=src_list, ref_list=ref_list)
    
    # Get evaluator
    evaluator = SumEvaluator(device='cuda', max_length=1024)
    
    # Evaluate only fluency dimension
    scores_list = evaluator.evaluate(eval_data, dims=['fluency'], print_result=False)
    
    # Extract fluency scores (returns list of dicts)
    fluency_scores = [score['fluency'] for score in scores_list]
    
    return fluency_scores

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
    
    # Evaluate each model and save separate files
    models = ['PRIMERA', 'PEGASUS', 'BART', 'TreeSum']
    
    for model in models:
        print(f"\n=== Processing {model} ===")
        
        scores = evaluate_fluency(matched_samples, model)
        
        # Create model-specific summary data
        model_summary = [{
            'model': model,
            'fluency_mean': np.mean(scores),
            'fluency_std': np.std(scores),
            'fluency_min': np.min(scores),
            'fluency_max': np.max(scores),
            'fluency_median': np.median(scores)
        }]
        
        # Create model-specific detailed data
        model_detailed = []
        for i, sample in enumerate(matched_samples):
            model_detailed.append({
                'sample_id': sample['sample_id'],
                'fluency': scores[i]
            })
        
        # Save model-specific files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary file for this model
        summary_df = pd.DataFrame(model_summary)
        summary_df = summary_df.set_index('model')
        summary_file = results_dir / f"unieval_fluency_summary_{model}_{timestamp}.csv"
        summary_df.to_csv(summary_file)
        print(f"✅ {model} summary saved to {summary_file}")
        
        # Save detailed file for this model
        detailed_df = pd.DataFrame(model_detailed)
        detailed_df = detailed_df.set_index('sample_id')
        detailed_file = results_dir / f"unieval_fluency_detailed_{model}_{timestamp}.csv"
        detailed_df.to_csv(detailed_file)
        print(f"✅ {model} detailed saved to {detailed_file}")
        
        # Print model summary
        print(f"\n=== {model} Fluency Summary ===")
        print(summary_df)

if __name__ == "__main__":
    main()
