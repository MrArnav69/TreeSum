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

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Ablation Studies/Architectural Ablation: Hierarchical vs. Linear Processing/Evaluation of Advanced Metrics/BARTScore'))
try:
    from bart_score import BARTScorer
except ImportError:
    print("⚠️ BARTScore not found. Ensure 'BARTScore' folder exists.")

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

def evaluate_bartscore(samples, model_name, scorer, direction="doc2sum"):
    """Evaluate BARTScore"""
    print(f"Evaluating BARTScore for {model_name} ({direction})...")
    
    sources = []
    hypotheses = []
    
    for sample in samples:
        sources.append(sample[model_name]['document'])
        hypotheses.append(sample[model_name]['generated_summary'])
    
    # Calculate BARTScore with A40 GPU optimization (matching original config)
    if direction == "doc2sum":
        scores = scorer.score(sources, hypotheses, batch_size=32)
    else:  # sum2doc
        scores = scorer.score(hypotheses, sources, batch_size=32)
    
    return scores

def main():
    print("=== BARTScore Evaluation for 2500 samples ===")
    
    # Set device - optimize for A40 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimize for A40 GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Initialize BARTScorer with A40 optimization
    scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    
    # Load datasets
    datasets = find_datasets()
    
    # Create matched samples
    matched_samples = create_matched_samples(datasets)
    
    # Evaluate each model and save separate files
    models = ['PRIMERA', 'PEGASUS', 'BART', 'TreeSum']
    
    for model in models:
        print(f"\n=== Processing {model} ===")
        
        # Evaluate both directions like original
        doc2sum_scores = evaluate_bartscore(matched_samples, model, scorer, "doc2sum")
        sum2doc_scores = evaluate_bartscore(matched_samples, model, scorer, "sum2doc")
        
        # Create model-specific summary data
        model_summary = [{
            'model': model,
            'bartscore_doc2sum_mean': np.mean(doc2sum_scores),
            'bartscore_doc2sum_std': np.std(doc2sum_scores),
            'bartscore_doc2sum_min': np.min(doc2sum_scores),
            'bartscore_doc2sum_max': np.max(doc2sum_scores),
            'bartscore_doc2sum_median': np.median(doc2sum_scores),
            'bartscore_sum2doc_mean': np.mean(sum2doc_scores),
            'bartscore_sum2doc_std': np.std(sum2doc_scores),
            'bartscore_sum2doc_min': np.min(sum2doc_scores),
            'bartscore_sum2doc_max': np.max(sum2doc_scores),
            'bartscore_sum2doc_median': np.median(sum2doc_scores)
        }]
        
        # Create model-specific detailed data
        model_detailed = []
        for i, sample in enumerate(matched_samples):
            model_detailed.append({
                'sample_id': sample['sample_id'],
                'bartscore_doc2sum': doc2sum_scores[i],
                'bartscore_sum2doc': sum2doc_scores[i]
            })
        
        # Save model-specific files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary file for this model
        summary_df = pd.DataFrame(model_summary)
        summary_df = summary_df.set_index('model')
        summary_file = results_dir / f"bartscore_summary_{model}_{timestamp}.csv"
        summary_df.to_csv(summary_file)
        print(f"✅ {model} summary saved to {summary_file}")
        
        # Save detailed file for this model
        detailed_df = pd.DataFrame(model_detailed)
        detailed_df = detailed_df.set_index('sample_id')
        detailed_file = results_dir / f"bartscore_detailed_{model}_{timestamp}.csv"
        detailed_df.to_csv(detailed_file)
        print(f"✅ {model} detailed saved to {detailed_file}")
        
        # Print model summary
        print(f"\n=== {model} BARTScore Summary ===")
        print(summary_df)

if __name__ == "__main__":
    main()
