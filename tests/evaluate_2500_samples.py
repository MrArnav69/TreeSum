#!/usr/bin/env python3
"""
Script to combine TreeSum summary files (p1-p5) and evaluate against baseline models.
Combines 5 TreeSum files (p1-p5) to get 2500 samples and evaluates ROUGE scores
against PRIMERA, PEGASUS, and BART baselines on matching indices.
"""

import json
import os
from typing import List, Dict, Any
from rouge_score import rouge_scorer
import numpy as np
from collections import defaultdict

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file containing summaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def combine_treesum_files(treesum_files: List[str]) -> List[Dict[str, Any]]:
    """Combine multiple TreeSum files into one list."""
    combined_data = []
    
    for file_path in treesum_files:
        print(f"Loading {file_path}...")
        data = load_json_file(file_path)
        combined_data.extend(data)
        print(f"Added {len(data)} samples from {file_path}")
    
    print(f"Total combined TreeSum samples: {len(combined_data)}")
    return combined_data

def extract_matching_samples(baseline_data: List[Dict[str, Any]], 
                           treesum_data: List[Dict[str, Any]], 
                           num_samples: int = 2500) -> tuple:
    """Extract matching samples from baseline and TreeSum data."""
    # Create mapping from sample_id to treesum data (first 2500 samples)
    treesum_map = {item['sample_id']: item for item in treesum_data[:num_samples]}
    
    matching_baseline = []
    matching_treesum = []
    
    for baseline_item in baseline_data:
        sample_id = baseline_item['sample_id']
        if sample_id in treesum_map:  # Remove the sample_id < num_samples condition
            matching_baseline.append(baseline_item)
            matching_treesum.append(treesum_map[sample_id])
    
    print(f"Found {len(matching_baseline)} matching samples")
    return matching_baseline, matching_treesum

def calculate_rouge_scores(references: List[str], candidates: List[str]) -> Dict[str, float]:
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores)
    }

def main():
    # Define file paths
    treesum_files = [
        "/Users/mrarnav69/Documents/TreeSum/results/TreeSum/treesum_summaries_p1.json",
        "/Users/mrarnav69/Documents/TreeSum/results/TreeSum/treesum_summaries_p2.json", 
        "/Users/mrarnav69/Documents/TreeSum/results/TreeSum/treesum_summaries_p3.json",
        "/Users/mrarnav69/Documents/TreeSum/results/TreeSum/treesum_summaries_p4.json",
        "/Users/mrarnav69/Documents/TreeSum/results/TreeSum/treesum_summaries_p5.json"
    ]
    
    baseline_files = {
        "PRIMERA": "/Users/mrarnav69/Documents/TreeSum/results/PRIMERA-multinews/vanilla_primera_summaries_2500.json",
        "PEGASUS": "/Users/mrarnav69/Documents/TreeSum/results/PEGASUS-multi_news/vanilla_pegasus_summaries_2500.json",
        "BART": "/Users/mrarnav69/Documents/TreeSum/results/BART-large-cnn/vanilla_bart_summaries_2500.json"
    }
    
    print("=== TreeSum vs Baselines Evaluation (2500 samples) ===\n")
    
    # Load and combine TreeSum files
    treesum_data = combine_treesum_files(treesum_files)
    
    # Load baseline data
    baseline_data = {}
    for model_name, file_path in baseline_files.items():
        print(f"Loading {model_name} baseline...")
        baseline_data[model_name] = load_json_file(file_path)
        print(f"Loaded {len(baseline_data[model_name])} samples for {model_name}")
    
    print("\n=== Evaluating on 2500 matching samples ===\n")
    
    # Evaluate each baseline against TreeSum
    results = {}
    
    for model_name, baseline_samples in baseline_data.items():
        print(f"\nEvaluating {model_name} vs TreeSum...")
        
        # Extract matching samples
        matching_baseline, matching_treesum = extract_matching_samples(
            baseline_samples, treesum_data, num_samples=2500
        )
        
        if len(matching_baseline) == 0:
            print(f"No matching samples found for {model_name}")
            continue
        
        # Extract reference summaries and generated summaries
        references = [item['reference_summary'] for item in matching_baseline]
        baseline_candidates = [item['generated_summary'] for item in matching_baseline]
        treesum_candidates = [item['generated_summary'] for item in matching_treesum]
        
        # Calculate ROUGE scores
        baseline_scores = calculate_rouge_scores(references, baseline_candidates)
        treesum_scores = calculate_rouge_scores(references, treesum_candidates)
        
        results[model_name] = {
            'baseline': baseline_scores,
            'treesum': treesum_scores,
            'num_samples': len(matching_baseline)
        }
        
        print(f"{model_name} Results ({len(matching_baseline)} samples):")
        print(f"  Baseline ROUGE-1: {baseline_scores['rouge1']:.4f}")
        print(f"  TreeSum ROUGE-1:  {treesum_scores['rouge1']:.4f}")
        print(f"  Baseline ROUGE-2: {baseline_scores['rouge2']:.4f}")
        print(f"  TreeSum ROUGE-2:  {treesum_scores['rouge2']:.4f}")
        print(f"  Baseline ROUGE-L: {baseline_scores['rougeL']:.4f}")
        print(f"  TreeSum ROUGE-L:  {treesum_scores['rougeL']:.4f}")
    
    # Save results to file
    output_file = "/Users/mrarnav69/Documents/TreeSum/tests/evaluation_results_2500.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results saved to {output_file} ===")
    
    # Print summary table
    print("\n=== Summary Table ===")
    print(f"{'Model':<10} {'Metric':<8} {'Baseline':<10} {'TreeSum':<10} {'Diff':<10}")
    print("-" * 50)
    
    for model_name, model_results in results.items():
        # ROUGE metrics
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            baseline_score = model_results['baseline'][metric]
            treesum_score = model_results['treesum'][metric]
            diff = treesum_score - baseline_score
            print(f"{model_name:<10} {metric.upper():<8} {baseline_score:<10.4f} {treesum_score:<10.4f} {diff:+.4f}")

if __name__ == "__main__":
    main()
