#!/usr/bin/env python3
"""
Script to check if TreeSum samples match with baseline samples.
Checks if the 2000 TreeSum samples can be found in the baseline files.
"""

import json
from typing import List, Dict, Any, Set

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

def check_sample_matching(treesum_data: List[Dict[str, Any]], 
                         baseline_data: List[Dict[str, Any]], 
                         model_name: str) -> Dict[str, Any]:
    """Check how many TreeSum samples match with baseline samples."""
    
    # Get sample IDs from TreeSum (first 2000)
    treesum_ids = set(item['sample_id'] for item in treesum_data[:2000])
    
    # Get sample IDs from baseline
    baseline_ids = set(item['sample_id'] for item in baseline_data)
    
    # Find matching IDs
    matching_ids = treesum_ids.intersection(baseline_ids)
    
    # Check if documents match for matching IDs
    document_matches = 0
    reference_matches = 0
    
    treesum_dict = {item['sample_id']: item for item in treesum_data[:2000]}
    baseline_dict = {item['sample_id']: item for item in baseline_data}
    
    for sample_id in matching_ids:
        treesum_item = treesum_dict[sample_id]
        baseline_item = baseline_dict[sample_id]
        
        if treesum_item['document'] == baseline_item['document']:
            document_matches += 1
        if treesum_item['reference_summary'] == baseline_item['reference_summary']:
            reference_matches += 1
    
    return {
        'model': model_name,
        'treesum_samples': len(treesum_ids),
        'baseline_samples': len(baseline_ids),
        'matching_ids': len(matching_ids),
        'document_matches': document_matches,
        'reference_matches': reference_matches,
        'matching_percentage': (len(matching_ids) / len(treesum_ids)) * 100,
        'missing_from_treesum': len(treesum_ids - baseline_ids),
        'missing_from_baseline': len(baseline_ids - treesum_ids)
    }

def main():
    # Define file paths
    treesum_files = [
        "/Users/mrarnav69/Documents/TreeSum/results/TreeSum/treesum_summaries_p1.json",
        "/Users/mrarnav69/Documents/TreeSum/results/TreeSum/treesum_summaries_p2.json", 
        "/Users/mrarnav69/Documents/TreeSum/results/TreeSum/treesum_summaries_p3.json",
        "/Users/mrarnav69/Documents/TreeSum/results/TreeSum/treesum_summaries_p4.json"
    ]
    
    baseline_files = {
        "PRIMERA": "/Users/mrarnav69/Documents/TreeSum/results/PRIMERA-multinews/vanilla_primera_summaries_2500.json",
        "PEGASUS": "/Users/mrarnav69/Documents/TreeSum/results/PEGASUS-multi_news/vanilla_pegasus_summaries_2500.json",
        "BART": "/Users/mrarnav69/Documents/TreeSum/results/BART-large-cnn/vanilla_bart_summaries_2500.json"
    }
    
    print("=== Checking Sample Matching ===\n")
    
    # Load and combine TreeSum files
    treesum_data = combine_treesum_files(treesum_files)
    
    # Check matching for each baseline
    results = []
    
    for model_name, file_path in baseline_files.items():
        print(f"\n--- Checking {model_name} ---")
        baseline_data = load_json_file(file_path)
        
        result = check_sample_matching(treesum_data, baseline_data, model_name)
        results.append(result)
        
        print(f"TreeSum samples: {result['treesum_samples']}")
        print(f"{model_name} samples: {result['baseline_samples']}")
        print(f"Matching sample IDs: {result['matching_ids']} ({result['matching_percentage']:.1f}%)")
        print(f"Document matches: {result['document_matches']}")
        print(f"Reference summary matches: {result['reference_matches']}")
        print(f"Missing from TreeSum: {result['missing_from_treesum']}")
        print(f"Missing from {model_name}: {result['missing_from_baseline']}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"{'Model':<10} {'TreeSum':<8} {'Baseline':<8} {'Matching':<8} {'%':<6} {'Docs':<6} {'Refs':<6}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model']:<10} {result['treesum_samples']:<8} {result['baseline_samples']:<8} "
              f"{result['matching_ids']:<8} {result['matching_percentage']:<6.1f} "
              f"{result['document_matches']:<6} {result['reference_matches']:<6}")
    
    # Check if we have at least 2000 matching samples across all baselines
    common_ids = None
    baseline_data_dict = {}
    
    for model_name, file_path in baseline_files.items():
        baseline_data = load_json_file(file_path)
        baseline_ids = set(item['sample_id'] for item in baseline_data)
        
        if common_ids is None:
            common_ids = set(item['sample_id'] for item in treesum_data[:2000])
        common_ids = common_ids.intersection(baseline_ids)
    
    print(f"\nCommon sample IDs across all models: {len(common_ids)}")
    
    if len(common_ids) >= 2000:
        print("✓ Can evaluate on 2000+ matching samples across all baselines")
    else:
        print(f"⚠ Only {len(common_ids)} common samples available for evaluation")

if __name__ == "__main__":
    main()
