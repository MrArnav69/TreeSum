"""
================================================================================
OFFLINE METRICS EVALUATION FOR ABLATION STUDY
================================================================================

This script reads pre-generated summaries from batch files and computes:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
- BERTScore (Precision, Recall, F1)

Works on any machine (Mac M3, CPU, GPU) - no summarization model needed!

Usage:
    python evaluate_results.py --results_dir results_flat_overlap
    python evaluate_results.py --results_dir results_flat_1024

Author: Arnav Gupta
Date: 2026-02-05
================================================================================
"""

import sys
import subprocess
import os
import json
import argparse
from typing import List, Dict
import time

# ============================================================================
# ENVIRONMENT SETUP (Auto-install dependencies)
# ============================================================================
def setup_environment():
    """Installs missing dependencies."""
    required_packages = [
        "datasets", 
        "evaluate", 
        "rouge_score", 
        "bert_score",
        "torch",
        "transformers"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    # NLTK Data for ROUGE
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)
    
    print("✓ Environment setup complete.\n")

# Run setup
setup_environment()

import evaluate
import torch

# ============================================================================
# LOAD SUMMARIES FROM BATCH FILES
# ============================================================================
def load_summaries_from_batches(results_dir: str) -> List[Dict]:
    """
    Load all summaries from batch files (summaries_batch_1.json, etc.)
    Recursively searches subdirectories if needed.
    """
    all_results = []
    
    # 1. First, search the directory itself
    batch_files = [f for f in os.listdir(results_dir) if f.startswith('summaries_batch_') and f.endswith('.json')]
    
    # 2. If no batches found, look in immediate subdirectories (one level deep)
    if not batch_files:
        print(f"No batch files found in {results_dir}, searching subdirectories...")
        for entry in os.scandir(results_dir):
            if entry.is_dir():
                sub_batches = [f for f in os.listdir(entry.path) if f.startswith('summaries_batch_') and f.endswith('.json')]
                if sub_batches:
                    print(f"Found batches in {entry.path}")
                    results_dir = entry.path # Switch to this directory
                    batch_files = sub_batches
                    break
    
    if not batch_files:
        return []

    # Sort batch files numerically
    batch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for filename in batch_files:
        batch_file = os.path.join(results_dir, filename)
        print(f"Loading {batch_file}...")
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            all_results.extend(batch_data)
    
    print(f"✓ Loaded {len(all_results)} summaries total\n")
    return all_results

# ============================================================================
# COMPUTE METRICS
# ============================================================================
def compute_metrics(results: List[Dict], use_gpu: bool = False) -> Dict:
    """
    Compute ROUGE and BERTScore metrics.
    
    Args:
        results: List of dicts with 'generated_summary' and 'reference_summary'
        use_gpu: Whether to use GPU for BERTScore (Mac M3 will use MPS)
    
    Returns:
        Dict of aggregated metrics
    """
    predictions = [r['generated_summary'] for r in results]
    references = [r['reference_summary'] for r in results]
    
    # 1. ROUGE Scores
    print("[1/2] Computing ROUGE Scores...")
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=references)
    print("✓ ROUGE computed\n")
    
    # 2. BERTScore
    print("[2/2] Computing BERTScore...")
    print("   (This may take 5-10 minutes on CPU)")
    print("   Note: Using CPU to avoid memory issues on Mac M3")
    
    # Force CPU for Mac M3 to avoid OOM (DeBERTa is too large for unified memory)
    device = "cpu"
    print(f"   Using device: {device}")
    
    bertscore = evaluate.load("bertscore")
    
    start_time = time.time()
    bert_results = bertscore.compute(
        predictions=predictions, 
        references=references, 
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        device=device,
        batch_size=16  # Small batches to avoid OOM
    )
    elapsed = time.time() - start_time
    print(f"✓ BERTScore computed in {elapsed/60:.1f} minutes\n")
    
    # 3. Aggregate
    metrics = {
        'num_samples': len(results),
        'rouge1': rouge_results['rouge1'] * 100,
        'rouge2': rouge_results['rouge2'] * 100,
        'rougeL': rouge_results['rougeL'] * 100,
        'rougeLsum': rouge_results['rougeLsum'] * 100,
        'bertscore_precision': sum(bert_results['precision']) / len(bert_results['precision']) * 100,
        'bertscore_recall': sum(bert_results['recall']) / len(bert_results['recall']) * 100,
        'bertscore_f1': sum(bert_results['f1']) / len(bert_results['f1']) * 100,
    }
    
    # Add chunk statistics if available
    if 'num_chunks' in results[0]:
        metrics['avg_chunks_per_doc'] = sum(r['num_chunks'] for r in results) / len(results)
    
    return metrics

# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_metrics(metrics: Dict, results_dir: str, method_name: str):
    """Save metrics to JSON file."""
    output_file = os.path.join(results_dir, f'metrics_{method_name}.json')
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {output_file}\n")

def print_metrics(metrics: Dict, method_name: str):
    """Pretty print metrics."""
    print("=" * 70)
    print(f"FINAL METRICS: {method_name.upper()}")
    print("=" * 70)
    print(f"Samples:         {metrics['num_samples']}")
    print(f"ROUGE-1:         {metrics['rouge1']:.2f}")
    print(f"ROUGE-2:         {metrics['rouge2']:.2f}")
    print(f"ROUGE-L:         {metrics['rougeL']:.2f}")
    print(f"ROUGE-Lsum:      {metrics['rougeLsum']:.2f}")
    print(f"BERTScore P:     {metrics['bertscore_precision']:.2f}")
    print(f"BERTScore R:     {metrics['bertscore_recall']:.2f}")
    print(f"BERTScore F1:    {metrics['bertscore_f1']:.2f}")
    if 'avg_chunks_per_doc' in metrics:
        print(f"Avg Chunks/Doc:  {metrics['avg_chunks_per_doc']:.1f}")
    print("=" * 70)

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate pre-generated summaries')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory containing summaries_batch_*.json files (default: auto-detect)')
    parser.add_argument('--method_name', type=str, default=None,
                        help='Method name for output file (default: auto-detect from dir name)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU/MPS for BERTScore (faster on Mac M3)')
    
    args = parser.parse_args()
    
    # Auto-detect results directory if not specified
    if args.results_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = ['results_flat_overlap', 'results_flat_1024', 'Overlap 1024', 'Flat 1024', 'Flat 1024 Overlap']
        
        for candidate in candidates:
            # 1. Check relative to script's directory (Robust for local runs)
            path_rel_script = os.path.join(script_dir, candidate)
            # 2. Check relative to current CWD (Works if user is already in the folder)
            path_rel_cwd = os.path.abspath(candidate)
            
            for candidate_path in [path_rel_script, path_rel_cwd]:
                if os.path.exists(candidate_path) and os.path.isdir(candidate_path):
                    args.results_dir = candidate_path
                    print(f"Auto-detected results directory: {candidate}")
                    break
            if args.results_dir:
                break
        
        if args.results_dir is None:
            print("❌ No results directory found. Please specify with --results_dir")
            print(f"   Looked for: {', '.join(candidates)}")
            return
    
    # Auto-detect method name from directory
    if args.method_name is None:
        if 'overlap' in args.results_dir.lower():
            args.method_name = 'flat_overlap'
        elif '1024' in args.results_dir:
            args.method_name = 'flat_1024'
        else:
            args.method_name = 'evaluation'
    
    print("=" * 70)
    print("OFFLINE METRICS EVALUATION")
    print("=" * 70)
    print(f"Results Directory: {args.results_dir}")
    print(f"Method Name:       {args.method_name}")
    print(f"Use GPU/MPS:       {args.use_gpu}")
    print("=" * 70)
    print()
    
    # 1. Load summaries
    results = load_summaries_from_batches(args.results_dir)
    
    if len(results) == 0:
        print("❌ No summaries found! Check the results directory.")
        return
    
    # 2. Compute metrics
    metrics = compute_metrics(results, use_gpu=args.use_gpu)
    
    # 3. Save and print
    save_metrics(metrics, args.results_dir, args.method_name)
    print_metrics(metrics, args.method_name)

if __name__ == "__main__":
    main()
