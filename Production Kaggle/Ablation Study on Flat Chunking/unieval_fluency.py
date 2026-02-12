#!/usr/bin/env python3
"""
Final Evaluation Script - UniEval Fluency Only
Optimized for A40 GPU - 1000 samples
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Add UniEval to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'UniEval'))

from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def find_datasets():
    """Find datasets."""
    base_dir = Path(".")
    files = {
        'flat_1024': 'summaries_flat_1024.json',
        'flat_overlap': 'summaries_flat_overlap.json',
        'treesum_pt1': 'summaries_treesum_pt1_first_500.json',
        'treesum_pt2': 'summaries_treesum_pt2_last_500.json'
    }
    
    datasets = {}
    for name, filename in files.items():
        file_path = base_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                datasets[name] = json.load(f)
            print(f"✅ {filename}")
        else:
            raise FileNotFoundError(f"❌ Missing {filename}")
    return datasets

def create_matched_samples(datasets):
    """Create matched samples."""
    treesum_combined = datasets['treesum_pt1'] + datasets['treesum_pt2']
    
    flat_1024_map = {item['sample_idx']: item for item in datasets['flat_1024']}
    flat_overlap_map = {item['sample_idx']: item for item in datasets['flat_overlap']}
    treesum_map = {item['sample_id']: item for item in treesum_combined}
    
    common_ids = set(flat_1024_map.keys()) & set(flat_overlap_map.keys()) & set(treesum_map.keys())
    
    matched_samples = []
    for sample_id in sorted(common_ids):
        sample = {
            'sample_id': sample_id,
            'flat_1024': flat_1024_map[sample_id],
            'flat_overlap': flat_overlap_map[sample_id],
            'treesum': treesum_map[sample_id]
        }
        matched_samples.append(sample)
    
    print(f"✅ {len(matched_samples)} matched samples")
    return matched_samples, treesum_combined

def evaluate_unieval(evaluator, summaries):
    """Evaluate UniEval Fluency only - optimized for A40 GPU with OOM protection."""
    dimensions = ['fluency']
    all_scores = {dim: [] for dim in dimensions}
    
    # Dynamic batch sizing for OOM protection
    initial_batch_size = 128  # A40 GPU optimized
    min_batch_size = 1
    
    for i in tqdm(range(0, len(summaries), initial_batch_size), desc="UniEval Fluency"):
        batch_start = i
        batch_end = min(i + initial_batch_size, len(summaries))
        batch_sums = summaries[batch_start:batch_end]
        
        # Try with current batch size, reduce if OOM
        current_batch_size = len(batch_sums)
        success = False
        
        while current_batch_size >= min_batch_size and not success:
            try:
                # Clear GPU cache before processing
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process current batch
                actual_batch = batch_sums[:current_batch_size]
                filtered_sums = [s if s.strip() else "Empty summary" for s in actual_batch]
                data = convert_to_json(output_list=filtered_sums)
                
                for dim in dimensions:
                    try:
                        eval_scores = evaluator.evaluate(data, dims=[dim], overall=False, print_result=False)
                        
                        if isinstance(eval_scores, list) and len(eval_scores) > 0:
                            dim_scores = [sample_scores.get(dim, 0.0) for sample_scores in eval_scores]
                            all_scores[dim].extend(dim_scores)
                        else:
                            all_scores[dim].extend([0.0] * len(filtered_sums))
                            
                    except Exception as e:
                        print(f"UniEval {dim} error: {e}")
                        all_scores[dim].extend([0.0] * len(filtered_sums))
                
                success = True
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                    # Reduce batch size and retry
                    current_batch_size = current_batch_size // 2
                    print(f"⚠️  OOM detected, reducing batch size to {current_batch_size}")
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.clear_cache()
                        torch.cuda.empty_cache()
                else:
                    # Different error, use fallback
                    print(f"UniEval batch error: {e}")
                    for dim in dimensions:
                        all_scores[dim].extend([0.0] * current_batch_size)
                    success = True
            except Exception as e:
                print(f"UniEval batch error: {e}")
                for dim in dimensions:
                    all_scores[dim].extend([0.0] * current_batch_size)
                success = True
        
        if not success:
            # Failed even with minimum batch size
            print(f"⚠️  Failed to process batch {i}-{batch_end}, using zeros")
            for dim in dimensions:
                all_scores[dim].extend([0.0] * len(batch_sums))
    
    return all_scores

def main():
    """Main UniEval Fluency evaluation - A40 GPU optimized."""
    print("🚀 UNIEVAL FLUENCY EVALUATION - A40 GPU OPTIMIZED")
    print("="*60)
    print("📊 Evaluating 1000 samples across 3 datasets")
    print("🔥 No token limits + OOM protection enabled")
    
    # Load and prepare data
    datasets = find_datasets()
    matched_samples, treesum_combined = create_matched_samples(datasets)
    
    # Use all 1000 samples
    print(f"🎯 Processing all {len(matched_samples)} samples")
    
    # GPU optimization settings
    import torch
    if torch.cuda.is_available():
        print(f"🎮 GPU detected: {torch.cuda.get_device_name()}")
        print(f"📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Enable memory optimization
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Initialize UniEval for A40 GPU
    print("🤖 Initializing UniEval for A40 GPU...")
    unieval_evaluator = get_evaluator('summarization', device='cuda', max_length=16384)  # No token limit
    print("✅ UniEval initialized for GPU acceleration")
    
    # Prepare data
    document_map = {sample['sample_id']: sample['document'] for sample in treesum_combined}
    
    datasets_info = {
        'flat_1024': {
            'generated': [sample['flat_1024']['generated_summary'] for sample in matched_samples]
        },
        'flat_overlap': {
            'generated': [sample['flat_overlap']['generated_summary'] for sample in matched_samples]
        },
        'treesum': {
            'generated': [sample['treesum']['generated_summary'] for sample in matched_samples]
        }
    }
    
    # Evaluate all datasets
    results = {}
    for dataset_name, data in datasets_info.items():
        print(f"\n📊 {dataset_name} ({len(data['generated'])} samples)...")
        
        # UniEval Fluency
        unieval_scores = evaluate_unieval(unieval_evaluator, data['generated'])
        
        results[dataset_name] = {
            'fluency_scores': unieval_scores['fluency']
        }
        
        print(f"   Fluency: {np.mean(unieval_scores['fluency']):.4f}")
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed results
    results_data = []
    for i, sample in enumerate(matched_samples):
        row = {
            'sample_id': sample['sample_id'],
            'flat_1024_fluency': results['flat_1024']['fluency_scores'][i],
            'flat_overlap_fluency': results['flat_overlap']['fluency_scores'][i],
            'treesum_fluency': results['treesum']['fluency_scores'][i]
        }
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    detailed_file = RESULTS_DIR / f"unieval_fluency_detailed_{timestamp}.csv"
    df.to_csv(detailed_file, index=False)
    
    # Summary statistics
    summary_stats = {}
    for dataset_name in results.keys():
        summary_stats[dataset_name] = {
            'fluency_mean': np.mean(results[dataset_name]['fluency_scores']),
            'fluency_std': np.std(results[dataset_name]['fluency_scores']),
            'fluency_min': np.min(results[dataset_name]['fluency_scores']),
            'fluency_max': np.max(results[dataset_name]['fluency_scores']),
            'fluency_median': np.median(results[dataset_name]['fluency_scores'])
        }
    
    summary_df = pd.DataFrame(summary_stats).T
    summary_file = RESULTS_DIR / f"unieval_fluency_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file)
    
    print(f"\n✅ Results saved:")
    print(f"   📊 {detailed_file} (detailed)")
    print(f"   📈 {summary_file} (summary)")
    print(f"🎯 Evaluated {len(matched_samples)} samples with UniEval Fluency")
    
    # Print detailed summary
    print(f"\n📈 Detailed Summary:")
    for dataset_name, stats in summary_stats.items():
        print(f"\n{dataset_name}:")
        print(f"  Mean: {stats['fluency_mean']:.4f}")
        print(f"  Std:  {stats['fluency_std']:.4f}")
        print(f"  Min:  {stats['fluency_min']:.4f}")
        print(f"  Max:  {stats['fluency_max']:.4f}")
        print(f"  Med:  {stats['fluency_median']:.4f}")

if __name__ == "__main__":
    main()
