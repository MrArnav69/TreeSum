#!/usr/bin/env python3
"""
AlignScore Evaluation Script - A40 GPU Optimized
1000 samples evaluation
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Add AlignScore to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'AlignScore'))

from alignscore import AlignScore

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

def evaluate_alignscore(align_scorer, documents, summaries):
    """Evaluate AlignScore - optimized for A40 GPU with OOM protection."""
    scores = []
    
    # Dynamic batch sizing for OOM protection
    initial_batch_size = 16  # CPU optimized
    min_batch_size = 1
    
    for i in tqdm(range(0, len(documents), initial_batch_size), desc="AlignScore"):
        batch_start = i
        batch_end = min(i + initial_batch_size, len(documents))
        batch_docs = documents[batch_start:batch_end]
        batch_sums = summaries[batch_start:batch_end]
        
        # Try with current batch size, reduce if OOM
        current_batch_size = len(batch_docs)
        success = False
        
        while current_batch_size >= min_batch_size and not success:
            try:
                # Clear GPU cache before processing
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process current batch
                actual_docs = batch_docs[:current_batch_size]
                actual_sums = batch_sums[:current_batch_size]
                
                valid_pairs = [(doc, summary) for doc, summary in zip(actual_docs, actual_sums) 
                             if doc.strip() and summary.strip()]
                
                if not valid_pairs:
                    scores.extend([0.0] * current_batch_size)
                    success = True
                    continue
                
                valid_docs, valid_sums = zip(*valid_pairs)
                
                # AlignScore: context (document) -> claim (summary)
                batch_scores = align_scorer.score(contexts=list(valid_docs), claims=list(valid_sums))
                
                # Pad scores if some pairs were invalid
                while len(batch_scores) < current_batch_size:
                    batch_scores.append(0.0)
                
                scores.extend(batch_scores)
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
                    print(f"AlignScore error: {e}")
                    scores.extend([0.0] * current_batch_size)
                    success = True
            except Exception as e:
                print(f"AlignScore error: {e}")
                scores.extend([0.0] * current_batch_size)
                success = True
        
        if not success:
            # Failed even with minimum batch size
            print(f"⚠️  Failed to process batch {i}-{batch_end}, using zeros")
            scores.extend([0.0] * len(batch_docs))
    
    return scores

def main():
    """Main AlignScore evaluation - A40 GPU optimized."""
    print("🎯 ALIGNSCORE EVALUATION - A40 GPU OPTIMIZED")
    print("="*60)
    print("🔥 Using AlignScore-large (best model)")
    print("📊 Mode: nli_sp (NLI + chunk-sentence splitting)")
    print("📈 Evaluating 1000 samples across 3 datasets")
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
    
    # Initialize AlignScore for A40 GPU
    print("🤖 Initializing AlignScore-large for A40 GPU...")
    try:
        align_scorer = AlignScore(
            model='roberta-large',      # Best model
            batch_size=64,              # Reduced for CPU
            device='cuda',               # CPU due to CUDA compatibility
            ckpt_path='AlignScore-large.ckpt',  # Best checkpoint
            evaluation_mode='nli_sp'    # Best mode (NLI + chunk-sentence)
        )
        print("✅ AlignScore-large loaded successfully for CPU")
    except Exception as e:
        print(f"❌ AlignScore failed to load: {e}")
        print("💡 Make sure you're in environment with torch<2.0")
        return
    
    # Prepare data
    document_map = {sample['sample_id']: sample['document'] for sample in treesum_combined}
    
    datasets_info = {
        'flat_1024': {
            'docs': [document_map[sample['sample_id']] for sample in matched_samples],
            'generated': [sample['flat_1024']['generated_summary'] for sample in matched_samples]
        },
        'flat_overlap': {
            'docs': [document_map[sample['sample_id']] for sample in matched_samples],
            'generated': [sample['flat_overlap']['generated_summary'] for sample in matched_samples]
        },
        'treesum': {
            'docs': [sample['treesum']['document'] for sample in matched_samples],
            'generated': [sample['treesum']['generated_summary'] for sample in matched_samples]
        }
    }
    
    # Evaluate all datasets
    results = {}
    for dataset_name, data in datasets_info.items():
        print(f"\n📊 {dataset_name} ({len(data['docs'])} samples)...")
        
        alignscore_scores = evaluate_alignscore(align_scorer, data['docs'], data['generated'])
        
        results[dataset_name] = {
            'alignscore_scores': alignscore_scores
        }
        
        print(f"   AlignScore: {np.mean(alignscore_scores):.4f}")
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed results
    results_data = []
    for i, sample in enumerate(matched_samples):
        row = {
            'sample_id': sample['sample_id'],
            'flat_1024_alignscore': results['flat_1024']['alignscore_scores'][i],
            'flat_overlap_alignscore': results['flat_overlap']['alignscore_scores'][i],
            'treesum_alignscore': results['treesum']['alignscore_scores'][i]
        }
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    detailed_file = RESULTS_DIR / f"alignscore_detailed_{timestamp}.csv"
    df.to_csv(detailed_file, index=False)
    
    # Save summary statistics
    summary_stats = {}
    for dataset_name in results.keys():
        summary_stats[dataset_name] = {
            'alignscore_mean': np.mean(results[dataset_name]['alignscore_scores']),
            'alignscore_std': np.std(results[dataset_name]['alignscore_scores']),
            'alignscore_min': np.min(results[dataset_name]['alignscore_scores']),
            'alignscore_max': np.max(results[dataset_name]['alignscore_scores']),
            'alignscore_median': np.median(results[dataset_name]['alignscore_scores'])
        }
    
    summary_df = pd.DataFrame(summary_stats).T
    summary_file = RESULTS_DIR / f"alignscore_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file)
    
    print(f"\n✅ Results saved:")
    print(f"   📊 {detailed_file} (detailed)")
    print(f"   📈 {summary_file} (summary)")
    print(f"🎯 Evaluated {len(matched_samples)} samples with AlignScore-large")
    
    # Print detailed summary
    print(f"\n📈 Detailed Summary:")
    for dataset_name, stats in summary_stats.items():
        print(f"\n{dataset_name}:")
        print(f"  Mean: {stats['alignscore_mean']:.4f}")
        print(f"  Std:  {stats['alignscore_std']:.4f}")
        print(f"  Min:  {stats['alignscore_min']:.4f}")
        print(f"  Max:  {stats['alignscore_max']:.4f}")
        print(f"  Med:  {stats['alignscore_median']:.4f}")

if __name__ == "__main__":
    main()
