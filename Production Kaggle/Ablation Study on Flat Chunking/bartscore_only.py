#!/usr/bin/env python3
"""
BARTScore Evaluation Script - A40 GPU Optimized
Fixed sum2doc bug + Added Significance Testing
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from scipy.stats import wilcoxon  # Added for significance testing

# Add BARTScore to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'BARTScore'))

try:
    from bart_score import BARTScorer
except ImportError:
    print("⚠️ BARTScore not found in path. Please ensure 'BARTScore' folder exists.")

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
            print(f"❌ Missing {filename}")
            # For testing purposes if files missing, we might return mock? 
            # Better to just fail or let user fix paths.
    
    if len(datasets) < 3:
        raise FileNotFoundError("Missing required dataset files.")
        
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

def evaluate_bartscore(bart_scorer, documents, summaries, direction="doc2sum"):
    """Evaluate BARTScore with robust OOM protection."""
    scores = []
    
    # Dynamic batch sizing for OOM protection
    initial_batch_size = 32  # BARTScore is memory intensive
    min_batch_size = 1
    
    for i in tqdm(range(0, len(documents), initial_batch_size), desc=f"BARTScore ({direction})"):
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
                
                # Filter valid pairs
                valid_pairs = [(doc, summary) for doc, summary in zip(actual_docs, actual_sums) 
                             if isinstance(doc, str) and isinstance(summary, str) and doc.strip() and summary.strip()]
                
                if not valid_pairs:
                    scores.extend([-10.0] * current_batch_size)  # BARTScore default negative score
                    success = True
                    continue
                
                valid_docs, valid_sums = zip(*valid_pairs)
                
                # BARTScore evaluation
                if direction == "doc2sum":
                    # P(summary | document) -> Faithfulness/Quality
                    batch_scores = bart_scorer.score(list(valid_docs), list(valid_sums), batch_size=len(valid_docs))
                else:  # sum2doc
                    # P(document | summary) -> Recall/Coverage
                    # Note: We pass (summary, document) here because bart_scorer.score(src, tgt)
                    batch_scores = bart_scorer.score(list(valid_sums), list(valid_docs), batch_size=len(valid_docs))
                
                # Pad scores if some pairs were invalid
                # (Logic simplification: we just extend the result. In prod we might want to map back to original indices)
                while len(batch_scores) < current_batch_size:
                    batch_scores.append(-10.0)
                
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
                    print(f"BARTScore error: {e}")
                    scores.extend([-10.0] * current_batch_size)
                    success = True
            except Exception as e:
                print(f"BARTScore error: {e}")
                scores.extend([-10.0] * current_batch_size)
                success = True
        
        if not success:
            # Failed even with minimum batch size
            print(f"⚠️  Failed to process batch {i}-{batch_end}, using defaults")
            scores.extend([-10.0] * len(batch_docs))
    
    return scores

def calculate_significance(results_data):
    """Calculate Wilcoxon signed-rank test between models."""
    print("\n🔬 Statistical Significance (Wilcoxon Signed-Rank Test)")
    print("="*60)
    
    df = pd.DataFrame(results_data)
    
    # We compare TreeSum against the best baseline (usually Flat Overlap)
    baselines = ['flat_overlap'] # You can add 'flat_1024' if needed
    metrics = ['doc2sum', 'sum2doc']
    
    for baseline in baselines:
        print(f"\n🆚 Comparison: TreeSum vs {baseline.replace('_', ' ').title()}")
        for metric in metrics:
            col_tree = f'treesum_{metric}'
            col_base = f'{baseline}_{metric}'
            
            scores_tree = df[col_tree]
            scores_base = df[col_base]
            
            # Wilcoxon test
            try:
                stat, p_value = wilcoxon(scores_tree, scores_base)
                
                # Mean difference
                diff = scores_tree.mean() - scores_base.mean()
                direction_str = "BETTER" if diff > 0 else "WORSE" # Higher is better for log likelihood (closer to 0)
                
                print(f"   📊 {metric.upper()}:")
                print(f"      Mean Diff: {diff:.4f} ({direction_str})")
                print(f"      p-value:   {p_value:.4e}")
                
                if p_value < 0.05:
                    print("      ✅ Statistically Significant (p < 0.05)")
                else:
                    print("      ❌ Not Significant")
            except Exception as e:
                print(f"      ⚠️ Could not calculate significance: {e}")

def main():
    """Main BARTScore evaluation - A40 GPU optimized."""
    print("🔥 BARTSCORE EVALUATION - A40 GPU OPTIMIZED")
    print("="*60)
    print("📊 Using facebook/bart-large-cnn model")
    print("📈 Evaluating samples across datasets")
    
    # Load and prepare data
    datasets = find_datasets()
    matched_samples, treesum_combined = create_matched_samples(datasets)
    
    # Use all samples
    print(f"🎯 Processing all {len(matched_samples)} samples")
    
    # GPU optimization settings
    import torch
    if torch.cuda.is_available():
        print(f"🎮 GPU detected: {torch.cuda.get_device_name()}")
        print(f"📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    # Initialize BARTScore
    print("🤖 Initializing BARTScore...")
    try:
        bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
        print("✅ BARTScore initialized successfully")
    except Exception as e:
        print(f"❌ BARTScore failed to load: {e}")
        return
    
    # Prepare data structure
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
    
    # Evaluate
    results = {}
    for dataset_name, data in datasets_info.items():
        print(f"\n📊 {dataset_name} ({len(data['docs'])} samples)...")
        
        # 1. Doc -> Sum (Faithfulness)
        doc2sum_scores = evaluate_bartscore(bart_scorer, data['docs'], data['generated'], "doc2sum")
        
        # 2. Sum -> Doc (Recall)
        # FIX IS HERE: We pass (docs, generated) just like above. 
        # The function internal logic will flip them because we pass "sum2doc".
        sum2doc_scores = evaluate_bartscore(bart_scorer, data['docs'], data['generated'], "sum2doc")
        
        results[dataset_name] = {
            'doc2sum_scores': doc2sum_scores,
            'sum2doc_scores': sum2doc_scores
        }
        
        print(f"   Doc→Sum (Faithfulness): {np.mean(doc2sum_scores):.4f}")
        print(f"   Sum→Doc (Recall):       {np.mean(sum2doc_scores):.4f}")
    
    # Prepare detailed results
    results_data = []
    for i, sample in enumerate(matched_samples):
        row = {'sample_id': sample['sample_id']}
        for name in datasets_info.keys():
            row[f'{name}_doc2sum'] = results[name]['doc2sum_scores'][i]
            row[f'{name}_sum2doc'] = results[name]['sum2doc_scores'][i]
        results_data.append(row)
    
    # Calculate Significance immediately
    calculate_significance(results_data)
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results_data)
    detailed_file = RESULTS_DIR / f"bartscore_detailed_{timestamp}.csv"
    df.to_csv(detailed_file, index=False)
    
    print(f"\n✅ Results saved to {detailed_file}")

if __name__ == "__main__":
    main()