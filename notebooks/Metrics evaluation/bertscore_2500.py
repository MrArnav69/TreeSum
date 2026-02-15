#!/usr/bin/env python3
"""
BERTScore Evaluation - Ultra-Conservative Fix with Multiple Safety Layers
This version uses aggressive truncation and processes samples individually if needed
"""
import json
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import torch
from bert_score import score as bert_score
from transformers import AutoTokenizer

results_dir = Path("results_metrics")
results_dir.mkdir(exist_ok=True)

def find_datasets():
    """Load all dataset files"""
    files = {
        'PRIMERA': "/workspace/TreeSum/results/PRIMERA-multinews/vanilla_primera_summaries_2500.json",
        'PEGASUS': "/workspace/TreeSum/results/PEGASUS-multi_news/vanilla_pegasus_summaries_2500.json",
        'BART': "/workspace/TreeSum/results/BART-large-cnn/vanilla_bart_summaries_2500.json",
        'TreeSum_p1': "/workspace/TreeSum/results/TreeSum/treesum_summaries_p1.json",
        'TreeSum_p2': "/workspace/TreeSum/results/TreeSum/treesum_summaries_p2.json",
        'TreeSum_p3': "/workspace/TreeSum/results/TreeSum/treesum_summaries_p3.json",
        'TreeSum_p4': "/workspace/TreeSum/results/TreeSum/treesum_summaries_p4.json",
        'TreeSum_p5': "/workspace/TreeSum/results/TreeSum/treesum_summaries_p5.json"
    }
    
    datasets = {}
    for name, file_path in files.items():
        file_path = Path(file_path)
        if file_path.exists():
            with open(file_path, 'r') as f:
                datasets[name] = json.load(f)
            print(f"✅ Loaded {name}: {len(datasets[name])} samples")
        else:
            raise FileNotFoundError(f"❌ Missing {file_path}")
    return datasets

def create_matched_samples(datasets):
    """Create matched samples across all models"""
    treesum_combined = (datasets['TreeSum_p1'] + datasets['TreeSum_p2'] + 
                       datasets['TreeSum_p3'] + datasets['TreeSum_p4'] + datasets['TreeSum_p5'])
    
    print(f"TreeSum combined: {len(treesum_combined)} samples")
    
    primera_map = {item['sample_id']: item for item in datasets['PRIMERA']}
    pegasus_map = {item['sample_id']: item for item in datasets['PEGASUS']}
    bart_map = {item['sample_id']: item for item in datasets['BART']}
    treesum_map = {item['sample_id']: item for item in treesum_combined}
    
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
    
    print(f"\n✅ Found {len(matched_samples)} matching samples across all models")
    return matched_samples

def ultra_safe_truncate(text, tokenizer, max_tokens=200, max_chars=800):
    """
    Multi-layer truncation with safety checks
    1. Token-level truncation
    2. Character-level safety check
    3. Fallback to character truncation if tokenization fails
    """
    try:
        # Layer 1: Token-level truncation
        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens
        )
        truncated = tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Layer 2: Character-level safety check
        if len(truncated) > max_chars:
            truncated = truncated[:max_chars]
        
        return truncated
    
    except Exception as e:
        # Layer 3: Fallback to simple character truncation
        return text[:max_chars]

def evaluate_bertscore_sample_by_sample(predictions, references, model_type, device):
    """
    Process samples one at a time to isolate problematic samples
    """
    print("  Processing samples individually (failsafe mode)...")
    P_scores = []
    R_scores = []
    F1_scores = []
    
    for i, (pred, ref) in enumerate(tqdm(list(zip(predictions, references)), desc="  Computing")):
        try:
            P, R, F1 = bert_score(
                [pred],  # Single sample
                [ref],
                model_type=model_type,
                device=device,
                batch_size=1,
                lang='en',
                verbose=False,
                rescale_with_baseline=False,
                idf=False
            )
            P_scores.append(P.item() * 100)
            R_scores.append(R.item() * 100)
            F1_scores.append(F1.item() * 100)
        except Exception as e:
            print(f"  ⚠️  Sample {i} failed: {e}, using zeros")
            P_scores.append(0.0)
            R_scores.append(0.0)
            F1_scores.append(0.0)
    
    return P_scores, R_scores, F1_scores

def evaluate_bertscore(samples, model_name, model_type="microsoft/deberta-xlarge-mnli", 
                       device='cuda', batch_size=16, max_tokens=200):
    """
    Evaluate BERTScore with ultra-conservative settings
    """
    print(f"\nEvaluating BERTScore for {model_name}...")
    print(f"Model: {model_type}")
    print(f"Batch size: {batch_size}")
    print(f"Max tokens per text: {max_tokens} (ultra-conservative)")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    predictions = []
    references = []
    valid_indices = []
    
    # Collect and truncate
    print("Collecting and truncating texts...")
    for i, sample in enumerate(samples):
        pred = sample[model_name]['generated_summary']
        ref = sample[model_name]['reference_summary']
        
        if pred.strip() and ref.strip():
            predictions.append(pred)
            references.append(ref)
            valid_indices.append(i)
    
    print(f"Valid samples: {len(predictions)}/{len(samples)}")
    
    if not predictions:
        return {
            'precision': [0.0] * len(samples),
            'recall': [0.0] * len(samples),
            'f1': [0.0] * len(samples)
        }
    
    # Ultra-safe truncation with multiple safety layers
    print("Applying ultra-safe truncation...")
    max_chars = max_tokens * 4  # Character safety limit
    predictions_truncated = [ultra_safe_truncate(p, tokenizer, max_tokens, max_chars) for p in predictions]
    references_truncated = [ultra_safe_truncate(r, tokenizer, max_tokens, max_chars) for r in references]
    
    # Verify truncation
    pred_lens = [len(p) for p in predictions_truncated]
    ref_lens = [len(r) for r in references_truncated]
    print(f"After truncation - Pred: {min(pred_lens)}-{max(pred_lens)} chars, "
          f"Ref: {min(ref_lens)}-{max(ref_lens)} chars")
    
    # Try batch processing first
    print("Computing BERTScore...")
    try:
        P, R, F1 = bert_score(
            predictions_truncated,
            references_truncated,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            lang='en',
            verbose=True,
            rescale_with_baseline=False,
            idf=False
        )
        P_scores = (P.cpu().numpy() * 100).tolist()
        R_scores = (R.cpu().numpy() * 100).tolist()
        F1_scores = (F1.cpu().numpy() * 100).tolist()
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        print("  Switching to sample-by-sample processing...")
        
        # Fallback to one-by-one processing
        P_scores, R_scores, F1_scores = evaluate_bertscore_sample_by_sample(
            predictions_truncated,
            references_truncated,
            model_type,
            device
        )
    
    # Create full arrays
    result_precision = [0.0] * len(samples)
    result_recall = [0.0] * len(samples)
    result_f1 = [0.0] * len(samples)
    
    for idx, valid_idx in enumerate(valid_indices):
        result_precision[valid_idx] = P_scores[idx]
        result_recall[valid_idx] = R_scores[idx]
        result_f1[valid_idx] = F1_scores[idx]
    
    valid_f1 = [s for s in result_f1 if s > 0]
    if valid_f1:
        print(f"✅ Mean F1: {np.mean(valid_f1):.2f} ({len(valid_f1)} samples)")
    
    return {
        'precision': result_precision,
        'recall': result_recall,
        'f1': result_f1
    }

def main():
    print("=" * 80)
    print("BERTScore Evaluation - Ultra-Conservative with Failsafe")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n" + "=" * 80)
    print("Loading Datasets")
    print("=" * 80)
    datasets = find_datasets()
    
    print("\n" + "=" * 80)
    print("Creating Matched Samples")
    print("=" * 80)
    matched_samples = create_matched_samples(datasets)
    
    # Ultra-conservative settings
    BATCH_SIZE = 16  # Smaller batch
    MAX_TOKENS = 200  # Very conservative token limit
    MODEL_TYPE = "microsoft/deberta-xlarge-mnli"
    
    print(f"\n🔧 Ultra-Conservative Configuration:")
    print(f"   Model: {MODEL_TYPE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max tokens: {MAX_TOKENS}")
    print(f"   Max chars: {MAX_TOKENS * 4}")
    print(f"   Failsafe: Sample-by-sample processing if batch fails")
    
    print("\n" + "=" * 80)
    print("Evaluating Models")
    print("=" * 80)
    
    models = ['PRIMERA', 'PEGASUS', 'BART', 'TreeSum']
    all_results = {}
    
    for model in models:
        start_time = datetime.now()
        scores = evaluate_bertscore(
            matched_samples,
            model,
            model_type=MODEL_TYPE,
            device=str(device),
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TOKENS
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"⏱️  Completed in {elapsed:.1f} seconds\n")
        
        all_results[model] = scores
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("Compiling Results")
    print("=" * 80)
    
    detailed_results = []
    for i, sample in enumerate(matched_samples):
        row = {'sample_id': sample['sample_id']}
        for model in models:
            row[f'{model}_bertscore_precision'] = all_results[model]['precision'][i]
            row[f'{model}_bertscore_recall'] = all_results[model]['recall'][i]
            row[f'{model}_bertscore_f1'] = all_results[model]['f1'][i]
        detailed_results.append(row)
    
    summary_data = []
    for model in models:
        model_scores = all_results[model]
        valid_precision = [s for s in model_scores['precision'] if s > 0]
        valid_recall = [s for s in model_scores['recall'] if s > 0]
        valid_f1 = [s for s in model_scores['f1'] if s > 0]
        
        summary_data.append({
            'model': model,
            'n_samples': len(valid_f1),
            'bertscore_precision_mean': np.mean(valid_precision) if valid_precision else 0,
            'bertscore_precision_std': np.std(valid_precision) if valid_precision else 0,
            'bertscore_precision_median': np.median(valid_precision) if valid_precision else 0,
            'bertscore_recall_mean': np.mean(valid_recall) if valid_recall else 0,
            'bertscore_recall_std': np.std(valid_recall) if valid_recall else 0,
            'bertscore_recall_median': np.median(valid_recall) if valid_recall else 0,
            'bertscore_f1_mean': np.mean(valid_f1) if valid_f1 else 0,
            'bertscore_f1_std': np.std(valid_f1) if valid_f1 else 0,
            'bertscore_f1_median': np.median(valid_f1) if valid_f1 else 0,
            'bertscore_f1_min': np.min(valid_f1) if valid_f1 else 0,
            'bertscore_f1_max': np.max(valid_f1) if valid_f1 else 0
        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('model')
    summary_file = results_dir / f"bertscore_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file)
    print(f"\n✅ Summary saved: {summary_file}")
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df = detailed_df.set_index('sample_id')
    detailed_file = results_dir / f"bertscore_detailed_{timestamp}.csv"
    detailed_df.to_csv(detailed_file)
    print(f"✅ Detailed results saved: {detailed_file}")
    
    print("\n" + "=" * 80)
    print("BERTSCORE SUMMARY")
    print("=" * 80)
    print(summary_df.to_string())
    print("\n" + "=" * 80)
    
    print("\nModel Ranking by F1 Score:")
    ranking = summary_df.sort_values('bertscore_f1_mean', ascending=False)
    for i, (model, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {model}: {row['bertscore_f1_mean']:.2f} (±{row['bertscore_f1_std']:.2f})")

if __name__ == "__main__":
    main()