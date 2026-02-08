
import sys
import os
import torch
import json
import time
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate

# Add src to path to use the updated library
sys.path.append("/Users/mrarnav69/Documents/TreeSum/src")
from hierarchical_summarizer import HierarchicalSummarizer

def run_verification():
    print("="*60)
    print("TREESUM 2.1 SOTA VERIFICATION (5 SAMPLES)")
    print("="*60)
    
    # 1. Load Dataset
    print("\n[1/3] Loading Multi-News (Parquet)...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    samples = dataset.select(range(5))
    
    # SOTA Push: Use Context-Aware Mode and Alpha=1.0
    device = 'cpu'
    print(f"\n[2/3] Initializing SOTA Summarizer (Context=ON) on {device}...")
    summarizer = HierarchicalSummarizer(device=device, semantic_weight=1.0, context_aware=True)
    
    # 3. Process Samples
    print("\n[3/3] Generating Summaries...")
    all_refs = []
    all_preds = []
    
    start_time = time.time()
    for i, sample in enumerate(tqdm(samples)):
        doc = sample['document']
        ref = sample['summary']
        
        try:
            output = summarizer.summarize_document(doc)
            pred = output['final_summary']
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            pred = ""
            
        all_refs.append(ref)
        all_preds.append(pred)
        
    total_time = time.time() - start_time
    
    # 4. Compute Metrics
    print("\nComputing ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=all_preds, references=all_refs)
    
    print("Computing BERTScore (DeBERTa-XLarge, CPU)...")
    # We use CPU for BERTScore to avoid OOM on Mac unified memory 
    bertscore = evaluate.load("bertscore")
    bert_res = bertscore.compute(
        predictions=all_preds, references=all_refs, lang="en", 
        model_type="microsoft/deberta-xlarge-mnli", device="cpu", batch_size=4
    )
    
    metrics = {
        'rouge1': rouge_res['rouge1'] * 100,
        'rouge2': rouge_res['rouge2'] * 100,
        'rougeL': rouge_res['rougeL'] * 100,
        'bertscore_f1': np.mean(bert_res['f1']) * 100,
        'avg_time_per_doc': total_time / 5
    }
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS (n=5):")
    print("="*60)
    print(json.dumps(metrics, indent=2))
    print("="*60)
    
    # Save to file
    with open('local_verification_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    run_verification()
