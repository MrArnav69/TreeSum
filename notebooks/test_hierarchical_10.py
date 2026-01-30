import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import sys

# Ensure src directory is in path
sys.path.append('/Users/mrarnav69/Documents/TreeSum/src')
from hierarchical_summarizer import HierarchicalSummarizer

def test_hierarchical_10(num_samples: int = 10, device: str = 'cpu'):
    print(f"🚀 Testing Hierarchical Summarizer with {num_samples} samples")
    print(f"   Device:  {device}")
    
    # 1. Load Dataset
    print("Loading Multi-News dataset (Test Split)...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Select random samples to match baseline_metrics.json conditions
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    data_samples = [dataset[int(i)] for i in indices]
    
    # 2. Initialize Model
    print("Initializing Hierarchical Summarizer...")
    summarizer = HierarchicalSummarizer(device=device)
    
    # 3. Load Metrics
    print("Loading ROUGE metric...")
    rouge = evaluate.load('rouge')
    
    results = []
    generated_summaries = []
    reference_summaries = []
    
    # 4. Run Inference
    print("\n⚡ Generating Summaries...")
    for item in tqdm(data_samples, desc="Processing"):
        doc = item['document']
        ref = item['summary']
        
        try:
            output = summarizer.summarize_document(doc)
            gen_summary = output['final_summary']
            
            results.append({
                'reference': ref,
                'generated': gen_summary,
                'num_chunks': len(output['chunks'])
            })
            
            generated_summaries.append(gen_summary)
            reference_summaries.append(ref)
            
        except Exception as e:
            print(f"Error processing doc: {e}")
            continue
            
    # 5. Compute ROUGE
    print("\nComputing ROUGE Scores...")
    metrics = rouge.compute(
        predictions=generated_summaries, 
        references=reference_summaries,
        use_stemmer=True
    )
    
    print("\n" + "="*50)
    print("TEST RESULTS (10 Samples)")
    print("="*50)
    print(f"ROUGE-1: {metrics['rouge1']*100:.2f}")
    print(f"ROUGE-2: {metrics['rouge2']*100:.2f}")
    print(f"ROUGE-L: {metrics['rougeL']*100:.2f}")
    print("="*50)
    
    # Save to results file
    os.makedirs('/Users/mrarnav69/Documents/TreeSum/results', exist_ok=True)
    df = pd.DataFrame(results)
    output_path = '/Users/mrarnav69/Documents/TreeSum/results/test_10_samples.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved test results to '{output_path}'")

if __name__ == "__main__":
    test_hierarchical_10(num_samples=10, device='cpu')
