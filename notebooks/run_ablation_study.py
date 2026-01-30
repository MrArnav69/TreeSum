
import os
import torch
import pandas as pd
import numpy as np
import evaluate
from datasets import load_dataset
from tqdm import tqdm
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from hierarchical_summarizer import HierarchicalSummarizer
from semantic_document_chunker import SemanticDocumentChunker

# Configuration
NUM_SAMPLES = 20  # Small enough for quick iteration, large enough for signal
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../results/ablation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_experiment(mode: str, samples, rouge_metric):
    """
    Run experiment for a specific ablation mode.
    Modes:
    - 'baseline': Vanilla PEGASUS (Truncated input)
    - 'flat_concat': Chunk -> Concatenate -> Summarize (No intermediate summaries)
    - 'map_reduce': Chunk -> Summarize -> Concatenate -> Final Summary (Standard)
    - 'treesum': Chunk -> Summarize -> Recursive Tree Reduction (Ours)
    """
    print(f"\n🚀 Running Mode: {mode.upper()}")
    
    # Initialize Model
    # (Re-initialize to clear cache/weights for fairness)
    summarizer = HierarchicalSummarizer(device=DEVICE)
    
    # Configure Chunker based on mode
    if mode == 'baseline':
        # No chunking logic needed, handled in loop
        pass
    elif mode == 'flat_concat':
        # Standard chunking, but we will purposefully skip stage 2 summarization logic in loop
        pass
    else:
        # Standard Setup for MapReduce and TreeSum
        pass

    predictions = []
    references = []
    
    for item in tqdm(samples, desc=f"{mode}"):
        doc = item['document']
        ref = item['summary']
        
        try:
            if mode == 'baseline':
                # Vanilla Truncation: Just feed raw text to _generate
                # It will truncate to 1024 tokens automatically
                summary = summarizer._generate([doc], max_length=512)[0]
                
            elif mode == 'flat_concat':
                # Naive Hierarchical: Segments -> Cat -> Summarize (Limit Hit likely)
                chunks = summarizer.chunker.chunk_document(doc)
                chunk_texts = [c['text'] for c in chunks]
                concat_text = " ".join(chunk_texts)
                # This effectively tests "Smart Chunking" vs "Raw truncation"
                summary = summarizer._generate([concat_text], max_length=512)[0]
                
            elif mode == 'treesum':
                # Our Full Pipeline
                result = summarizer.summarize_document(doc)
                summary = result['final_summary']
                
            elif mode == 'map_reduce':
                # Standard Map-Reduce (Fixed 2-stage, no recursion)
                # We force the reduce stage to be simple
                chunks = summarizer.chunker.chunk_document(doc)
                chunk_summaries = summarizer._stage1_map_summaries([c['text'] for c in chunks])
                combined = " ".join(chunk_summaries)
                summary = summarizer._generate([combined], max_length=512)[0]
                
            predictions.append(summary)
            references.append(ref)
            
        except Exception as e:
            print(f"Error in {mode}: {e}")
            predictions.append("") # Penalty for failure
            references.append(ref)

    # Compute ROUGE
    scores = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
    return {k: v * 100 for k, v in scores.items()}

def main():
    print(f"Loading Dataset (Seed: {SEED})...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Deterministic subset
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
    samples = [dataset[int(i)] for i in indices]
    
    rouge = evaluate.load('rouge')
    
    results_table = {}
    
    # 1. Baseline (The "Strawman")
    results_table['Baseline (Truncated)'] = run_experiment('baseline', samples, rouge)
    
    # 2. Flat Concat (Testing Chunker Utility)
    results_table['Flat Hierarchical'] = run_experiment('flat_concat', samples, rouge)
    
    # 3. Standard Map-Reduce (Standard Approach)
    results_table['Map-Reduce'] = run_experiment('map_reduce', samples, rouge)
    
    # 4. TreeSum (Our Proposed Method)
    results_table['TreeSum (Recursive)'] = run_experiment('treesum', samples, rouge)
    
    # Save Results
    df = pd.DataFrame(results_table).T
    print("\n=== ABLATION STUDY RESULTS ===")
    print(df[['rouge1', 'rouge2', 'rougeL']])
    
    csv_path = os.path.join(OUTPUT_DIR, 'ablation_results.csv')
    df.to_csv(csv_path)
    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()
