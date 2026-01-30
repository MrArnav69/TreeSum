
import os
import json
import torch
import pandas as pd
import evaluate
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from hierarchical_summarizer import HierarchicalSummarizer

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), '../results/ablation/ablation_data.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../results/ablation')
DEVICE = 'cpu' # Force CPU to avoid MPS hangs on large recursion

def run_experiment_v2():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}. Run prepare_ablation_data.py first.")
        return

    with open(DATA_PATH, 'r') as f:
        samples = json.load(f)

    # Initialize shared model to save memory/load time
    print(f"Initializing model on {DEVICE}...")
    # Use the environment-safe python path if needed, but here we assume the current env is correct
    summarizer = HierarchicalSummarizer(device=DEVICE)
    rouge = evaluate.load('rouge')
    
    results_table = {}
    modes = ['baseline', 'flat_concat', 'map_reduce', 'treesum']
    
    for mode in modes:
        print(f"\n🚀 Processing Mode: {mode.upper()}")
        predictions = []
        references = []
        
        for item in tqdm(samples):
            doc = item['document']
            ref = item['summary']
            
            try:
                if mode == 'baseline':
                    summary = summarizer._generate([doc], max_length=512)[0]
                elif mode == 'flat_concat':
                    chunks = summarizer.chunker.chunk_document(doc)
                    concat_text = " ".join([c['text'] for c in chunks])
                    summary = summarizer._generate([concat_text], max_length=512)[0]
                elif mode == 'map_reduce':
                    chunks = summarizer.chunker.chunk_document(doc)
                    chunk_summaries = summarizer._stage1_map_summaries([c['text'] for c in chunks])
                    summary = summarizer._generate([" ".join(chunk_summaries)], max_length=512)[0]
                elif mode == 'treesum':
                    res = summarizer.summarize_document(doc)
                    summary = res['final_summary']
                
                predictions.append(summary)
                references.append(ref)
            except Exception as e:
                print(f"Error in {mode}: {e}")
                predictions.append("")
                references.append(ref)
        
        scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        results_table[mode] = {k: v * 100 for k, v in scores.items()}
        print(f"Results for {mode}: {results_table[mode]}")

    # Save Results
    df = pd.DataFrame(results_table).T
    df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_results.csv'))
    print(f"\nSaved final results to {os.path.join(OUTPUT_DIR, 'ablation_results.csv')}")

if __name__ == "__main__":
    run_experiment_v2()
