
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
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '../results/ablation/ablation_checkpoint.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../results/ablation')
DEVICE = 'cpu' # Stay on CPU for stability

def run_treesum_only():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}.")
        return
    with open(DATA_PATH, 'r') as f:
        samples = json.load(f)

    # 2. Load Existing Scores
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}.")
        results_table = {}
    else:
        with open(CHECKPOINT_PATH, 'r') as f:
            results_table = json.load(f)

    # 3. Initialize Model
    print(f"Initializing model for TREESUM (Ours) on {DEVICE}...")
    summarizer = HierarchicalSummarizer(device=DEVICE)
    rouge = evaluate.load('rouge')
    
    # 4. Process TreeSum
    print(f"\n🚀 Processing Mode: TREESUM (Recursive Tree-Reduction)")
    predictions = []
    references = []
    
    for item in tqdm(samples):
        doc = item['document']
        ref = item['summary']
        try:
            res = summarizer.summarize_document(doc)
            predictions.append(res['final_summary'])
            references.append(ref)
        except Exception as e:
            print(f"Error in treesum: {e}")
            predictions.append("")
            references.append(ref)
    
    scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    results_table['treesum'] = {k: v * 100 for k, v in scores.items()}
    print(f"\nFinal Results for TreeSum: {results_table['treesum']}")

    # 5. Save Final CSV
    df = pd.DataFrame(results_table).T
    results_path = os.path.join(OUTPUT_DIR, 'ablation_results.csv')
    df.to_csv(results_path)
    print(f"\nSuccessfully completed ablation study!")
    print(f"Final results table saved to: {results_path}")
    print("\nSummary Table:")
    print(df[['rouge1', 'rouge2', 'rougeL']])

if __name__ == "__main__":
    run_treesum_only()
