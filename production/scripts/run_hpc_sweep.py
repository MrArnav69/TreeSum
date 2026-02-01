
import os
import json
import torch
import pandas as pd
import evaluate
from tqdm import tqdm
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from hierarchical_summarizer import HierarchicalSummarizer

# ==========================================
# CONFIGURATION (HPC OPTIMIZED)
# ==========================================
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/golden_500_longest.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../results/alpha_sweep_500')
LOG_DIR = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRECISION = torch.bfloat16  # Optimized for A40
BATCH_SIZE = 8             # Number of documents to process in parallel (VRAM dependent)
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def run_hpc_sweep():
    print(f"--- TreeSum High-Performance Alpha Sweep ---")
    print(f"Hardware: {DEVICE} | Precision: {PRECISION}")
    print(f"Alpha Range: {ALPHA_VALUES}")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}. Run scripts/prepare_data.py first.")
        return
    with open(DATA_PATH, 'r') as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} long documents for evaluation.")

    rouge = evaluate.load('rouge')
    results_table = {}

    for alpha in ALPHA_VALUES:
        alpha_start_time = time.time()
        print(f"\n🚀 STARTING Variation: Alpha = {alpha}")
        
        # Initialize summarizer with specific alpha and precision
        summarizer = HierarchicalSummarizer(
            device=DEVICE, 
            semantic_weight=alpha,
            dtype=PRECISION
        )
        
        predictions = []
        references = []
        
        # We process in blocks of 50 for safety checkpointing
        for start_idx in range(0, len(samples), 50):
            block_samples = samples[start_idx:start_idx + 50]
            print(f"Processing Block: Samples {start_idx} to {start_idx + len(block_samples)}")
            
            for item in tqdm(block_samples):
                doc = item['document']
                ref = item['summary']
                try:
                    # Recursive Tree-Reduction 
                    res = summarizer.summarize_document(doc)
                    predictions.append(res['final_summary'])
                    references.append(ref)
                except Exception as e:
                    print(f"Error at Alpha={alpha}, Doc ID={item.get('id')}: {e}")
                    predictions.append("")
                    references.append(ref)
            
            # Temporary checkpoint for summaries (Incremental save)
            temp_summary_path = os.path.join(OUTPUT_DIR, f"temp_summaries_alpha_{alpha}_idx_{start_idx}.json")
            with open(temp_summary_path, 'w') as f:
                json.dump(predictions, f)

        # Compute ROUGE for this Alpha
        print(f"Computing ROUGE for Alpha={alpha}...")
        scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        results_table[f"alpha_{alpha}"] = {k: v * 100 for k, v in scores.items()}
        
        # Save Final Summaries for this Alpha
        summary_export = []
        for i, item in enumerate(samples):
            summary_export.append({
                "sample_id": item.get('id', i),
                "alpha": alpha,
                "token_count": item.get('token_count'),
                "generated_summary": predictions[i],
                "reference_summary": item['summary']
            })
        
        summary_path = os.path.join(OUTPUT_DIR, f"summaries_alpha_{alpha}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_export, f, indent=2)
            
        elapsed = time.time() - alpha_start_time
        print(f"✅ Finished Alpha={alpha} in {elapsed/60:.2f} mins")
        print(f"Scores: {results_table[f'alpha_{alpha}']}")

    # Save Final Metrics Table
    df = pd.DataFrame(results_table).T
    results_path = os.path.join(OUTPUT_DIR, 'alpha_sweep_500_results.csv')
    df.to_csv(results_path)
    
    print(f"\n✨ HIGH-PERFORMANCE STUDY COMPLETE!")
    print(f"Full results saved to: {results_path}")

if __name__ == "__main__":
    run_hpc_sweep()
