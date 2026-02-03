
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
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/golden_100_random.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../results/alpha_sweep_final')
LOG_DIR = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRECISION = torch.float32  # Safety first: float32 avoids all 'word salad' precision issues
BATCH_SIZE = 16            # Reduced slightly for better stability
ALPHA_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] # High-fidelity sweep

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
    bertscore = evaluate.load('bertscore')

    results_path = os.path.join(OUTPUT_DIR, 'alpha_sweep_200_results.csv')
    if os.path.exists(results_path):
        print(f"Resuming from existing results file: {results_path}")
        df_existing = pd.read_csv(results_path, index_col=0)
        results_table = df_existing.to_dict(orient='index')
    else:
        results_table = {}
    
    # Filter alpha values to run
    alphas_to_run = [a for a in ALPHA_VALUES if f"alpha_{a}" not in results_table]
    print(f"Remaining Alpha Range to process: {alphas_to_run}")

    for alpha in alphas_to_run:
        alpha_start_time = time.time()
        print(f"\n🚀 STARTING Variation: Alpha = {alpha}")
        
        # Initialize summarizer with specific alpha and precision
        summarizer = HierarchicalSummarizer(
            device=DEVICE, 
            semantic_weight=alpha,
            dtype=PRECISION,
            batch_size=BATCH_SIZE
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
            
            # Incremental checkpoint for this alpha
            checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            temp_path = os.path.join(checkpoint_dir, f"temp_{alpha}_{start_idx}.json")
            with open(temp_path, 'w') as f:
                json.dump(predictions, f)

        # 3. Compute Metrics
        print(f"Computing ROUGE for Alpha={alpha}...")
        r_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        
        print(f"Computing BERTScore for Alpha={alpha}...")
        b_scores = bertscore.compute(predictions=predictions, references=references, lang="en", device=DEVICE)
        
        # Aggregate results
        results_table[f"alpha_{alpha}"] = {
            # ROUGE
            "rouge1": r_scores['rouge1'] * 100,
            "rouge2": r_scores['rouge2'] * 100,
            "rougeL": r_scores['rougeL'] * 100,
            # BERTScore (F1 is the standard reported metric)
            "bert_precision": sum(b_scores['precision']) / len(b_scores['precision']) * 100,
            "bert_recall": sum(b_scores['recall']) / len(b_scores['recall']) * 100,
            "bert_f1": sum(b_scores['f1']) / len(b_scores['f1']) * 100
        }
        
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
            
        # 4. Human-Readable Export (Structured for Review)
        txt_path = os.path.join(OUTPUT_DIR, f"report_alpha_{alpha}.txt")
        with open(txt_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f" TREESUM PRODUCTION REPORT | ALPHA Variation: {alpha}\n")
            f.write("="*60 + "\n\n")
            for entry in summary_export:
                f.write(f"📄 DOCUMENT ID: {entry['sample_id']}\n")
                f.write(f"📏 SOURCE LENGTH: {entry['token_count']} tokens\n")
                f.write(f"✨ GENERATED SUMMARY:\n{entry['generated_summary']}\n")
                f.write(f"📖 REFERENCE (GOLD):\n{entry['reference_summary'][:300]}...\n")
                f.write("\n" + "-"*60 + "\n\n")
            
        # Update and Save Results Table (Incremental Save)
        elapsed = time.time() - alpha_start_time
        print(f"✅ Finished Alpha={alpha} in {elapsed/60:.2f} mins")
        print(f"Scores: {results_table[f'alpha_{alpha}']}")

    # Save Final Metrics Table
    # 2. Check for existing results (Smart Resume)
    results_path = os.path.join(OUTPUT_DIR, 'alpha_sweep_100_results.csv')
    df = pd.DataFrame(results_table).T
    df.to_csv(results_path)
    
    print(f"\n✨ HIGH-PERFORMANCE STUDY COMPLETE!")
    print(f"Full results saved to: {results_path}")

if __name__ == "__main__":
    run_hpc_sweep()
