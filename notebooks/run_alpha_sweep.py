
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../results/alpha_sweep')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = 'cpu' # Stay on CPU for stability

def run_alpha_sweep():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}. Run prepare_ablation_data.py first.")
        return
    with open(DATA_PATH, 'r') as f:
        samples = json.load(f)

    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0] # 0 = Pure Lexical, 1 = Pure Semantic
    rouge = evaluate.load('rouge')
    results_table = {}

    for alpha in alpha_values:
        print(f"\n🚀 Testing Alpha = {alpha} (Semantic Weight)")
        # Initialize summarizer with specific alpha
        summarizer = HierarchicalSummarizer(device=DEVICE, semantic_weight=alpha)
        
        predictions = []
        references = []
        
        for item in tqdm(samples):
            doc = item['document']
            ref = item['summary']
            try:
                # We use the full TreeSum (Recursive) strategy for the sweep
                res = summarizer.summarize_document(doc)
                predictions.append(res['final_summary'])
                references.append(ref)
            except Exception as e:
                print(f"Error at Alpha={alpha}: {e}")
                predictions.append("")
                references.append(ref)
        
        # Compute and Store ROUGE
        scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        results_table[f"alpha_{alpha}"] = {k: v * 100 for k, v in scores.items()}
        print(f"Results for Alpha={alpha}: {results_table[f'alpha_{alpha}']}")

        # Save summaries for this variation
        summary_export = []
        for i, item in enumerate(samples):
            summary_export.append({
                "sample_id": item.get('id', i),
                "alpha": alpha,
                "document": item['document'][:500] + "...", # Preview for readability
                "generated_summary": predictions[i],
                "reference_summary": item['summary']
            })
        
        summary_path = os.path.join(OUTPUT_DIR, f"summaries_alpha_{alpha}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_export, f, indent=2)
        print(f"Summaries for Alpha={alpha} saved to {summary_path}")

    # Save Final ROUGE Results
    df = pd.DataFrame(results_table).T
    results_path = os.path.join(OUTPUT_DIR, 'alpha_sweep_results.csv')
    df.to_csv(results_path)
    
    print(f"\n✅ Alpha sweep complete! Results saved to {results_path}")
    print("\nSummary Table (ROUGE-1):")
    print(df[['rouge1', 'rouge2', 'rougeL']])

if __name__ == "__main__":
    run_alpha_sweep()
