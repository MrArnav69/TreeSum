
import os
import sys
import json
import torch
import time
import pandas as pd
from tqdm import tqdm
import subprocess
from typing import List, Dict, Optional

"""
================================================================================
FINAL CONSOLIDATED EVALUATOR (A40 PRODUCTION)
================================================================================
Goal: Merge TreeSum (Part 1/2), Flat 1024, and Flat Overlap into one aligned set.
Metrics: SummaC (Faithfulness) and UniEval (Multi-dimensional quality).
Runtime: Optimized for A40 GPU (Cuda).
================================================================================
"""

# ==========================================
# 1. SETUP & DEPENDENCIES
# ==========================================
def setup_eval_env():
    """Install required libraries for SummaC and UniEval."""
    packages = ["summac", "transformers", "torch", "nltk", "pandas", "tqdm"]
    for p in packages:
        try:
            __import__(p)
        except ImportError:
            print(f"Installing {p}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
    
    import nltk
    nltk.download('punkt', quiet=True)

setup_eval_env()

from transformers import AutoTokenizer, T5ForConditionalGeneration
from summac.model_summac import SummaCConv

# ==========================================
# 2. CONSOLIDATION & ALIGNMENT
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILES = {
    "treesum_p1": "treesum_part1_results/summaries_alpha_1.0.json",
    "treesum_p2": "treesum_part2_results/summaries_alpha_1.0.json",
    "flat_1024": "Flat 1024/results_flat_1024/summaries_flat_1024.json",
    "flat_overlap": "Flat 1024 Overlap/results_flat_overlap/summaries_flat_overlap.json",
}

MASTER_INDICES_PATH = "shared_sample_indices.json"

def consolidate_and_align():
    print("Consolidating and aligning result sets...")
    
    # Load Master Indices
    with open(MASTER_INDICES_PATH, 'r') as f:
        master_indices = json.load(f)
    print(f"Master index loaded: {len(master_indices)} samples.")

    # Load All Summaries
    results = {}
    for key, path in INPUT_FILES.items():
        full_path = os.path.join(BASE_DIR, path)
        if not os.path.exists(full_path):
            print(f"WARNING: File missing: {full_path}")
            results[key] = {}
        else:
            with open(full_path, 'r') as f:
                data = json.load(f)
                results[key] = {item['sample_id']: item for item in data}
            print(f"Loaded {key}: {len(results[key])} samples.")

    # Merge TreeSum
    treesum_merged = {**results["treesum_p1"], **results["treesum_p2"]}
    print(f"Merged TreeSum: {len(treesum_merged)} samples.")

    # Align
    aligned_data = []
    missing_count = 0
    
    for sid in master_indices:
        if sid in treesum_merged and sid in results["flat_1024"] and sid in results["flat_overlap"]:
            item = {
                "sample_id": sid,
                "document": treesum_merged[sid].get("document", ""),
                "reference": treesum_merged[sid].get("reference_summary", ""),
                "summary_treesum": treesum_merged[sid].get("generated_summary", ""),
                "summary_flat_1024": results["flat_1024"][sid].get("generated_summary", ""),
                "summary_flat_overlap": results["flat_overlap"][sid].get("generated_summary", ""),
            }
            aligned_data.append(item)
        else:
            missing_count += 1

    print(f"Alignment Complete. Final Aligned Set: {len(aligned_data)} samples.")
    if missing_count > 0:
        print(f"WARNING: {missing_count} samples were missing summaries and were skipped.")
        
    return aligned_data

# ==========================================
# 3. ADVANCED METRIC EVALUATOR
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class AdvancedEvaluator:
    def __init__(self):
        print("Loading SummaC model...")
        # SummaCConv internally batches sentence-level NLI
        self.summac_model = SummaCConv(models=["vitaminc"], bins='percentile', granularity="sentence", device=DEVICE)
        
        print("Loading UniEval model (MingZhong/UniEval-summarization)...")
        self.unieval_tokenizer = AutoTokenizer.from_pretrained("MingZhong/UniEval-summarization")
        self.unieval_model = T5ForConditionalGeneration.from_pretrained("MingZhong/UniEval-summarization").to(DEVICE)
        
        # Token IDs for 'Yes' (4273) and 'No' (150) in T5-based UniEval
        self.pos_id = 4273
        self.neg_id = 150

    def score_summac_batch(self, sources: List[str], summaries: List[str]) -> List[float]:
        """Compute SummaC consistency scores for a list of pairs."""
        if not summaries: return []
        try:
            # SummaC's score() takes two lists of the same length
            res = self.summac_model.score(sources, summaries)
            return [float(s) for s in res["scores"]]
        except Exception as e:
            print(f"SummaC Batch Error: {e}")
            return [0.0] * len(summaries)

    def score_unieval_batch(self, pairs: List[Dict[str, str]], dimensions: List[str]) -> List[Dict[str, float]]:
        """Compute UniEval scores in batches for efficiency."""
        all_results = [{} for _ in range(len(pairs))]
        
        for dim in dimensions:
            # UniEval prompt format: question: [dim] content: [source] summary: [summary]
            prompts = [f"question: {dim} content: {p['source'][:2000]} summary: {p['summary']}" for p in pairs]
            
            # Use a smaller sub-batching inside if the list is too long, but A40 handles 100 easily.
            inputs = self.unieval_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
            
            with torch.no_grad():
                # T5 decoder_input_ids=[0] forces the model to predict the first token (the answer)
                outputs = self.unieval_model(**inputs, decoder_input_ids=torch.zeros((len(pairs), 1), dtype=torch.long).to(DEVICE))
                logits = outputs.logits[:, 0, :] # [Batch, Vocab]
                
                # Extract probabilities for 'Yes' and 'No'
                relevant_logits = logits[:, [self.pos_id, self.neg_id]]
                probs = torch.softmax(relevant_logits, dim=-1)
                yes_probs = probs[:, 0].cpu().tolist()
                
                for i, prob in enumerate(yes_probs):
                    all_results[i][dim] = prob
                    
        return all_results

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    aligned_data = consolidate_and_align()
    if not aligned_data:
        print("No data to evaluate. exiting.")
        return

    evaluator = AdvancedEvaluator()
    results = []
    
    # Process in batches for UniEval/SummaC efficiency
    # For A40 (48GB), 32 is a very safe and efficient batch size. 
    # Since each sample has 3 summaries, this means 96 summaries per GPU pass.
    BATCH_SIZE = 32 
    
    print(f"\nStarting Advanced Evaluation (Batch Size: {BATCH_SIZE}, Device: {DEVICE})...")
    for i in tqdm(range(0, len(aligned_data), BATCH_SIZE), desc="Overall Progress"):
        batch = aligned_data[i : i + BATCH_SIZE]
        
        # 1. Prepare flat lists for batch inference
        batch_sources = []
        batch_summaries = []
        batch_mapping = [] # List of (method_name, original_index)
        
        for idx, item in enumerate(batch):
            source = item["document"]
            for m in ["treesum", "flat_1024", "flat_overlap"]:
                summary = item[f"summary_{m}"]
                batch_sources.append(source)
                batch_summaries.append(summary)
                batch_mapping.append((m, idx))
        
        # 2. Batch SummaC Scoring
        batch_summac_scores = evaluator.score_summac_batch(batch_sources, batch_summaries)
        
        # 3. Batch UniEval Scoring
        uni_pairs = [{"source": s, "summary": sum_} for s, sum_ in zip(batch_sources, batch_summaries)]
        batch_uni_scores = evaluator.score_unieval_batch(uni_pairs, ["coherence", "consistency", "fluency", "relevance"])
        
        # 4. Map results back to aligned_data structures
        batch_res_items = [{"sample_id": item["sample_id"]} for item in batch]
        
        for k, (m, batch_idx) in enumerate(batch_mapping):
            res_idx = batch_idx
            # SummaC
            batch_res_items[res_idx][f"{m}_summac"] = batch_summac_scores[k]
            # UniEval
            for dim, score in batch_uni_scores[k].items():
                batch_res_items[res_idx][f"{m}_unieval_{dim}"] = score
        
        results.extend(batch_res_items)

        # Save checkpoint periodically
        if i % (BATCH_SIZE * 5) == 0:
            pd.DataFrame(results).to_csv("advanced_eval_checkpoint.csv", index=False)

    # Save Final Report
    df = pd.DataFrame(results)
    df.to_csv("final_advanced_metrics_1000.csv", index=False)
    
    # Calculate Averages
    print("\n" + "="*50)
    print("FINAL CONSOLIDATED RESULTS (AVERAGES)")
    print("="*50)
    
    summary_stats = []
    for m in ["treesum", "flat_1024", "flat_overlap"]:
        m_stats = {"Method": m}
        m_stats["SummaC"] = df[f"{m}_summac"].mean()
        for dim in ["coherence", "consistency", "fluency", "relevance"]:
            m_stats[f"UniEval_{dim}"] = df[f"{m}_unieval_{dim}"].mean()
        summary_stats.append(m_stats)
    
    stats_df = pd.DataFrame(summary_stats)
    print(stats_df.to_string(index=False))
    stats_df.to_csv("final_advanced_averages.csv", index=False)
    
    print("\n✓ Evaluation complete. Files saved: 'final_advanced_metrics_1000.csv' and 'final_advanced_averages.csv'")

if __name__ == "__main__":
    main()
