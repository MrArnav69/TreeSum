
"""
================================================================================
TREESUM PRODUCTION: 1000-SAMPLE SOTA STUDY (A40 OPTIMIZED)
================================================================================

This script is the final "Golden" run for the research paper.
It uses:
- Hardware: A40/H100 (bfloat16 enabled)
- Data: golden_1000_shared.json (Set A - Fully Aligned with Baselines)
- Algorithm: TreeSum 2.1 (Semantic + Adaptive Overlap)
- Metrics: ROUGE, BERTScore (DeBERTa-XL), and Preparation for SummaC/UniEval

Optimization:
- bfloat16 precision for speed and memory efficiency
- Batch Size 32+ (VRAM permitting)
- Automated resumption from checkpoints

Author: Antigravity
Date: 2026-02-08
================================================================================
"""

import os
import sys
import json
import time
import logging
import torch
import pandas as pd
import evaluate
from tqdm import tqdm
from typing import List, Dict, Optional, Any

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from hierarchical_summarizer import HierarchicalSummarizer

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/golden_1000_shared.json'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/mega_sota_run'))
LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/a40_production.log'))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# A40/H100 Optimized Settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Use bfloat16 on Ampere architectures (A40) for better numerical stability than float16
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
BATCH_SIZE = 32 # A40 has 48GB VRAM
ALPHA = 1.0

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_production_sota():
    logger.info("="*80)
    logger.info("STARTING TREESUM A40 PRODUCTION RUN")
    logger.info(f"Hardware: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Precision: {DTYPE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info("="*80)

    # 1. Load Golden Data
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}")
        return
    
    with open(DATA_PATH, 'r') as f:
        samples = json.load(f)
    logger.info(f"Loaded {len(samples)} sample-aligned documents.")

    # 2. Check for Resume
    final_output_path = os.path.join(OUTPUT_DIR, "summaries_final.json")
    results = []
    if os.path.exists(final_output_path):
        with open(final_output_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Resuming from {len(results)} samples.")
    
    processed_ids = {r['sample_id'] for r in results}
    remaining_samples = [s for s in samples if s['id'] not in processed_ids]

    if not remaining_samples:
        logger.info("✅ All samples already processed!")
        return

    # 3. Initialize Summarizer
    logger.info("Initializing TreeSum Engine...")
    summarizer = HierarchicalSummarizer(
        device=DEVICE,
        semantic_weight=ALPHA,
        dtype=DTYPE,
        batch_size=BATCH_SIZE,
        context_aware=False # Independent mode as per 2.1 Refinement
    )

    # 4. Evaluation Loop
    logger.info(f"Processing {len(remaining_samples)} remaining samples...")
    
    # We save every 50 samples for safety
    batch_buffer = []
    
    for i, sample in enumerate(tqdm(remaining_samples)):
        doc = sample['document']
        ref = sample['summary']
        s_id = sample['id']
        
        try:
            start_time = time.time()
            output = summarizer.summarize_document(doc)
            elapsed = time.time() - start_time
            
            result_item = {
                "sample_id": s_id,
                "generated_summary": output['final_summary'],
                "reference_summary": ref,
                "document": doc,
                "num_chunks": output.get('num_chunks', 0),
                "time_taken": elapsed
            }
            results.append(result_item)
            batch_buffer.append(result_item)
            
        except Exception as e:
            logger.error(f"FAILED Sample ID {s_id}: {str(e)}")
            # Log empty result to maintain alignment but flag failure
            results.append({
                "sample_id": s_id,
                "generated_summary": "[ERROR]",
                "reference_summary": ref,
                "document": doc,
                "error": str(e)
            })

        # Incremental Save
        if (i + 1) % 50 == 0:
            with open(final_output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Checkpoint saved: {len(results)} samples total.")

    # Final Save
    with open(final_output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✨ Production Run Complete! Final results at: {final_output_path}")

    # 5. Quick Metric Preview
    logger.info("Computing preliminary scores...")
    rouge = evaluate.load('rouge')
    
    preds = [r['generated_summary'] for r in results if r['generated_summary'] != "[ERROR]"]
    refs = [r['reference_summary'] for r in results if r['generated_summary'] != "[ERROR]"]
    
    if preds:
        scores = rouge.compute(predictions=preds, references=refs)
        logger.info(f"ROUGE-1: {scores['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {scores['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {scores['rougeL']:.4f}")

if __name__ == "__main__":
    run_production_sota()
