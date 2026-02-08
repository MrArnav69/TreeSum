
"""
================================================================================
TREESUM PRODUCTION: THREE-WAY CHUNKING STRATEGY ABLATION (A40 OPTIMIZED)
================================================================================

This script performs a synchronized ablation study comparing three chunking strategies:
1. Flat 1024 (Baseline)
2. Flat Overlap (128 Tokens)
3. TreeSum (Semantic/Adaptive)

All experiments run on the SAME 1,000 samples (golden_1000_shared.json) to 
ensure index-perfect scientific validity for SummaC/UniEval evaluation.

Author: Antigravity/Research Team
Date: 2026-02-08
================================================================================
"""

import os
import sys
import json
import time
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional, Any
from transformers import PegasusForConditionalGeneration, AutoTokenizer

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from hierarchical_summarizer import HierarchicalSummarizer

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/golden_1000_shared.json'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/chunking_ablation'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
BATCH_SIZE = 32 # A40 has 48GB VRAM

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. FLAT CHUNKER (Baseline 1)
# ==========================================
class FlatChunker1024:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = 1024
    def chunk_document(self, text: str) -> List[Dict]:
        text = text.replace("|||", " ").strip()
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunks.append({'text': self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)})
        return chunks

# ==========================================
# 2. OVERLAP CHUNKER (Baseline 2)
# ==========================================
class FlatChunkerOverlap:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = 1024
        self.overlap = 128
        self.step = self.max_tokens - self.overlap
    def chunk_document(self, text: str) -> List[Dict]:
        text = text.replace("|||", " ").strip()
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), self.step):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunks.append({'text': self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)})
            if i + self.max_tokens >= len(tokens): break
        return chunks

# ==========================================
# MASTER RUNNER
# ==========================================
def run_ablation():
    logger.info("Initializing A40 Workspace...")
    with open(DATA_PATH, 'r') as f:
        samples = json.load(f)
    
    # Initialize Shared Model
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-multi_news", use_fast=False)
    
    methods = ["flat_1024", "flat_overlap", "treesum"]
    master_results = {m: [] for m in methods}
    
    # To save time and avoid reloading the model, we wrap the summarizer
    summarizer = HierarchicalSummarizer(device=DEVICE, dtype=DTYPE, batch_size=BATCH_SIZE)
    
    for method in methods:
        logger.info(f"\n🚀 STARTING METHOD: {method}")
        
        # Inject the correct chunker for this variation
        if method == "flat_1024":
            summarizer.chunker = FlatChunker1024(tokenizer)
        elif method == "flat_overlap":
            summarizer.chunker = FlatChunkerOverlap(tokenizer)
        else:
            # Restore TreeSum SOTA chunker defaults
            summarizer.chunker = None 
            summarizer.__init__(device=DEVICE, dtype=DTYPE, batch_size=BATCH_SIZE, semantic_weight=1.0)
        
        output_file = os.path.join(OUTPUT_DIR, f"summaries_{method}.json")
        processed_data = []
        
        for item in tqdm(samples, desc=f"Experiment: {method}"):
            try:
                res = summarizer.summarize_document(item['document'])
                processed_data.append({
                    "sample_id": item['id'],
                    "generated_summary": res['final_summary'],
                    "reference_summary": item['summary']
                })
            except Exception as e:
                logger.error(f"Error at ID {item['id']}: {e}")
                processed_data.append({"sample_id": item['id'], "summary": "[ERROR]"})
        
        # Save results for this method
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        logger.info(f"✅ Saved {method} results.")

    logger.info("\n✨ CHUNKING ABLATION COMPLETE!")

if __name__ == "__main__":
    run_ablation()
