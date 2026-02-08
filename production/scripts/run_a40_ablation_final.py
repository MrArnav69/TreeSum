
"""
================================================================================
TREESUM PRODUCTION: MONOLITHIC CHUNKING ABLATION STUDY (A40 OPTIMIZED)
================================================================================

This script is a 100% self-contained research pipeline for the A40 GPU.
It performs a direct head-to-head comparison of three strategies:
1. Flat 1024 (Baseline)
2. Flat Overlap (Baseline + 128 tokens)
3. TreeSum (Semantic/Adaptive - TreeSum 2.1)

CRITICAL FIXES:
- Uses random.sample(seed=42) to match EXACT indices of original Flat baselines.
- Uses bfloat16 for high-speed A40 execution.
- Includes ALL functions from kaggle_complete_experiment-3.py for parity.

Author: Arnav Gupta / Antigravity
Date: 2026-02-08
================================================================================
"""

import os
import re
import json
import time
import torch
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from transformers import PegasusForConditionalGeneration, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_SEED = 42
NUM_SAMPLES = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
BATCH_SIZE = 32 # Optimized for A40 48GB
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/a40_chunking_ablation_monolithic'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SECTION 1: DOCUMENT CHUNKERS (IDENTICAL TO KAGGLE RUNS)
# ============================================================================

def clean_document_shared(text: str) -> str:
    """Standard TreeSum cleaning used in all variations."""
    text = re.sub(r'Enlarge this image.*?AP', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'toggle caption.*?AP', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    return text.strip()

class FlatChunker1024:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def chunk_document(self, document: str) -> List[Dict]:
        doc = clean_document_shared(document)
        tokens = self.tokenizer.encode(doc, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), 1024):
            chunk_tokens = tokens[i:i + 1024]
            chunks.append({'text': self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)})
        return chunks

class FlatChunkerOverlap:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = 1024
        self.overlap = 128
        self.step = 1024 - 128
    def chunk_document(self, document: str) -> List[Dict]:
        doc = clean_document_shared(document)
        tokens = self.tokenizer.encode(doc, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), self.step):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunks.append({'text': self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)})
            if i + self.max_tokens >= len(tokens): break
        return chunks

class SemanticDocumentChunker:
    """Full TreeSum Semantic Chunker from Exp-3."""
    def __init__(self, tokenizer, semantic_weight=1.0):
        self.tokenizer = tokenizer
        self.max_tokens = 1000
        self.overlap_tokens = 128
        self.semantic_weight = semantic_weight
        self._semantic_model = None
        self._token_cache = {}
        self._embedding_cache = {}

    @property
    def semantic_model(self):
        if self._semantic_model is None:
            self._semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)
        return self._semantic_model

    def get_token_count(self, text: str) -> int:
        if text in self._token_cache: return self._token_cache[text]
        cnt = len(self.tokenizer.tokenize(text))
        self._token_cache[text] = cnt
        return cnt

    def find_optimal_overlap_sentences(self, prev_sents: List[str], target: int) -> List[str]:
        if not prev_sents: return []
        overlap = []; curr = 0
        for sent in reversed(prev_sents):
            s_cnt = self.get_token_count(sent)
            if curr > 0 and curr + s_cnt > target * 1.5: break
            overlap.insert(0, sent)
            curr += s_cnt
            if curr >= target * 0.8: break
        return overlap

    def chunk_document(self, document: str) -> List[Dict]:
        doc = clean_document_shared(document)
        sents = sent_tokenize(doc)
        chunks = []; curr_sents = []; curr_tokens = 0
        for sent in sents:
            s_cnt = self.get_token_count(sent)
            if curr_tokens + s_cnt > self.max_tokens:
                chunks.append({'text': " ".join(curr_sents), 'sentences': curr_sents})
                overlap = self.find_optimal_overlap_sentences(curr_sents, self.overlap_tokens)
                curr_sents = overlap + [sent]
                curr_tokens = sum(self.get_token_count(s) for s in curr_sents)
            else:
                curr_sents.append(sent); curr_tokens += s_cnt
        if curr_sents: chunks.append({'text': " ".join(curr_sents), 'sentences': curr_sents})
        return chunks

# ============================================================================
# SECTION 2: HIERARCHICAL SUMMARIZER
# ============================================================================

class HierarchicalSummarizer:
    def __init__(self, device, dtype, batch_size):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("google/pegasus-multi_news", use_fast=False)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            "google/pegasus-multi_news", torch_dtype=self.dtype
        ).to(self.device)
        self.chunker = None

    def _generate(self, inputs: List[str], max_length: int = 512, min_length: int = 64) -> List[str]:
        batch = self.tokenizer(inputs, truncation=True, padding="longest", max_length=1024, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                num_beams=8, max_length=max_length, min_length=min_length,
                length_penalty=0.8, no_repeat_ngram_size=3, early_stopping=True
            )
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def summarize_document(self, document: str) -> str:
        chunks = self.chunker.chunk_document(document)
        chunk_texts = [c['text'] for c in chunks]
        local_max = 128 if len(chunk_texts) > 5 else 256
        chunk_sums = []
        for i in range(0, len(chunk_texts), self.batch_size):
            chunk_sums.extend(self._generate(chunk_texts[i:i+self.batch_size], max_length=local_max))
        
        # Tree Reduction
        curr_sums = chunk_sums
        while True:
            combined = " ".join(curr_sums)
            if len(self.tokenizer.encode(combined)) <= 1000:
                final = self._generate([combined], max_length=512, min_length=128)[0]
                return final
            
            new_sums = []; group = []; g_len = 0
            for s in curr_sums:
                s_len = len(self.tokenizer.encode(s))
                if g_len + s_len > 1000:
                    new_sums.append(self._generate([" ".join(group)], max_length=256)[0])
                    group = [s]; g_len = s_len
                else:
                    group.append(s); g_len += s_len
            if group: new_sums.append(self._generate([" ".join(group)], max_length=256)[0])
            curr_sums = new_sums

# ============================================================================
# MASTER EXECUTION
# ============================================================================

def run_production_ablation():
    logger.info("🚀 INITIALIZING PRODUCTION ABLATION STUDY")
    logger.info(f"Seed: {RANDOM_SEED} | Device: {DEVICE} | Precision: {DTYPE}")

    # 1. Synchronized Data Loading (Set A Alignment)
    logger.info("Loading Multi-News dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    samples = dataset.select(indices)
    logger.info(f"✅ Selected {len(samples)} samples perfectly aligned with Flat baselines.")

    # 2. Initialize Model
    summarizer = HierarchicalSummarizer(DEVICE, DTYPE, BATCH_SIZE)
    rouge = evaluate.load('rouge')
    
    methods = ["flat_1024", "flat_overlap", "treesum"]
    results_summary = {}

    for method in methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 PROCESSING METHOD: {method.upper()}")
        logger.info(f"{'='*80}")

        # Inject correct chunker
        if method == "flat_1024":
            summarizer.chunker = FlatChunker1024(summarizer.tokenizer)
        elif method == "flat_overlap":
            summarizer.chunker = FlatChunkerOverlap(summarizer.tokenizer)
        else:
            summarizer.chunker = SemanticDocumentChunker(summarizer.tokenizer)

        predictions = []
        references = []
        
        output_file = os.path.join(OUTPUT_DIR, f"summaries_{method}.json")
        checkpoint_file = os.path.join(OUTPUT_DIR, f"checkpoint_{method}.json")
        
        # Resume if possible
        processed_data = []
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                processed_data = json.load(f)
            logger.info(f"Resuming from {len(processed_data)} samples.")

        for i in tqdm(range(len(processed_data), len(samples)), desc=method):
            doc = samples[i]['document']
            ref = samples[i]['summary']
            try:
                pred = summarizer.summarize_document(doc)
                processed_data.append({
                    "sample_id": int(indices[i]),
                    "generated_summary": pred,
                    "reference_summary": ref,
                    "document": doc
                })
            except Exception as e:
                logger.error(f"Error at method {method}, Index {i}: {e}")
                processed_data.append({"sample_id": int(indices[i]), "summary": "[ERROR]"})
            
            if (i + 1) % 50 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(processed_data, f, indent=2)

        # Final save for method
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        # Quick Rouge check
        preds_valid = [r['generated_summary'] for r in processed_data if r['generated_summary'] != "[ERROR]"]
        refs_valid = [r['reference_summary'] for r in processed_data if r['generated_summary'] != "[ERROR]"]
        scores = rouge.compute(predictions=preds_valid, references=refs_valid)
        results_summary[method] = {k: v * 100 for k, v in scores.items()}
        logger.info(f"Method {method} ROUGE-1: {results_summary[method]['rouge1']:.2f}")

    # Final Export
    df = pd.DataFrame(results_summary).T
    df.to_csv(os.path.join(OUTPUT_DIR, "ablation_results_rouge.csv"))
    logger.info("\n✨ PRODUCTION ABLATION COMPLETE!")

if __name__ == "__main__":
    run_production_ablation()
