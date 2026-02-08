
"""
================================================================================
TREESUM A40 PRODUCTION: 1000-SAMPLE ALPHA SWEEP ABLATION STUDY
================================================================================

This script is a customized version of 'kaggle_complete_experiment-3.py'
optimized for the A40 GPU.

Key Changes for A40:
1. Data: Loads from 'golden_1000_shared.json' (Set A Alignment).
2. Hardware: Optimized for A40 (bfloat16, Batch Size 32).
3. Logic: 100% functional parity with the Kaggle experiment.
4. Scale: Evaluates all 1000 samples across the alpha sweep.

Author: Antigravity/Research Team
Date: 2026-02-08
================================================================================
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
import time
import warnings
from dataclasses import dataclass, field
from collections import defaultdict

# Transformers & Evaluation
from transformers import PegasusForConditionalGeneration, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import evaluate

# NLTK setup
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Scipy
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/golden_1000_shared.json'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/a40_ablation_sweep'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] # High-fidelity sweep
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# A40 Optimized: bfloat16 + Large Batch
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
BATCH_SIZE = 32

logger.info(f"A40 CONFIGURATION: 1000 samples, {len(ALPHA_VALUES)} alphas, Device: {DEVICE}, Precision: {DTYPE}")

# ============================================================================
# SECTION 4: SEMANTIC DOCUMENT CHUNKER (EXACT COPY FROM EXP-3)
# ============================================================================

@dataclass
class ChunkingConfig:
    max_tokens: int = 1024
    overlap_tokens: int = 128
    min_chunk_tokens: int = 256
    use_sentence_boundaries: bool = True
    preserve_paragraphs: bool = True
    use_semantic_coherence: bool = True
    semantic_similarity_threshold: float = 0.7
    adaptive_overlap: bool = True
    overlap_tolerance: float = 0.2

class SemanticDocumentChunker:
    def __init__(self, 
                 tokenizer=None,
                 max_tokens: int = 1024, 
                 overlap_tokens: int = 128,
                 use_semantic_coherence: bool = True,
                 semantic_weight: float = 0.7):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.use_semantic_coherence = use_semantic_coherence
        self.semantic_weight = semantic_weight
        self.adaptive_overlap = True
        self.semantic_similarity_threshold = 0.7
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

    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        uncached = [s for s in sentences if s not in self._embedding_cache]
        if uncached:
            embs = self.semantic_model.encode(uncached, convert_to_numpy=True)
            for s, e in zip(uncached, embs): self._embedding_cache[s] = e
        return np.array([self._embedding_cache[s] for s in sentences])

    def compute_semantic_coherence(self, sentences: List[str]) -> float:
        if len(sentences) < 2: return 1.0
        embs = self.get_sentence_embeddings(sentences)
        sims = []
        for i in range(len(embs)-1):
            sim = np.dot(embs[i], embs[i+1])/(np.linalg.norm(embs[i])*np.linalg.norm(embs[i+1]))
            sims.append((sim+1)/2)
        return np.mean(sims)

    def find_optimal_overlap_sentences(self, prev_sents: List[str], target_tokens: int) -> List[str]:
        if not prev_sents: return []
        overlap = []; curr = 0
        lb = target_tokens * 0.75
        ub = target_tokens * 1.5
        for sent in reversed(prev_sents):
            s_cnt = self.get_token_count(sent)
            if curr > 0 and curr + s_cnt > ub: break
            overlap.insert(0, sent)
            curr += s_cnt
            if curr >= lb:
                if self.compute_semantic_coherence(overlap) < self.semantic_similarity_threshold: break
                if curr >= target_tokens: break
        return overlap

    def chunk_document(self, document: str) -> List[Dict]:
        sents = sent_tokenize(document.replace("|||", " "))
        chunks = []; curr_sents = []; curr_tokens = 0
        for sent in sents:
            s_cnt = self.get_token_count(sent)
            if curr_tokens + s_cnt > self.max_tokens:
                if curr_tokens >= 256 or not chunks:
                    chunks.append({'text': " ".join(curr_sents), 'sentences': curr_sents})
                    overlap = self.find_optimal_overlap_sentences(curr_sents, self.overlap_tokens)
                    curr_sents = overlap + [sent]
                    curr_tokens = sum(self.get_token_count(s) for s in curr_sents)
                else:
                    curr_sents.append(sent); curr_tokens += s_cnt
            else:
                curr_sents.append(sent); curr_tokens += s_cnt
        if curr_sents: chunks.append({'text': " ".join(curr_sents), 'sentences': curr_sents})
        return chunks

# ============================================================================
# SECTION 5: HIERARCHICAL SUMMARIZER (EXACT COPY FROM EXP-3)
# ============================================================================

class HierarchicalSummarizer:
    def __init__(self, device, semantic_weight, batch_size, dtype):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("google/pegasus-multi_news", use_fast=False)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            "google/pegasus-multi_news", torch_dtype=self.dtype
        ).to(self.device)
        self.chunker = SemanticDocumentChunker(
            tokenizer=self.tokenizer, semantic_weight=semantic_weight
        )

    def _generate(self, inputs: List[str], max_length: int = 512, min_length: int = 64) -> List[str]:
        batch = self.tokenizer(inputs, truncation=True, padding="longest", max_length=1024, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                num_beams=8, max_length=max_length, min_length=min_length,
                length_penalty=0.8, no_repeat_ngram_size=3, early_stopping=True
            )
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def summarize_document(self, document: str) -> Dict:
        chunks = self.chunker.chunk_document(document)
        chunk_texts = [c['text'] for c in chunks]
        local_max = 128 if len(chunk_texts) > 5 else 256
        chunk_sums = []
        for i in range(0, len(chunk_texts), self.batch_size):
            chunk_sums.extend(self._generate(chunk_texts[i:i+self.batch_size], max_length=local_max))
        
        # Reduction
        curr_sums = chunk_sums
        while True:
            combined = " ".join(curr_sums)
            if len(self.tokenizer.encode(combined)) <= 1000:
                final = self._generate([combined], max_length=512, min_length=128)[0]
                return {'final_summary': final}
            
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
# EXPERIMENT RUNNER
# ============================================================================

def run_a40_ablation():
    # 1. Load Data
    logger.info(f"Loading Golden Data from {DATA_PATH}...")
    with open(DATA_PATH, 'r') as f:
        samples = json.load(f)
    
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
    
    results_table = {}
    
    for alpha in ALPHA_VALUES:
        logger.info(f"🚀 Starting Alpha Sweep: {alpha}")
        summarizer = HierarchicalSummarizer(DEVICE, alpha, BATCH_SIZE, DTYPE)
        
        preds = []; refs = []
        for item in tqdm(samples, desc=f"Alpha={alpha}"):
            try:
                res = summarizer.summarize_document(item['document'])
                preds.append(res['final_summary'])
                refs.append(item['summary'])
            except Exception as e:
                logger.error(f"Error at alpha {alpha}, ID {item['id']}: {e}")
                preds.append(""); refs.append(item['summary'])
        
        # Metrics
        r_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        
        # BERTScore (Safety: move main model to CPU if needed, but A40 should handle both)
        logger.info("Computing BERTScore...")
        b_scores = bertscore.compute(
            predictions=preds, references=refs, lang="en", 
            model_type="microsoft/deberta-xlarge-mnli", device=DEVICE, batch_size=16
        )
        
        results_table[f"alpha_{alpha}"] = {
            'rouge1': r_scores['rouge1'] * 100,
            'rouge2': r_scores['rouge2'] * 100,
            'rougeL': r_scores['rougeL'] * 100,
            'bertscore_f1': np.mean(b_scores['f1']) * 100
        }
        
        # Save checkpoints
        alpha_out = os.path.join(OUTPUT_DIR, f"summaries_alpha_{alpha}.json")
        with open(alpha_out, 'w') as f:
            json.dump([{"id": s['id'], "summary": p} for s, p in zip(samples, preds)], f, indent=2)
        
        # Clean up for next alpha
        del summarizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Final Export
    df = pd.DataFrame(results_table).T
    df.to_csv(os.path.join(OUTPUT_DIR, "alpha_sweep_results.csv"))
    logger.info(f"✨ Study complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_a40_ablation()
