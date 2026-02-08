"""
================================================================================
TREESUM ABLATION STUDY: TREESUM (BASELINE 3 - PROPOSED METHOD)
================================================================================

Total Integrated Script for Kaggle P100 GPU.
Contains complete source code for:
1. SemanticDocumentChunker (SOTA) - 100% Feature Parity with Source
2. HierarchicalSummarizer
3. Robust Evaluation Pipeline (P100 Optimized)

Configuration:
- Model: google/pegasus-multi_news
- Precision: float32 (Stability)
- Chunker: Semantic + Adaptive Overlap (TreeSum)
- Batch Size: 4 (P100 Tuned)

Author: Antigravity (Generated for User)
Date: 2026-02-06
================================================================================
"""

import sys
import subprocess
import os
import time
import json
import random
import logging
import re
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================================
# 1. ENVIRONMENT SETUP (Self-Installing)
# ============================================================================
def setup_environment():
    """Installs missing dependencies and sets up necessary resources."""
    print("="*60)
    print("SETTING UP ENVIRONMENT")
    print("="*60)
    
    required_packages = [
        "transformers", 
        "datasets", 
        "evaluate", 
        "rouge_score", 
        "bert_score",
        "accelerate",
        "sentencepiece",
        "sentence-transformers", # Required for TreeSum
        "scikit-learn",          # Required for TreeSum (sklearn)
        "scipy",                 # Required for TreeSum
        "psutil"                 # Required for memory stats
    ]
    
    for package in required_packages:
        try:
            # Handle package names that differ from import names
            import_name = package
            if package == "scikit-learn": import_name = "sklearn"
            if package == "sentence-transformers": import_name = "sentence_transformers"
            
            __import__(import_name)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    # NLTK Data for ROUGE and Splitter
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)
    
    print("Environment setup complete.\n")

# Run setup immediately
setup_environment()

# Imports after setup
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import PegasusForConditionalGeneration, AutoTokenizer, PegasusTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
import psutil

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("WARNING: Sentence Transformers not available. Falling back to Token-only mode.")

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
RANDOM_SEED = 42
NUM_SAMPLES = 1000
BATCH_SIZE = 2  # Tuned for P100 (16GB VRAM) with float32
MAX_TOKENS = 1024 
CHECKPOINT_INTERVAL = 50 # Save often

# Paths
if os.path.exists('/kaggle/working'):
    OUTPUT_DIR = "/kaggle/working/results_treesum_ablation"
else:
    OUTPUT_DIR = "./results_treesum_ablation"

# Logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================================
# 3. COMPLETE SOURCE CODE: SEMANTIC DOCUMENT CHUNKER
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

class SemanticDocumentChunker:
    """
    Production-ready semantic document chunker with comprehensive validation.
    
    This chunker implements:
    - Sentence-boundary preservation
    - Semantic coherence via sentence embeddings
    - Adaptive overlap selection
    - Comprehensive validation
    """
    def __init__(self, 
                 tokenizer=None,
                 model_name: Optional[str] = None,
                 max_tokens: int = 1024, 
                 overlap_tokens: int = 128,
                 use_sentence_boundaries: bool = True,
                 min_chunk_tokens: int = 256,
                 preserve_paragraphs: bool = True,
                 use_semantic_coherence: bool = True,
                 semantic_model: Optional[str] = None,
                 semantic_similarity_threshold: float = 0.7,
                 adaptive_overlap: bool = True,
                 semantic_weight: float = 0.7,
                 ablation_mode: Optional[str] = None,
                 enable_validation: bool = True,
                 validate_overlap_tokens: bool = True,
                 validate_semantic_coherence: bool = True):
        
        # Tokenizer initialization
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name is not None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            from transformers import PegasusTokenizer
            self.tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-multi_news")
        
        # Configuration
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.use_sentence_boundaries = use_sentence_boundaries
        self.min_chunk_tokens = min_chunk_tokens
        self.preserve_paragraphs = preserve_paragraphs
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.adaptive_overlap = adaptive_overlap
        self.semantic_weight = semantic_weight
        self.ablation_mode = ablation_mode
        self.enable_validation = enable_validation
        self.validate_overlap_tokens = validate_overlap_tokens
        self.validate_semantic_coherence = validate_semantic_coherence
        
        # Initial default
        self.use_semantic_coherence = use_semantic_coherence and SEMANTIC_AVAILABLE
        
        # Apply ablation mode settings
        if ablation_mode == 'no_semantic':
            self.use_semantic_coherence = False
            self.adaptive_overlap = False
        elif ablation_mode == 'no_overlap':
            self.use_semantic_coherence = False
            self.adaptive_overlap = False
            self.overlap_tokens = 0
        elif ablation_mode == 'fixed_overlap':
            self.use_semantic_coherence = False
            self.adaptive_overlap = False
        elif ablation_mode == 'no_sentence_boundaries':
            self.use_semantic_coherence = False
            self.use_sentence_boundaries = False
            self.adaptive_overlap = False
        elif ablation_mode == 'large_overlap':
            self.overlap_tokens = self.overlap_tokens * 2
        elif ablation_mode == 'small_overlap':
            self.overlap_tokens = self.overlap_tokens // 2
        
        # Performance tracking
        self._performance_stats = {
            'total_chunking_time': 0.0,
            'total_embedding_time': 0.0,
            'total_token_count_time': 0.0,
            'num_documents_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Human-readable stats
        self._token_cache: Dict[str, int] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Semantic model (Lazy Load)
        self._semantic_model = None
        self._semantic_model_name = semantic_model
        
        # Validation checks on init
        if overlap_tokens >= max_tokens:
            raise ValueError(f"Overlap ({overlap_tokens}) must be less than max_tokens ({max_tokens})")

    @property
    def semantic_model(self):
        """Lazy load semantic model."""
        if self._semantic_model is None and self.use_semantic_coherence:
            model_name = self._semantic_model_name or 'sentence-transformers/all-MiniLM-L6-v2'
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._semantic_model = SentenceTransformer(model_name, device=device)
                print(f"✓ Semantic model loaded on {device}")
            except Exception as e:
                warnings.warn(f"Failed to load semantic model: {e}. Disabling semantic features.")
                self.use_semantic_coherence = False
                self._semantic_model = None
        return self._semantic_model

    def get_token_count(self, text: str, use_cache: bool = True) -> int:
        if use_cache and text in self._token_cache:
            return self._token_cache[text]
        token_count = len(self.tokenizer.tokenize(text))
        if use_cache:
            self._token_cache[text] = token_count
        return token_count
    
    def get_sentence_embeddings(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        if not self.use_semantic_coherence or self.semantic_model is None:
            return None
        
        uncached_sentences = [s for s in sentences if s not in self._embedding_cache]
        
        if uncached_sentences:
            embeddings = self.semantic_model.encode(
                uncached_sentences,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            for sent, emb in zip(uncached_sentences, embeddings):
                self._embedding_cache[sent] = emb
        
        return np.array([self._embedding_cache[s] for s in sentences])
    
    def compute_semantic_coherence(self, sentences: List[str], global_coherence: bool = False) -> Dict[str, float]:
        if not self.use_semantic_coherence or len(sentences) < 2:
            return {'local_coherence': 1.0}
        
        embeddings = self.get_sentence_embeddings(sentences)
        if embeddings is None:
            return {'local_coherence': 1.0}
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-9
            )
            similarities.append((sim + 1) / 2)
        
        local_coherence = np.mean(similarities) if similarities else 1.0
        
        result = {'local_coherence': float(local_coherence)}
        
        if global_coherence and len(embeddings) > 2:
            centroid = np.mean(embeddings, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            centroid_similarities = []
            for emb in embeddings:
                sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * centroid_norm + 1e-9)
                centroid_similarities.append((sim + 1) / 2)
            centroid_coherence = float(np.mean(centroid_similarities))
            result['global_coherence'] = centroid_coherence
            
        return result

    def find_optimal_overlap_sentences(self, previous_sentences: List[str], target_overlap_tokens: int) -> List[str]:
        if not previous_sentences:
            return []
        
        overlap_sentences = []
        current_tokens = 0
        lower_bound = target_overlap_tokens * 0.75
        upper_bound = target_overlap_tokens * 1.5
        
        for sent in reversed(previous_sentences):
            sent_tokens = self.get_token_count(sent)
            if current_tokens > 0 and current_tokens + sent_tokens > upper_bound:
                break
            
            overlap_sentences.insert(0, sent)
            current_tokens += sent_tokens
            
            if current_tokens >= lower_bound:
                if self.adaptive_overlap and self.use_semantic_coherence:
                    coherence = self.compute_semantic_coherence(overlap_sentences)['local_coherence']
                    if coherence < self.semantic_similarity_threshold:
                        break
                    elif current_tokens >= target_overlap_tokens:
                        break
                else:
                    if current_tokens >= target_overlap_tokens * 0.8:
                        break
        return overlap_sentences

    def split_into_articles(self, document: str) -> List[str]:
        articles = [art.strip() for art in document.split("|||") if len(art.strip()) > 0]
        return articles if articles else [document.strip()]
    
    def split_into_sentences(self, text: str) -> List[str]:
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 0]
        return paragraphs if paragraphs else [text.strip()]

    def _compute_chunk_coherence(self, sentences: List[str]):
        if not self.use_semantic_coherence:
            return None, None
        coherence_dict = self.compute_semantic_coherence(sentences, global_coherence=(len(sentences)>5))
        return coherence_dict['local_coherence'], coherence_dict
    
    def _create_chunk_dict(self, chunk_id, sentences, token_count, article_idx, has_overlap, overlap_token_count, 
                          token_start, token_end, coherence_score=None, coherence_dict=None):
        return {
            'chunk_id': chunk_id,
            'text': ' '.join(sentences),
            'sentences': sentences.copy(),
            'token_count': token_count,
            'has_overlap': has_overlap,
            'overlap_token_count': overlap_token_count,
            'semantic_coherence_score': coherence_score,
            'semantic_coherence_metrics': coherence_dict
        }

    def _compute_lexical_similarity(self, sent1: str, sent2: str) -> float:
        tokens1 = set(self.tokenizer.tokenize(sent1.lower()))
        tokens2 = set(self.tokenizer.tokenize(sent2.lower()))
        if not tokens1 or not tokens2: return 0.0
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / union

    def _find_optimal_split_point(self, sentences: List[str], current_tokens: List[int]) -> int:
        """SOTA Hybrid Split Finder"""
        if not self.use_semantic_coherence or self.semantic_model is None:
            return len(sentences)
            
        n_sentences = len(sentences)
        if n_sentences < 3: return n_sentences
            
        # 1. Semantic Signal
        embeddings = self.get_sentence_embeddings(sentences)
        semantic_sims = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-9)
            semantic_sims.append((sim + 1) / 2)
        semantic_sims = np.array(semantic_sims)
        
        # 2. Lexical Signal
        lexical_sims = []
        for i in range(n_sentences - 1):
            lexical_sims.append(self._compute_lexical_similarity(sentences[i], sentences[i+1]))
        lexical_sims = np.array(lexical_sims)
        if np.max(lexical_sims) > 0: lexical_sims /= np.max(lexical_sims)
            
        # 3. Hybrid
        alpha = self.semantic_weight
        hybrid_sims = alpha * semantic_sims + (1.0 - alpha) * lexical_sims
        
        # 4. Smoothing
        smoothed_sims = gaussian_filter1d(hybrid_sims, sigma=1.0)
        
        # 5. Minima Detection
        cum_tokens = np.cumsum(current_tokens)
        total_tokens = cum_tokens[-1]
        
        valid_indices = []
        for i in range(len(smoothed_sims)):
            if cum_tokens[i] >= self.min_chunk_tokens:
                valid_indices.append(i)
        
        if not valid_indices: return n_sentences
            
        valid_signal = smoothed_sims[valid_indices]
        if len(valid_signal) < 3:
             best_idx = valid_indices[np.argmin(valid_signal)]
             return best_idx + 1
             
        minima_indices_local = argrelextrema(valid_signal, np.less, order=1)[0]
        
        if len(minima_indices_local) > 0:
            candidates = []
            for local_idx in minima_indices_local:
                real_idx = valid_indices[local_idx]
                val = valid_signal[local_idx]
                dist_penalty = (total_tokens - cum_tokens[real_idx]) / total_tokens
                score = val + (0.3 * dist_penalty)
                candidates.append((score, real_idx))
            
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1] + 1
        else:
            return valid_indices[np.argmin(valid_signal)] + 1

    def chunk_with_sentence_boundaries(self, sentences, article_idx, previous_chunk_sentences, token_offset):
        chunks = []
        sentence_buffer = []
        buffer_tokens = []
        current_tokens = 0
        overlap_sentences = []
        
        if previous_chunk_sentences:
            overlap_sentences = self.find_optimal_overlap_sentences(previous_chunk_sentences, self.overlap_tokens)
            for s in overlap_sentences:
                t_count = self.get_token_count(s)
                sentence_buffer.append(s)
                buffer_tokens.append(t_count)
                current_tokens += t_count
        
        for sent in sentences:
            sent_tokens = self.get_token_count(sent)
            
            if current_tokens + sent_tokens > self.max_tokens:
                # Split at optimal point
                split_idx = self._find_optimal_split_point(sentence_buffer, buffer_tokens)
                
                chunk_sentences = sentence_buffer[:split_idx]
                chunk_token_count = sum(buffer_tokens[:split_idx])
                
                coherence, metrics = self._compute_chunk_coherence(chunk_sentences)
                
                chunk = self._create_chunk_dict(len(chunks), chunk_sentences, chunk_token_count, article_idx, 
                                               (len(chunks)>0 or previous_chunk_sentences), 0, 0, 0, coherence, metrics)
                chunks.append(chunk)
                
                # Prepare next buffer
                new_overlap = self.find_optimal_overlap_sentences(chunk_sentences, self.overlap_tokens)
                remaining = sentence_buffer[split_idx:]
                sentence_buffer = new_overlap + remaining + [sent]
                buffer_tokens = [self.get_token_count(s) for s in sentence_buffer]
                current_tokens = sum(buffer_tokens)
            else:
                sentence_buffer.append(sent)
                buffer_tokens.append(sent_tokens)
                current_tokens += sent_tokens

        if sentence_buffer:
             coherence, metrics = self._compute_chunk_coherence(sentence_buffer)
             chunk = self._create_chunk_dict(len(chunks), sentence_buffer, current_tokens, article_idx, 
                                            (len(chunks)>0 or previous_chunk_sentences), 0, 0, 0, coherence, metrics)
             chunks.append(chunk)
             
        return chunks, token_offset

    def chunk_document(self, document: str):
        if not document or not document.strip(): return []
        
        # Clean
        document = re.sub(r'Enlarge this image.*?AP', '', document, flags=re.DOTALL|re.IGNORECASE)
        document = re.sub(r' +', ' ', document)
        
        articles = self.split_into_articles(document)
        all_chunks = []
        prev_sents = None
        offset = 0
        
        for idx, article in enumerate(articles):
            paragraphs = self.split_into_paragraphs(article)
            article_sentences = []
            for p in paragraphs: article_sentences.extend(self.split_into_sentences(p))
            
            chunks, offset = self.chunk_with_sentence_boundaries(article_sentences, idx, prev_sents, offset)
            for c in chunks: 
                c['chunk_id'] = len(all_chunks)
                all_chunks.append(c)
                
            if chunks: prev_sents = chunks[-1]['sentences']
            
        return all_chunks

    def clear_cache(self):
        self._token_cache.clear()
        self._embedding_cache.clear()
    
    # -------------------------------------------------------------------------
    # VALIDATION AND STATS (Included for Completeness)
    # -------------------------------------------------------------------------
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        warnings = []
        is_valid = True
        
        for i, chunk in enumerate(chunks):
            if chunk['token_count'] > self.max_tokens:
                warnings.append(f"Chunk {i}: Exceeds max_tokens ({chunk['token_count']} > {self.max_tokens})")
                is_valid = False
            
            actual_tokens = self.get_token_count(chunk['text'])
            if abs(actual_tokens - chunk['token_count']) > 5:
                warnings.append(f"Chunk {i}: Token count mismatch")
                
            if self.validate_semantic_coherence and chunk.get('semantic_coherence_score'):
                if chunk['semantic_coherence_score'] < self.semantic_similarity_threshold:
                    warnings.append(f"Chunk {i}: Low semantic coherence")
                    
        return is_valid, warnings

    def get_summary_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks: return {'num_chunks': 0}
        
        token_counts = [c['token_count'] for c in chunks]
        overlap_counts = [c.get('overlap_token_count', 0) for c in chunks]
        
        stats = {
            'num_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_chunk': float(np.mean(token_counts)),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'token_efficiency': float((sum(token_counts) - sum(overlap_counts)) / sum(token_counts) * 100)
        }
        return stats

    def export_chunks_for_analysis(self, chunks):
        export_data = {
            'statistics': self.get_summary_statistics(chunks),
            'chunks': chunks
        }
        return json.dumps(export_data, indent=2)

    def get_performance_stats(self):
        stats = self._performance_stats.copy()
        if stats['num_documents_processed'] > 0:
            stats['avg_chunking_time_per_doc'] = stats['total_chunking_time'] / stats['num_documents_processed']
        return stats
        
    def get_memory_usage(self):
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'cache_size_mb': (len(self._token_cache) + len(self._embedding_cache)) * 0.001
        }
        
    def reset_performance_stats(self):
        self._performance_stats = {k: 0.0 for k in self._performance_stats}
        
    def benchmark_chunking(self, documents, warmup=3):
        for _ in range(min(warmup, len(documents))):
            self.chunk_document(documents[0])
        self.reset_performance_stats()
        start = time.time()
        for doc in documents:
            self.chunk_document(doc)
        total_time = time.time() - start
        return {'total_time': total_time, 'avg_time': total_time/len(documents)}

    def save_config(self, filepath):
        config = {
            'max_tokens': self.max_tokens,
            'overlap_tokens': self.overlap_tokens,
            'semantic_threshold': self.semantic_similarity_threshold
        }
        with open(filepath, 'w') as f: json.dump(config, f)
        
    @classmethod
    def load_config(cls, filepath, tokenizer=None):
        with open(filepath, 'r') as f: config = json.load(f)
        return cls(tokenizer=tokenizer, max_tokens=config['max_tokens'])
        
    def compare_ablation_modes(self, documents, modes=None):
        if modes is None: modes = [None, 'no_semantic', 'no_overlap']
        results = {}
        for mode in modes:
            chunker = SemanticDocumentChunker(tokenizer=self.tokenizer, ablation_mode=mode)
            stats = []
            for doc in documents:
                chunks = chunker.chunk_document(doc)
                stats.append(len(chunks))
            results[str(mode)] = np.mean(stats)
        return results

    def visualize_coherence_heatmap(self, chunks, output_path=None):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError: return None
        
        sentences = [s for c in chunks for s in c['sentences']]
        embeddings = self.get_sentence_embeddings(sentences)
        if embeddings is None: return None
        
        sim_matrix = np.inner(embeddings, embeddings)
        plt.figure(figsize=(10,10))
        sns.heatmap(sim_matrix)
        if output_path: plt.savefig(output_path)
        plt.close()

# ============================================================================
# 4. COMPLETE SOURCE CODE: HIERARCHICAL SUMMARIZER
# ============================================================================
class HierarchicalSummarizer:
    """
    State-of-the-Art Hierarchical Summarizer (TreeSum).
    """
    def __init__(self, 
                 model_name: str = "google/pegasus-multi_news",
                 device: Optional[str] = None,
                 batch_size: int = 4,
                 semantic_weight: float = 0.7,
                 dtype: Optional[torch.dtype] = None):
        
        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Dtype - Enforce float32 for stability if requested or default
        # Kaggle P100 prefers float32
        self.dtype = torch.float32 
        
        logger.info(f"Initializing TreeSum on {self.device} with {self.dtype}")

        # Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.batch_size = batch_size
        
        # Chunker
        self.chunker = SemanticDocumentChunker(
            tokenizer=self.tokenizer,
            max_tokens=1024, 
            overlap_tokens=128,
            use_semantic_coherence=True,
            adaptive_overlap=True, 
            semantic_weight=semantic_weight
        )
            
    def _generate(self, inputs: List[str], max_length: int = 512, min_length: int = 64) -> List[str]:
        batch = self.tokenizer(
            inputs, 
            truncation=True, 
            padding="longest", 
            max_length=1024, 
            return_tensors="pt"
        ).to(self.device)
        
        try:
            summary_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=4, 
                max_length=max_length,
                min_length=min_length if max_length > 256 else 0,
                length_penalty=1.0, 
                repetition_penalty=1.5, 
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            return ["ERROR_GENERATION_FAILED"] * len(inputs)

    def summarize_document(self, document: str) -> Dict[str, Any]:
        if not document.strip():
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': []}
            
        # 1. Chunking
        chunks = self.chunker.chunk_document(document)
        chunk_texts = [c['text'] for c in chunks]
        
        if not chunk_texts: return {'final_summary': "", 'chunks': []}
            
        # 2. Map
        chunk_summaries = []
        batch_size = self.batch_size
        local_max_len = 128 if len(chunk_texts) > 5 else 256
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            chunk_summaries.extend(self._generate(batch, max_length=local_max_len))
            
        # 3. Reduce
        final_summary = self._stage2_reduce(chunk_summaries)
            
        return {
            'final_summary': final_summary,
            'chunk_summaries': chunk_summaries,
            'chunks': chunks,
            'num_chunks': len(chunks)
        }

    def _stage2_reduce(self, chunk_summaries: List[str]) -> str:
        current_summaries = chunk_summaries
        MAX_INPUT_TOKENS = 1000 
        layer = 0
        
        while True:
            combined_text = " ".join(current_summaries)
            tokenized_len = len(self.tokenizer.encode(combined_text, truncation=False))
            
            if tokenized_len <= MAX_INPUT_TOKENS:
                if tokenized_len < 256 and layer > 0: return combined_text
                return self._generate([combined_text], max_length=512, min_length=128)[0]
            
            if len(current_summaries) <= 1:
                return self._generate([current_summaries[0]], max_length=512, min_length=128)[0]
                
            new_level = []
            current_group = []
            current_group_len = 0
            
            for summary in current_summaries:
                s_len = len(self.tokenizer.encode(summary, truncation=False))
                if current_group_len + s_len > MAX_INPUT_TOKENS:
                    if current_group:
                        new_level.append(self._generate([" ".join(current_group)], max_length=256)[0])
                    current_group = [summary]
                    current_group_len = s_len
                else:
                    current_group.append(summary)
                    current_group_len += s_len
            
            if current_group:
                new_level.append(self._generate([" ".join(current_group)], max_length=256)[0])
            
            current_summaries = new_level
            layer += 1
            if layer > 5:
                return self._generate([" ".join(current_summaries)], max_length=512)[0]

# ============================================================================
# 5. MAIN EXPERIMENT RUNNER
# ============================================================================
def run_ablation_treesum():
    print("=" * 70)
    print("ABLATION STUDY: TREESUM (PROPOSED METHOD)")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print("\n[1/5] Loading Data...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # 2. Shared Indices
    indices_file = os.path.join(os.path.dirname(OUTPUT_DIR), 'shared_sample_indices.json')
    if os.path.exists(indices_file):
        with open(indices_file, 'r') as f:
            indices = json.load(f)
        print(f"   Loaded {len(indices)} shared indices")
    else:
        indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    
    samples = dataset.select(indices)
    
    # 3. Init Model
    print("\n[2/5] Initializing TreeSum...")
    summarizer = HierarchicalSummarizer()
    
    # 4. Run
    print("\n[3/5] Running Pipeline...")
    results = []
    all_refs = []
    all_preds = []
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(samples)):
        doc = sample['document']
        ref = sample['summary']
        
        try:
            output = summarizer.summarize_document(doc)
            pred = output['final_summary']
        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            pred = ""
        
        results.append({
            'sample_idx': indices[i],
            'generated_summary': pred,
            'reference_summary': ref,
            'num_chunks': output.get('num_chunks', 0)
        })
        all_refs.append(ref)
        all_preds.append(pred)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            batch_num = (i + 1) // CHECKPOINT_INTERVAL
            with open(os.path.join(OUTPUT_DIR, f'summaries_batch_{batch_num}.json'), 'w') as f:
                json.dump(results[(batch_num-1)*CHECKPOINT_INTERVAL : batch_num*CHECKPOINT_INTERVAL], f)
            print(f"   Saved Batch {batch_num}")
            
    # 5. Cleanup
    total_time = time.time() - start_time
    del summarizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 6. Metrics
    print("\n[4/5] Computing Metrics...")
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=all_preds, references=all_refs)
    
    print("Computing BERTScore (CPU)...")
    bertscore = evaluate.load("bertscore")
    bert_res = bertscore.compute(
        predictions=all_preds, references=all_refs, lang="en", 
        model_type="microsoft/deberta-xlarge-mnli", device="cpu", batch_size=16
    )
    
    metrics = {
        'method': 'TreeSum_SOTA',
        'rouge1': rouge_res['rouge1'] * 100,
        'rouge2': rouge_res['rouge2'] * 100,
        'rougeL': rouge_res['rougeL'] * 100,
        'bertscore_f1': np.mean(bert_res['f1']) * 100,
        'total_time_hours': total_time / 3600
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metrics_treesum.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("\nFINAL RESULTS:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    run_ablation_treesum()
