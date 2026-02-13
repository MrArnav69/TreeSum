import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
import nltk
from nltk.tokenize import sent_tokenize
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
import psutil
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)

semantic_available = True

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

@dataclass
class ChunkMetadata:
    chunk_id: int
    text: str
    sentences: List[str]
    token_count: int
    sentence_count: int
    article_indices: List[int]
    has_overlap: bool
    overlap_token_count: int
    semantic_coherence_score: Optional[float] = None
    paragraph_boundaries: List[int] = field(default_factory=list)
    original_token_offsets: Optional[Tuple[int, int]] = None

class SemanticDocumentChunker:
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
                 semantic_weight: float = 1.0,
                 enable_validation: bool = True,
                 validate_overlap_tokens: bool = True,
                 validate_semantic_coherence: bool = True):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name is not None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            from transformers import PegasusTokenizer
            self.tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-multi_news")
        
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.use_sentence_boundaries = use_sentence_boundaries
        self.min_chunk_tokens = min_chunk_tokens
        self.preserve_paragraphs = preserve_paragraphs
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.adaptive_overlap = adaptive_overlap
        self.semantic_weight = semantic_weight
        self.enable_validation = enable_validation
        self.validate_overlap_tokens = validate_overlap_tokens
        self.validate_semantic_coherence = validate_semantic_coherence
        
        self.use_semantic_coherence = use_semantic_coherence and semantic_available
        
        self._performance_stats = {
            'total_chunking_time': 0.0,
            'total_embedding_time': 0.0,
            'total_token_count_time': 0.0,
            'num_documents_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self._semantic_model = None
        self._semantic_model_name = semantic_model
        self._token_cache: Dict[str, int] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        if overlap_tokens >= max_tokens:
            raise ValueError(f"Overlap ({overlap_tokens}) must be less than max_tokens ({max_tokens})")
        if min_chunk_tokens > max_tokens:
            raise ValueError(f"min_chunk_tokens ({min_chunk_tokens}) cannot exceed max_tokens ({max_tokens})")
    
    @property
    def semantic_model(self):
        if self._semantic_model is None and self.use_semantic_coherence:
            model_name = self._semantic_model_name or 'sentence-transformers/all-MiniLM-L6-v2'
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if torch.backends.mps.is_available():
                device = 'mps'
            self._semantic_model = SentenceTransformer(model_name, device=device)
        return self._semantic_model

    def get_token_count(self, text: str, use_cache: bool = True) -> int:
        if use_cache and text in self._token_cache:
            return self._token_cache[text]
        token_count = len(self.tokenizer.tokenize(text))
        if use_cache:
            self._token_cache[text] = token_count
        return token_count
    
    def get_sentence_embeddings(self, sentences: List[str], batch_size: int = 32, 
                                max_cache_size: Optional[int] = None) -> np.ndarray:
        if not self.use_semantic_coherence or self.semantic_model is None:
            return None
        if max_cache_size is not None and len(self._embedding_cache) > max_cache_size:
            cache_items = list(self._embedding_cache.items())
            num_to_remove = len(cache_items) // 5
            for key, _ in cache_items[:num_to_remove]:
                del self._embedding_cache[key]
        uncached_sentences = [s for s in sentences if s not in self._embedding_cache]
        if uncached_sentences:
            show_progress = len(uncached_sentences) > 100
            embeddings = self.semantic_model.encode(
                uncached_sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            for sent, emb in zip(uncached_sentences, embeddings):
                self._embedding_cache[sent] = emb
        result = np.array([self._embedding_cache[s] for s in sentences])
        return result
    
    def compute_semantic_coherence(self, sentences: List[str], global_coherence: bool = False) -> Dict[str, float]:
        if not self.use_semantic_coherence or len(sentences) < 2:
            return {
                'local_coherence': 1.0,
                'global_coherence': 1.0,
                'coherence_variance': 0.0,
                'min_coherence': 1.0,
                'max_coherence': 1.0
            }
        embeddings = self.get_sentence_embeddings(sentences)
        if embeddings is None:
            return {
                'local_coherence': 1.0,
                'global_coherence': 1.0,
                'coherence_variance': 0.0,
                'min_coherence': 1.0,
                'max_coherence': 1.0
            }
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append((sim + 1) / 2)
        local_coherence = np.mean(similarities) if similarities else 1.0
        result = {
            'local_coherence': float(local_coherence),
            'coherence_variance': float(np.var(similarities)) if similarities else 0.0,
            'min_coherence': float(np.min(similarities)) if similarities else 1.0,
            'max_coherence': float(np.max(similarities)) if similarities else 1.0
        }
        if global_coherence and len(embeddings) > 2:
            centroid = np.mean(embeddings, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            centroid_similarities = []
            for emb in embeddings:
                sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * centroid_norm)
                centroid_similarities.append((sim + 1) / 2)
            centroid_coherence = float(np.mean(centroid_similarities))
            if len(embeddings) > 10:
                num_segments = min(5, len(embeddings) // 3)
                segment_size = len(embeddings) // num_segments
                segment_centroids = []
                for i in range(num_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(embeddings)
                    segment_emb = embeddings[start_idx:end_idx]
                    segment_centroid = np.mean(segment_emb, axis=0)
                    segment_centroids.append(segment_centroid)
                segment_similarities = []
                for i in range(len(segment_centroids) - 1):
                    sim = np.dot(segment_centroids[i], segment_centroids[i+1]) / (
                        np.linalg.norm(segment_centroids[i]) * np.linalg.norm(segment_centroids[i+1])
                    )
                    segment_similarities.append((sim + 1) / 2)
                hierarchical_coherence = float(np.mean(segment_similarities)) if segment_similarities else centroid_coherence
                result['global_coherence'] = float(0.6 * centroid_coherence + 0.4 * hierarchical_coherence)
                result['hierarchical_coherence'] = hierarchical_coherence
            else:
                result['global_coherence'] = centroid_coherence
        else:
            result['global_coherence'] = local_coherence
        return result
    
    def find_optimal_overlap_sentences(self, 
                                       previous_sentences: List[str], 
                                       target_overlap_tokens: int) -> List[str]:
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
                    coherence_dict = self.compute_semantic_coherence(overlap_sentences, global_coherence=False)
                    coherence = coherence_dict['local_coherence']
                    if coherence < self.semantic_similarity_threshold:
                        break
                    elif current_tokens >= target_overlap_tokens:
                        break
                else:
                    if current_tokens >= target_overlap_tokens * 0.8:
                        break
        return overlap_sentences
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'Enlarge this image.*?AP', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'toggle caption.*?AP', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()
    
    def split_into_articles(self, document: str) -> List[str]:
        articles = [art.strip() for art in document.split("|||") if len(art.strip()) > 0]
        if len(articles) == 0:
            articles = [document.strip()]
        return articles
    
    def split_into_sentences(self, text: str) -> List[str]:
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 0]
        if len(paragraphs) == 0:
            paragraphs = [text.strip()]
        return paragraphs
    
    def _compute_chunk_coherence(self, sentences: List[str]) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        if not self.use_semantic_coherence:
            return None, None
        use_global = len(sentences) > 5
        coherence_dict = self.compute_semantic_coherence(sentences, global_coherence=use_global)
        coherence_score = coherence_dict['local_coherence']
        return coherence_score, coherence_dict
    
    def _create_chunk_dict(self, 
                          chunk_id: int,
                          sentences: List[str],
                          token_count: int,
                          article_idx: int,
                          has_overlap: bool,
                          overlap_token_count: int,
                          token_start: int,
                          token_end: int,
                          coherence_score: Optional[float] = None,
                          coherence_dict: Optional[Dict[str, float]] = None,
                          semantic_weight: Optional[float] = None) -> Dict[str, Any]:
        return {
            'chunk_id': chunk_id,
            'text': ' '.join(sentences),
            'sentences': sentences.copy(),
            'token_count': token_count,
            'sentence_count': len(sentences),
            'article_indices': [article_idx],
            'has_overlap': has_overlap,
            'overlap_token_count': overlap_token_count,
            'semantic_coherence_score': coherence_score,
            'semantic_coherence_metrics': coherence_dict,
            'original_token_offsets': (token_start, token_end),
            'semantic_weight': semantic_weight if semantic_weight is not None else self.semantic_weight
        }
    
    def _should_create_chunk(self, current_tokens: int, num_chunks: int) -> bool:
        return current_tokens >= self.min_chunk_tokens or num_chunks == 0
    
    def _compute_lexical_similarity(self, sent1: str, sent2: str) -> float:
        tokens1 = set(self.tokenizer.tokenize(sent1.lower()))
        tokens2 = set(self.tokenizer.tokenize(sent2.lower()))
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / union if union > 0 else 0.0

    def _find_optimal_split_point(self, sentences: List[str], current_tokens: List[int], alpha: Optional[float] = None) -> int:
        if alpha is None:
            alpha = self.semantic_weight
        if not self.use_semantic_coherence or self.semantic_model is None:
            return len(sentences) 
        n_sentences = len(sentences)
        if n_sentences < 3:
            return n_sentences
        embeddings = self.get_sentence_embeddings(sentences)
        if embeddings is None:
            return n_sentences
        semantic_sims = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            semantic_sims.append(sim)
        semantic_sims = np.array(semantic_sims)
        lexical_sims = []
        for i in range(n_sentences - 1):
            lex_sim = self._compute_lexical_similarity(sentences[i], sentences[i+1])
            lexical_sims.append(lex_sim)
        lexical_sims = np.array(lexical_sims)
        if np.max(lexical_sims) > 0:
            lexical_sims = lexical_sims / np.max(lexical_sims)

        hybrid_sims = alpha * semantic_sims + (1.0 - alpha) * lexical_sims
        if len(hybrid_sims) > 4:
            smoothed_sims = gaussian_filter1d(hybrid_sims, sigma=1.0)
        else:
            smoothed_sims = hybrid_sims
        cum_tokens = np.cumsum(current_tokens)
        total_tokens = cum_tokens[-1]
        valid_indices = []
        for i in range(len(smoothed_sims)):
            tokens_at_split = cum_tokens[i]
            if tokens_at_split >= self.min_chunk_tokens:
                valid_indices.append(i)
        if not valid_indices:
            return n_sentences
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
            best_split_idx = candidates[0][1]
            return best_split_idx + 1 
        else:
            lowest_idx = valid_indices[np.argmin(valid_signal)]
            return lowest_idx + 1

    def chunk_with_sentence_boundaries(self, 
                                       sentences: List[str], 
                                       article_idx: int,
                                       previous_chunk_sentences: Optional[List[str]] = None,
                                       original_token_start: int = 0,
                                       semantic_weight: Optional[float] = None) -> Tuple[List[Dict], int]:
        chunks = []
        sentence_buffer = []
        buffer_tokens = []
        current_tokens = 0
        overlap_sentences = []
        if previous_chunk_sentences:
            overlap_sentences = self.find_optimal_overlap_sentences(
                previous_chunk_sentences, 
                self.overlap_tokens
            )
            for s in overlap_sentences:
                t_count = self.get_token_count(s)
                sentence_buffer.append(s)
                buffer_tokens.append(t_count)
                current_tokens += t_count
        token_offset = original_token_start
        for sent in sentences:
            sent_tokens = self.get_token_count(sent)
            if current_tokens + sent_tokens > self.max_tokens:
                split_idx = self._find_optimal_split_point(sentence_buffer, buffer_tokens, alpha=semantic_weight)
                chunk_sentences = sentence_buffer[:split_idx]
                chunk_token_count = sum(buffer_tokens[:split_idx])
                coherence_score, coherence_dict = self._compute_chunk_coherence(chunk_sentences)
                actual_overlap_len = 0
                if previous_chunk_sentences:
                     matches = 0
                     for i in range(min(len(overlap_sentences), len(chunk_sentences))):
                         if chunk_sentences[i] == overlap_sentences[i]:
                             matches += 1
                     actual_overlap_len = sum(buffer_tokens[:matches])
                chunk = self._create_chunk_dict(
                    chunk_id=len(chunks),
                    sentences=chunk_sentences,
                    token_count=chunk_token_count,
                    article_idx=article_idx,
                    has_overlap=(len(chunks) > 0 or previous_chunk_sentences is not None),
                    overlap_token_count=actual_overlap_len if len(chunks) == 0 else sum(self.get_token_count(s) for s in self.find_optimal_overlap_sentences(chunks[-1]['sentences'], self.overlap_tokens)),
                    token_start=token_offset - current_tokens,
                    token_end=token_offset - current_tokens + chunk_token_count,
                    coherence_score=coherence_score,
                    coherence_dict=coherence_dict,
                    semantic_weight=semantic_weight
                )
                chunks.append(chunk)
                new_overlap = self.find_optimal_overlap_sentences(chunk_sentences, self.overlap_tokens)
                remaining_sentences = sentence_buffer[split_idx:]
                remaining_tokens = buffer_tokens[split_idx:]
                sentence_buffer = new_overlap + remaining_sentences
                buffer_tokens = [self.get_token_count(s) for s in sentence_buffer]
                current_tokens = sum(buffer_tokens)
                sentence_buffer.append(sent)
                buffer_tokens.append(sent_tokens)
                current_tokens += sent_tokens
                token_offset += sent_tokens
            else:
                sentence_buffer.append(sent)
                buffer_tokens.append(sent_tokens)
                current_tokens += sent_tokens
                token_offset += sent_tokens
        if sentence_buffer:
             coherence_score, coherence_dict = self._compute_chunk_coherence(sentence_buffer)
             chunk = self._create_chunk_dict(
                chunk_id=len(chunks),
                sentences=sentence_buffer,
                token_count=current_tokens,
                article_idx=article_idx,
                has_overlap=(len(chunks)>0 or previous_chunk_sentences is not None),
                overlap_token_count=0,
                token_start=token_offset-current_tokens,
                token_end=token_offset,
                coherence_score=coherence_score,
                coherence_dict=coherence_dict,
                semantic_weight=semantic_weight
             )
             chunks.append(chunk)
        return chunks, token_offset
    
    def _chunk_by_tokens_sentence_aware(self, text: str, article_idx: int) -> List[Dict]:
        sentences = self.split_into_sentences(text)
        chunks = []
        current_sentences = []
        current_tokens = 0
        for sent in sentences:
            sent_tokens = self.get_token_count(sent)
            if current_tokens + sent_tokens > self.max_tokens:
                if current_tokens >= self.min_chunk_tokens or len(chunks) == 0:
                    chunk_text = ' '.join(current_sentences)
                    chunks.append({
                        'chunk_id': len(chunks),
                        'text': chunk_text,
                        'sentences': current_sentences.copy(),
                        'token_count': current_tokens,
                        'sentence_count': len(current_sentences),
                        'article_indices': [article_idx],
                        'has_overlap': len(chunks) > 0,
                        'overlap_token_count': self.overlap_tokens if len(chunks) > 0 else 0
                    })
                    current_sentences = [sent]
                    current_tokens = sent_tokens
                else:
                    current_sentences.append(sent)
                    current_tokens += sent_tokens
            else:
                current_sentences.append(sent)
                current_tokens += sent_tokens
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append({
                'chunk_id': len(chunks),
                'text': chunk_text,
                'sentences': current_sentences.copy(),
                'token_count': current_tokens,
                'sentence_count': len(current_sentences),
                'article_indices': [article_idx],
                'has_overlap': len(chunks) > 0,
                'overlap_token_count': self.overlap_tokens if len(chunks) > 0 else 0
            })
        return chunks
    
    def chunk_document_batch(self, documents: List[str], 
                             batch_size: int = 10,
                             clear_cache_every: Optional[int] = None,
                             semantic_weight: Optional[float] = None) -> List[List[Dict[str, Any]]]:
        if clear_cache_every is None:
            clear_cache_every = batch_size
        all_chunks = []
        iterable = tqdm(documents, desc="Chunking documents")
        for i, doc in enumerate(iterable):
            chunks = self.chunk_document(doc, semantic_weight=semantic_weight)
            all_chunks.append(chunks)
            if (i + 1) % clear_cache_every == 0:
                self.clear_cache()
                if self.use_semantic_coherence:
                    max_embeddings = 10000
                    if len(self._embedding_cache) > max_embeddings:
                        cache_items = list(self._embedding_cache.items())
                        num_to_remove = len(cache_items) // 3
                        for key, _ in cache_items[:num_to_remove]:
                            del self._embedding_cache[key]
        return all_chunks
    
    def chunk_document(self, document: str, semantic_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        if not document or not document.strip():
            return []
        cleaned_doc = self.clean_text(document)
        if not cleaned_doc.strip():
            return []
        articles = self.split_into_articles(cleaned_doc)
        if len(articles) == 1:
            sentences = self.split_into_sentences(articles[0])
            if len(sentences) == 1:
                sent_tokens = self.get_token_count(sentences[0])
                return [{
                    'chunk_id': 0,
                    'text': sentences[0],
                    'sentences': sentences,
                    'token_count': sent_tokens,
                    'sentence_count': 1,
                    'article_indices': [0],
                    'has_overlap': False,
                    'overlap_token_count': 0,
                    'semantic_coherence_score': 1.0,
                    'semantic_coherence_metrics': {'local_coherence': 1.0, 'global_coherence': 1.0},
                    'original_token_offsets': (0, sent_tokens)
                }]
        all_chunks = []
        previous_chunk_sentences = None
        token_offset = 0
        for article_idx, article in enumerate(articles):
            if self.preserve_paragraphs:
                paragraphs = self.split_into_paragraphs(article)
                article_sentences = []
                paragraph_boundaries = []
                for para in paragraphs:
                    para_sentences = self.split_into_sentences(para)
                    article_sentences.extend(para_sentences)
                    paragraph_boundaries.append(len(article_sentences))
            else:
                article_sentences = self.split_into_sentences(article)
                paragraph_boundaries = []
            if self.use_sentence_boundaries:
                article_chunks, token_offset = self.chunk_with_sentence_boundaries(
                    article_sentences,
                    article_idx,
                    previous_chunk_sentences,
                    token_offset,
                    semantic_weight=semantic_weight
                )
            else:
                article_chunks = self._chunk_by_tokens_sentence_aware(article, article_idx)
                for chunk in article_chunks:
                    token_offset += chunk['token_count']
            for chunk in article_chunks:
                chunk['chunk_id'] = len(all_chunks)
                if paragraph_boundaries:
                    chunk['paragraph_boundaries'] = paragraph_boundaries
                all_chunks.append(chunk)
            if article_chunks:
                previous_chunk_sentences = article_chunks[-1].get('sentences', [])
        return all_chunks
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        warnings_list = []
        is_valid = True
        if self.enable_validation:
            for i, chunk in enumerate(chunks):
                if chunk['token_count'] > self.max_tokens:
                    warnings_list.append(f"Chunk {i}: Exceeds max_tokens ({chunk['token_count']} > {self.max_tokens})")
                    is_valid = False
                if chunk['token_count'] < self.min_chunk_tokens and i < len(chunks) - 1:
                    warnings_list.append(f"Chunk {i}: Below min_chunk_tokens ({chunk['token_count']} < {self.min_chunk_tokens})")
                actual_tokens = self.get_token_count(chunk['text'])
                if abs(actual_tokens - chunk['token_count']) > 5:
                    warnings_list.append(f"Chunk {i}: Token count mismatch (reported={chunk['token_count']}, actual={actual_tokens})")
                if self.use_sentence_boundaries and 'sentences' in chunk:
                    reconstructed = ' '.join(chunk['sentences'])
                    if reconstructed.strip() != chunk['text'].strip():
                        warnings_list.append(f"Chunk {i}: Sentence reconstruction mismatch")
                if self.validate_semantic_coherence and self.use_semantic_coherence:
                    coherence = chunk.get('semantic_coherence_score')
                    if coherence is not None and coherence < self.semantic_similarity_threshold:
                        warnings_list.append(f"Chunk {i}: Low semantic coherence ({coherence:.3f} < {self.semantic_similarity_threshold})")
                if self.validate_overlap_tokens and i > 0 and chunk.get('has_overlap'):
                    prev_chunk = chunks[i-1]
                    prev_text = prev_chunk.get('text', '')
                    current_text = chunk.get('text', '')
                    reported_overlap = chunk.get('overlap_token_count', 0)
                    if prev_text and current_text and reported_overlap > 0:
                        prev_tokens = self.tokenizer.tokenize(prev_text)
                        current_tokens = self.tokenizer.tokenize(current_text)
                        max_overlap = 0
                        check_length = min(len(prev_tokens), len(current_tokens), reported_overlap + 50)
                        for k in range(max(0, len(prev_tokens) - check_length), len(prev_tokens)):
                            for j in range(min(check_length, len(current_tokens))):
                                if k + j < len(prev_tokens) and j < len(current_tokens):
                                    if prev_tokens[k + j] == current_tokens[j]:
                                        match_len = 1
                                        while (k + j + match_len < len(prev_tokens) and 
                                               j + match_len < len(current_tokens) and
                                               prev_tokens[k + j + match_len] == current_tokens[j + match_len]):
                                            match_len += 1
                                        if match_len > max_overlap:
                                            max_overlap = match_len
                        actual_overlap_count = max_overlap
                        tolerance = max(20, int(reported_overlap * 0.20))
                        if abs(actual_overlap_count - reported_overlap) > tolerance:
                            warnings_list.append(f"Chunk {i}: Overlap token mismatch (reported={reported_overlap}, actual={actual_overlap_count}, overlap_ratio={actual_overlap_count/len(current_tokens)*100:.1f}%)")
                        min_chunk_size = min(len(prev_tokens), len(current_tokens))
                        if min_chunk_size > 0:
                            overlap_ratio = actual_overlap_count / min_chunk_size
                            if overlap_ratio < 0.05 and reported_overlap > 0:
                                warnings_list.append(f"Chunk {i}: Overlap too small ({overlap_ratio*100:.1f}% of smaller chunk)")
                if i > 0:
                    overlap_tokens_val = chunk.get('overlap_token_count', 0)
                    min_chunk_tokens_val = min(chunk['token_count'], chunks[i-1]['token_count'])
                    if overlap_tokens_val > min_chunk_tokens_val * 0.5:
                        warnings_list.append(f"Chunk {i}: Excessive overlap ({overlap_tokens_val} tokens, {overlap_tokens_val/min_chunk_tokens_val*100:.1f}% of smaller chunk)")
        return is_valid, warnings_list
    
    def get_summary_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {'num_chunks': 0}
        token_counts = [c['token_count'] for c in chunks]
        overlap_counts = [c.get('overlap_token_count', 0) for c in chunks if c.get('has_overlap', False)]
        sentence_counts = [c.get('sentence_count', 0) for c in chunks if 'sentence_count' in c]
        coherence_scores = [c.get('semantic_coherence_score') for c in chunks 
                           if c.get('semantic_coherence_score') is not None]
        global_coherence_scores = []
        coherence_variances = []
        for c in chunks:
            metrics = c.get('semantic_coherence_metrics')
            if metrics:
                if 'global_coherence' in metrics:
                    global_coherence_scores.append(metrics['global_coherence'])
                if 'coherence_variance' in metrics:
                    coherence_variances.append(metrics['coherence_variance'])
        stats = {
            'num_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'total_overlap_tokens': sum(overlap_counts),
            'net_tokens': sum(token_counts) - sum(overlap_counts),
            'avg_tokens_per_chunk': float(np.mean(token_counts)),
            'std_tokens_per_chunk': float(np.std(token_counts)),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'chunks_with_overlap': sum(1 for c in chunks if c.get('has_overlap', False)),
            'avg_overlap_tokens': float(np.mean(overlap_counts)) if overlap_counts else 0.0,
            'token_efficiency': float((sum(token_counts) - sum(overlap_counts)) / sum(token_counts) * 100) 
                              if sum(token_counts) > 0 else 0.0
        }
        if sentence_counts:
            stats.update({
                'total_sentences': sum(sentence_counts),
                'avg_sentences_per_chunk': float(np.mean(sentence_counts)),
                'min_sentences': min(sentence_counts),
                'max_sentences': max(sentence_counts)
            })
        if coherence_scores:
            stats.update({
                'avg_local_coherence': float(np.mean(coherence_scores)),
                'min_local_coherence': float(np.min(coherence_scores)),
                'max_local_coherence': float(np.max(coherence_scores))
            })
        if global_coherence_scores:
            stats.update({
                'avg_global_coherence': float(np.mean(global_coherence_scores)),
                'min_global_coherence': float(np.min(global_coherence_scores)),
                'max_global_coherence': float(np.max(global_coherence_scores))
            })
        if coherence_variances:
            stats.update({
                'avg_coherence_variance': float(np.mean(coherence_variances))
            })
        return stats
    
    def export_chunks_for_analysis(self, chunks: List[Dict[str, Any]]) -> str:
        import json
        export_data = {
            'chunking_config': {
                'max_tokens': self.max_tokens,
                'overlap_tokens': self.overlap_tokens,
                'use_sentence_boundaries': self.use_sentence_boundaries,
                'min_chunk_tokens': self.min_chunk_tokens,
                'preserve_paragraphs': self.preserve_paragraphs,
                'use_semantic_coherence': self.use_semantic_coherence,
                'semantic_similarity_threshold': self.semantic_similarity_threshold,
                'adaptive_overlap': self.adaptive_overlap
            },
            'statistics': self.get_summary_statistics(chunks),
            'chunks': chunks
        }
        return json.dumps(export_data, indent=2)
    
    def clear_cache(self):
        self._token_cache.clear()
        self._embedding_cache.clear()
    
    def get_model_integration_info(self) -> Dict[str, Any]:
        return {
            'tokenizer_name': self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else str(type(self.tokenizer)),
            'max_tokens': self.max_tokens,
            'overlap_tokens': self.overlap_tokens,
            'semantic_model': self.semantic_model.get_sentence_embedding_dimension() if self.semantic_model else None,
            'use_semantic_coherence': self.use_semantic_coherence,
            'config_hash': hash((self.max_tokens, self.overlap_tokens, self.min_chunk_tokens))
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        stats = self._performance_stats.copy()
        if stats['num_documents_processed'] > 0:
            stats['avg_chunking_time_per_doc'] = stats['total_chunking_time'] / stats['num_documents_processed']
            stats['avg_embedding_time_per_doc'] = stats['total_embedding_time'] / stats['num_documents_processed']
            stats['avg_token_count_time_per_doc'] = stats['total_token_count_time'] / stats['num_documents_processed']
            total_cache_ops = stats['cache_hits'] + stats['cache_misses']
            if total_cache_ops > 0:
                stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_ops
            else:
                stats['cache_hit_rate'] = 0.0
        return stats
    
    def get_memory_usage(self) -> Dict[str, float]:
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'cache_size_mb': (len(self._token_cache) + len(self._embedding_cache)) * 0.001
        }
    
    def reset_performance_stats(self):
        self._performance_stats = {
            'total_chunking_time': 0.0,
            'total_embedding_time': 0.0,
            'total_token_count_time': 0.0,
            'num_documents_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def benchmark_chunking(self, documents: List[str], warmup: int = 3) -> Dict[str, Any]:
        import time
        for i in range(min(warmup, len(documents))):
            self.chunk_document(documents[i])
        self.reset_performance_stats()
        start_time = time.time()
        for doc in documents:
            self.chunk_document(doc)
        end_time = time.time()
        total_time = end_time - start_time
        return {
            'num_documents': len(documents),
            'total_time_seconds': total_time,
            'avg_time_per_document': total_time / len(documents) if documents else 0,
            'throughput_docs_per_sec': len(documents) / total_time if total_time > 0 else 0,
            'performance_stats': self.get_performance_stats()
        }
    
    def save_config(self, filepath: str):
        import json
        config = {
            'max_tokens': self.max_tokens,
            'overlap_tokens': self.overlap_tokens,
            'use_sentence_boundaries': self.use_sentence_boundaries,
            'min_chunk_tokens': self.min_chunk_tokens,
            'preserve_paragraphs': self.preserve_paragraphs,
            'use_semantic_coherence': self.use_semantic_coherence,
            'semantic_similarity_threshold': self.semantic_similarity_threshold,
            'adaptive_overlap': self.adaptive_overlap,
            'validate_chunks': self.enable_validation,
            'validate_overlap_tokens': self.validate_overlap_tokens,
            'validate_semantic_coherence': self.validate_semantic_coherence,
            'tokenizer_name': self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else str(type(self.tokenizer)),
            'semantic_model': self.semantic_model.get_sentence_embedding_dimension() if self.semantic_model else None
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str, tokenizer=None, model_name: Optional[str] = None):
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls(
            tokenizer=tokenizer,
            model_name=model_name or config.get('tokenizer_name'),
            max_tokens=config['max_tokens'],
            overlap_tokens=config['overlap_tokens'],
            use_sentence_boundaries=config['use_sentence_boundaries'],
            min_chunk_tokens=config['min_chunk_tokens'],
            preserve_paragraphs=config['preserve_paragraphs'],
            use_semantic_coherence=config['use_semantic_coherence'],
            semantic_similarity_threshold=config['semantic_similarity_threshold'],
            adaptive_overlap=config['adaptive_overlap'],
            enable_validation=config.get('validate_chunks', True),
            validate_overlap_tokens=config.get('validate_overlap_tokens', True),
            validate_semantic_coherence=config.get('validate_semantic_coherence', True)
        )
