"""
================================================================================
TREESUM ABLATION STUDY: FLAT 1024-TOKEN CHUNKING WITH 128-TOKEN OVERLAP
================================================================================

Baseline 2 for the Chunking Strategy Ablation Study.

This script uses FLAT 1024-token chunking with:
- 128-token overlap (sliding window)
- NO semantic awareness
- NO sentence boundary preservation

The hierarchical tree-reduction architecture is IDENTICAL to TreeSum.
The cleaning logic is IDENTICAL to TreeSum.

Target: 1000 samples on P100 GPU (Kaggle)
Estimated Runtime: ~9 hours (slightly more chunks due to overlap)

Author: Arnav Gupta
Date: 2026-02-05
================================================================================
"""

import sys
import subprocess
import os

# ============================================================================
# ENVIRONMENT SETUP (Self-Installing for Kaggle/Colab)
# ============================================================================
def setup_environment():
    """Installs missing dependencies and sets up necessary resources."""
    required_packages = [
        "transformers", 
        "datasets", 
        "evaluate", 
        "rouge_score", 
        "bert_score",
        "accelerate",
        "sentencepiece"  # Required for Pegasus slow tokenizer
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    # NLTK Data for ROUGE
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)
    
    print("✓ Environment setup complete.")

# Run setup
setup_environment()

import torch
from transformers import PegasusForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
import evaluate
from typing import List, Dict, Tuple
import json
import re
import logging
from tqdm import tqdm
import time
import random
import nltk

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_SEED = 42
NUM_SAMPLES = 1000
BATCH_SIZE = 4  # Reduced from 8 for stability on P100
MAX_TOKENS = 1024  # Strict limit (Pegasus max positional embeddings)
OVERLAP_TOKENS = 128  # Sliding window overlap
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N samples

# Kaggle-compatible output path (uses /kaggle/working/ on Kaggle, ./results locally)
if os.path.exists('/kaggle/working'):
    OUTPUT_DIR = "/kaggle/working/results_flat_overlap"
else:
    OUTPUT_DIR = "./results_flat_overlap"

# Reproducibility
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# FLAT CHUNKER WITH 128-TOKEN OVERLAP (SLIDING WINDOW)
# ============================================================================
class FlatChunkerOverlap:
    """
    Baseline Chunker 2: 1024-token chunks with 128-token sliding overlap.
    
    This chunker:
    1. Cleans the document (IDENTICAL to TreeSum cleaning)
    2. Tokenizes the ENTIRE document
    3. Uses a sliding window: [0:1024], [896:1920], [1792:2816], ...
    4. Decodes each chunk back to text
    
    This is the "standard" chunking approach used by most RAG systems.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = MAX_TOKENS
        self.overlap = OVERLAP_TOKENS
        self.step = self.max_tokens - self.overlap  # 896 tokens per step
    
    def _clean_document(self, text: str) -> str:
        """
        Clean and normalize input text (IDENTICAL to TreeSum cleaning).
        
        Removes:
        - Image/caption metadata from Multi-News
        - Bracketed annotations
        - Excessive whitespace
        """
        # Remove image metadata (common in Multi-News)
        text = re.sub(r'Enlarge this image.*?AP', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'toggle caption.*?AP', '', text, flags=re.IGNORECASE)
        
        # Remove bracketed annotations
        text = re.sub(r'\[.*?\]', '', text)
        
        # Normalize whitespace
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        
        return text.strip()
    
    def chunk_document(self, document: str) -> List[Dict]:
        """
        Split document into 1024-token chunks with 128-token overlap.
        
        Uses sliding window approach:
        - Window 1: tokens[0:1024]
        - Window 2: tokens[896:1920] (128-token overlap with Window 1)
        - Window 3: tokens[1792:2816] (128-token overlap with Window 2)
        - ...
        
        Returns:
            List of dicts with 'text' key (for compatibility with HierarchicalSummarizer)
        """
        # 1. Clean (IDENTICAL to TreeSum)
        document = self._clean_document(document)
        
        if not document:
            return []
        
        # 2. Tokenize entire document (no truncation)
        tokens = self.tokenizer.encode(document, add_special_tokens=False)
        
        # 3. Sliding window chunking
        chunks = []
        for i in range(0, len(tokens), self.step):
            chunk_tokens = tokens[i : i + self.max_tokens]
            
            # 4. Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text, 
                    'token_count': len(chunk_tokens),
                    'has_overlap': i > 0  # All chunks except first have overlap
                })
            
            # Stop if we've processed the entire document
            if i + self.max_tokens >= len(tokens):
                break
        
        return chunks

# ============================================================================
# HIERARCHICAL SUMMARIZER (IDENTICAL TO TREESUM)
# ============================================================================
class HierarchicalSummarizerOverlap:
    """
    Hierarchical Summarizer with Overlap Chunking.
    
    The tree-reduction logic is IDENTICAL to the main TreeSum model.
    Only the chunker is replaced with FlatChunkerOverlap.
    """
    
    def __init__(self, 
                 model_name: str = "google/pegasus-multi_news",
                 device: str = None,
                 batch_size: int = BATCH_SIZE,
                 dtype: torch.dtype = None):
        
        # Device Selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing HierarchicalSummarizerOverlap on {self.device}")
        
        # Dtype Selection (Fixed: float32 to prevent hallucinations)
        self.dtype = torch.float32
        
        # Load Model & Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.batch_size = batch_size
        
        # Initialize Overlap Chunker (BASELINE 2)
        self.chunker = FlatChunkerOverlap(self.tokenizer)
        
    def _generate(self, inputs: List[str], max_length: int = 512, min_length: int = 64) -> List[str]:
        """
        Low-level generation with standard Pegasus beam search parameters.
        IDENTICAL to TreeSum generation settings.
        """
        batch = self.tokenizer(
            inputs, 
            truncation=True, 
            padding="longest", 
            max_length=1024, 
            return_tensors="pt"
        ).to(self.device)
        
        try:
            with torch.no_grad():
                summary_ids = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=8, 
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=0.8, 
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            return summaries
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Do NOT return empty strings silently if it's a critical error
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return ["ERROR_GENERATION_FAILED"] * len(inputs)
    
    def summarize_document(self, document: str) -> Dict:
        """
        Execute the full hierarchical pipeline.
        """
        if not document.strip():
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': []}
            
        # Stage 1: Overlap Chunking
        chunks = self.chunker.chunk_document(document)
        chunk_texts = [c['text'] for c in chunks]
        
        if not chunk_texts:
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': []}
            
        # Stage 2: Map (Chunk Summarization)
        chunk_summaries = self._stage1_map_summaries(chunk_texts)
            
        # Stage 3: Reduce (Tree Aggregation)
        final_summary, concatenated_summary = self._stage2_reduce_summaries(chunk_summaries)
            
        return {
            'final_summary': final_summary,
            'chunk_summaries': chunk_summaries,
            'chunks': chunks,
            'num_chunks': len(chunks)
        }

    def _stage1_map_summaries(self, chunk_texts: List[str]) -> List[str]:
        """Stage 1 (Map): Summarize each chunk independently."""
        chunk_summaries = []
        local_max_len = 128 if len(chunk_texts) > 5 else 256
        
        for i in range(0, len(chunk_texts), self.batch_size):
            batch = chunk_texts[i:i + self.batch_size]
            summaries = self._generate(batch, max_length=local_max_len)
            chunk_summaries.extend(summaries)
            
        return chunk_summaries

    def _stage2_reduce_summaries(self, chunk_summaries: List[str]) -> Tuple[str, str]:
        """
        Stage 2 (Reduce): Recursive tree reduction.
        IDENTICAL to TreeSum architecture.
        """
        concatenated_intermediate = " ".join(chunk_summaries)
        current_summaries = chunk_summaries
        layer = 0
        MAX_INPUT_TOKENS = 1000
        
        while True:
            combined_text = " ".join(current_summaries)
            tokenized_len = len(self.tokenizer.encode(combined_text, truncation=False))
            
            logger.info(f"Reduction Layer {layer}: {len(current_summaries)} chunks, {tokenized_len} tokens")
            
            # Base Case
            if tokenized_len <= MAX_INPUT_TOKENS:
                if tokenized_len < 256 and layer > 0:
                    return combined_text, concatenated_intermediate
                
                final_summary_list = self._generate([combined_text], max_length=512, min_length=128)
                return final_summary_list[0], concatenated_intermediate
            
            # Edge Case: Single long summary
            if len(current_summaries) <= 1:
                final_summary_list = self._generate([current_summaries[0]], max_length=512, min_length=128)
                return final_summary_list[0], concatenated_intermediate
            
            # Tree Grouping (Bin Packing)
            new_level_summaries = []
            current_group = []
            current_group_len = 0
            
            for summary in current_summaries:
                s_len = len(self.tokenizer.encode(summary, truncation=False))
                
                if current_group_len + s_len > MAX_INPUT_TOKENS:
                    if current_group:
                        group_text = " ".join(current_group)
                        new_summary = self._generate([group_text], max_length=256)[0]
                        new_level_summaries.append(new_summary)
                    
                    current_group = [summary]
                    current_group_len = s_len
                else:
                    current_group.append(summary)
                    current_group_len += s_len
            
            if current_group:
                group_text = " ".join(current_group)
                new_summary = self._generate([group_text], max_length=256)[0]
                new_level_summaries.append(new_summary)
            
            current_summaries = new_level_summaries
            layer += 1
            
            if layer > 5:
                logger.warning("Max reduction layers reached.")
                final_text = " ".join(current_summaries)
                final_summary_list = self._generate([final_text], max_length=512)
                return final_summary_list[0], concatenated_intermediate

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================
def run_ablation_flat_overlap():
    """
    Run the Flat 1024 + 128 Overlap Chunking Ablation Study.
    """
    print("=" * 70)
    print("ABLATION STUDY: FLAT 1024-TOKEN CHUNKING WITH 128 OVERLAP")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Dataset
    print("\n[1/5] Loading Multi-News Test Split...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # 2. Random Sampling (Reproducible & Shared Across All Ablations)
    # CRITICAL: Uses the SAME indices as run_flat_1024.py for valid comparison
    indices_file = os.path.join(os.path.dirname(OUTPUT_DIR), 'shared_sample_indices.json')
    
    if os.path.exists(indices_file):
        print(f"   Loading shared indices from {indices_file}")
        with open(indices_file, 'r') as f:
            indices = json.load(f)
        print(f"   Loaded {len(indices)} pre-selected sample indices")
    else:
        print(f"   Generating new random indices (seed={RANDOM_SEED})")
        random.seed(RANDOM_SEED)
        indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
        
        # Save indices for other ablation runs to use
        os.makedirs(os.path.dirname(indices_file) if os.path.dirname(indices_file) else '.', exist_ok=True)
        with open(indices_file, 'w') as f:
            json.dump(indices, f)
        print(f"   Saved shared indices to {indices_file}")
    
    samples = dataset.select(indices)
    print(f"   Selected {len(samples)} samples for ablation study")
    
    # Save indices copy in output dir for reference
    with open(os.path.join(OUTPUT_DIR, 'sample_indices.json'), 'w') as f:
        json.dump(indices, f)
    
    # 3. Initialize Summarizer
    print("\n[2/5] Initializing Hierarchical Summarizer (Overlap Chunking)...")
    summarizer = HierarchicalSummarizerOverlap()
    
    # 4. Load Metrics
    print("\n[3/5] Loading Evaluation Metrics...")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    # 5. Run Summarization
    print("\n[4/5] Running Summarization Pipeline...")
    results = []
    all_predictions = []
    all_references = []
    
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(samples, desc="Processing Documents")):
        doc = sample['document']
        ref = sample['summary']
        
        try:
            output = summarizer.summarize_document(doc)
            pred = output['final_summary']
            num_chunks = output.get('num_chunks', 0)
        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            pred = ""
            num_chunks = 0
        
        results.append({
            'sample_idx': indices[i],
            'generated_summary': pred,
            'reference_summary': ref,
            'num_chunks': num_chunks,
            'doc_length': len(doc.split())
        })
        
        all_predictions.append(pred)
        all_references.append(ref)

        # CRITICAL: Clear cache after EACH document to prevent fragmentation/OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Checkpoint every CHECKPOINT_INTERVAL samples
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(samples) - i - 1) / rate
            checkpoint_num = (i + 1) // CHECKPOINT_INTERVAL
            print(f"\n[Checkpoint {checkpoint_num}] {i+1}/{len(samples)} | Rate: {rate:.2f} docs/s | ETA: {remaining/3600:.1f}h")
            
            # Extract ONLY the summaries from this batch (100 at a time)
            batch_start = (checkpoint_num - 1) * CHECKPOINT_INTERVAL
            batch_end = checkpoint_num * CHECKPOINT_INTERVAL
            batch_summaries = results[batch_start:batch_end]
            
            # Save this batch's summaries as a separate file
            batch_file = os.path.join(OUTPUT_DIR, f'summaries_batch_{checkpoint_num}.json')
            with open(batch_file, 'w') as f:
                json.dump(batch_summaries, f, indent=2)
            print(f"   ✓ Batch {checkpoint_num} summaries saved ({len(batch_summaries)} samples)")
            
            # Also save cumulative progress for resume capability
            progress_data = {
                'checkpoint_num': checkpoint_num,
                'samples_processed': i + 1,
                'elapsed_hours': elapsed / 3600,
                'rate_docs_per_sec': rate
            }
            with open(os.path.join(OUTPUT_DIR, 'progress.json'), 'w') as f:
                json.dump(progress_data, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n✅ Summarization Complete! Total Time: {total_time/3600:.2f} hours")
    
    # 6. Free GPU Memory (Critical for BERTScore)
    print("\n[5/7] Freeing GPU Memory...")
    del summarizer.model
    del summarizer.tokenizer
    del summarizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("   ✓ GPU memory cleared")
    
    # 7. Compute Metrics
    print("\n[6/7] Computing ROUGE Scores...")
    rouge_results = rouge.compute(predictions=all_predictions, references=all_references)
    
    print("Computing BERTScore (on CPU to prevent OOM)...")
    bert_results = bertscore.compute(
        predictions=all_predictions, 
        references=all_references, 
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        device="cpu",
        batch_size=16
    )
    
    # 8. Aggregate Metrics
    metrics = {
        'method': 'Flat_1024_128_Overlap',
        'num_samples': len(results),
        'rouge1': rouge_results['rouge1'] * 100,
        'rouge2': rouge_results['rouge2'] * 100,
        'rougeL': rouge_results['rougeL'] * 100,
        'rougeLsum': rouge_results['rougeLsum'] * 100,
        'bertscore_precision': sum(bert_results['precision']) / len(bert_results['precision']) * 100,
        'bertscore_recall': sum(bert_results['recall']) / len(bert_results['recall']) * 100,
        'bertscore_f1': sum(bert_results['f1']) / len(bert_results['f1']) * 100,
        'avg_chunks_per_doc': sum(r['num_chunks'] for r in results) / len(results),
        'total_runtime_hours': total_time / 3600
    }
    
    # 9. Save Results
    print("Saving Results...")
    
    with open(os.path.join(OUTPUT_DIR, 'summaries_flat_overlap.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'metrics_flat_overlap.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 10. Print Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS: FLAT 1024-TOKEN CHUNKING WITH 128 OVERLAP")
    print("=" * 70)
    print(f"ROUGE-1: {metrics['rouge1']:.2f}")
    print(f"ROUGE-2: {metrics['rouge2']:.2f}")
    print(f"ROUGE-L: {metrics['rougeL']:.2f}")
    print(f"BERTScore F1: {metrics['bertscore_f1']:.2f}")
    print(f"Avg Chunks/Doc: {metrics['avg_chunks_per_doc']:.1f}")
    print(f"Total Runtime: {metrics['total_runtime_hours']:.2f} hours")
    print("=" * 70)
    
    return metrics

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_ablation_flat_overlap()
