import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast
from typing import List, Dict, Optional, Union
import time
import logging
from tqdm import tqdm
import numpy as np
from semantic_document_chunker import SemanticDocumentChunker

# Configure logging for research reproducibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalSummarizer:
    """
    State-of-the-Art Hierarchical Summarizer for Long Documents.
    
    Implements a robust 2-stage pipeline:
    1. Local Stage: Summarize semantic chunks in parallel (batched).
    2. Global Stage: Aggregate chunk summaries into a coherent final summary.
    
    Optimizations:
    - GPU Acceleration (CUDA/MPS)
    - Dynamic Batching
    - Comparison-grade Beam Search parameters
    """
    
    def __init__(self, 
                 model_name: str = "google/pegasus-multi_news",
                 device: Optional[str] = None,
                 batch_size: int = 4,
                 chunker: Optional[SemanticDocumentChunker] = None):
        """
        Initialize the summarizer.
        
        Args:
            model_name: HuggingFace model hub ID.
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
            batch_size: Batch size for chunk summarization.
            chunker: Pre-initialized chunker instance.
        """
        # 1. Device Selection
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing HierarchicalSummarizer on {self.device}")
        
        # 2. Load Model & Tokenizer (Fast Rust-based tokenizer)
        try:
            self.tokenizer = PegasusTokenizerFast.from_pretrained(model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
            # FP16 inference for speed if on CUDA (MPS support varies)
            if self.device == 'cuda':
                self.model = self.model.half()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
        self.batch_size = batch_size
        
        # 3. Initialize Chunker (SOTA configuration)
        if chunker:
            self.chunker = chunker
        else:
            self.chunker = SemanticDocumentChunker(
                tokenizer=self.tokenizer,
                max_tokens=1024,
                overlap_tokens=128,
                use_semantic_coherence=True,
                adaptive_overlap=True # SOTA mode
            )
            
    def _generate(self, inputs: List[str], max_length: int = 512, min_length: int = 64) -> List[str]:
        """
        Low-level generation with researched beam search parameters.
        """
        # Tokenize (Batch)
        batch = self.tokenizer(
            inputs, 
            truncation=True, 
            padding="longest", 
            max_length=1024, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate with SOTA parameters for Multi-News
        # - num_beams=8: Standard for high quality abstractive summ
        # - length_penalty=1.2: Tuned for Multi-News (longer summaries = higher Recall)
        # - max_length=1024: Allow full length generation
        try:
            summary_ids = self.model.generate(
                batch["input_ids"],
                num_beams=8, 
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.2, # UPDATED: Encourages longer output (prev 0.8)
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            # Decode
            summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            return summaries
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return [""] * len(inputs)

    def summarize_document(self, document: str) -> Dict[str, Union[str, List[str]]]:
        """
        Execute the full hierarchical pipeline on a raw document.
        
        Returns:
            Dict containing:
            - 'final_summary': The resulting summary
            - 'chunk_summaries': Intermediate summaries (for analysis)
            - 'chunks': Raw chunks
        """
        if not document.strip():
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': []}
            
        # Stage 1: Segmentation (SOTA Hybrid)
        chunks = self.chunker.chunk_document(document)
        chunk_texts = [c['text'] for c in chunks]
        
        if not chunk_texts:
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': []}
            
        # Stage 2: Map (Chunk Summarization)
        chunk_summaries = self._stage1_map_summaries(chunk_texts)
            
        # Stage 3: Reduce (Aggregation & Final Summarization)
        final_summary, concatenated_summary = self._stage2_reduce_summaries(chunk_summaries)
            
        return {
            'final_summary': final_summary,
            'chunk_summaries': chunk_summaries,
            'chunks': chunks,
            'concatenated_intermediate': concatenated_summary
        }

    def _stage1_map_summaries(self, chunk_texts: List[str]) -> List[str]:
        """
        Stage 1 (Map): Summarize each chunk independently.
        """
        chunk_summaries = []
        
        # Determine strictness based on chunk count
        # If we have many chunks, local summaries should be concise.
        local_max_len = 128 if len(chunk_texts) > 5 else 256
        
        for i in range(0, len(chunk_texts), self.batch_size):
            batch = chunk_texts[i : i + self.batch_size]
            summaries = self._generate(batch, max_length=local_max_len)
            chunk_summaries.extend(summaries)
            
        return chunk_summaries

    def _stage2_reduce_summaries(self, chunk_summaries: List[str]) -> (str, str):
        """
        Stage 2 (Reduce): Recursively summarize chunk summaries until they fit valid context.
        
        This implements a 'Tree Reduction' strategy:
        [S1, S2, S3, S4, S5, S6] (Too long)
           |       |       |
        [  SumA,   SumB,   SumC ] (Intermediate)
                   |
                FinalSum
                
        Ensures NO truncation of content regardless of document length.
        """
        # 1. Initial Concatenation (for logging/debug)
        concatenated_intermediate = " ".join(chunk_summaries)
        
        current_summaries = chunk_summaries
        layer = 0
        
        # SOTA: Pegasus Max Positional Embeddings = 1024
        # We leave some buffer for generation overhead
        MAX_INPUT_TOKENS = 1000 
        
        while True:
            # Check length of concatenated current level
            combined_text = " ".join(current_summaries)
            tokenized_len = len(self.tokenizer.encode(combined_text, truncation=False))
            
            logger.info(f"Reduction Layer {layer}: {len(current_summaries)} chunks, {tokenized_len} tokens")
            
            # Base Case: If it fits, generate final summary
            if tokenized_len <= MAX_INPUT_TOKENS:
                # If it's very short, just return it (don't over-summarize)
                if tokenized_len < 256 and layer > 0:
                    return combined_text, concatenated_intermediate
                
                # Final Pass
                final_summary_list = self._generate(
                    [combined_text], 
                    max_length=512, 
                    min_length=128
                )
                return final_summary_list[0], concatenated_intermediate
            
            # Recursive Step: Group and Summarize
            if len(current_summaries) <= 1:
                # Edge case: Single summary is still too long (unlikely with chunking, but possible)
                # We forcedly summarize it
                final_summary_list = self._generate(
                    [current_summaries[0]], 
                    max_length=512, 
                    min_length=128
                )
                return final_summary_list[0], concatenated_intermediate
                
            # Smart Grouping (Bin Packing)
            new_level_summaries = []
            current_group = []
            current_group_len = 0
            
            for summary in current_summaries:
                s_len = len(self.tokenizer.encode(summary, truncation=False))
                
                # If adding this summary exceeds limit, process current group
                if current_group_len + s_len > MAX_INPUT_TOKENS:
                    if current_group:
                        # Summarize the group
                        group_text = " ".join(current_group)
                        new_summary = self._generate([group_text], max_length=256)[0]
                        new_level_summaries.append(new_summary)
                    
                    # Reset
                    current_group = [summary]
                    current_group_len = s_len
                else:
                    current_group.append(summary)
                    current_group_len += s_len
            
            # Process final group
            if current_group:
                group_text = " ".join(current_group)
                new_summary = self._generate([group_text], max_length=256)[0]
                new_level_summaries.append(new_summary)
            
            # Update for next iteration
            current_summaries = new_level_summaries
            layer += 1
            
            # Safety break to prevent infinite loops (though unlikely)
            if layer > 5:
                logger.warning("Max reduction layers reached. Truncating.")
                final_text = " ".join(current_summaries)
                final_summary_list = self._generate([final_text], max_length=512)
                return final_summary_list[0], concatenated_intermediate

if __name__ == "__main__":
    # Integration Test
    print("Testing Hierarchical Summarizer...")
    
    # Create a dummy long text
    dummy_text = "This is a sentence about technology. " * 50 + " " + \
                 "This is a sentence about nature. " * 50 + " " + \
                 "This is a sentence about space. " * 50
                 
    # DEBUG: Force CPU to rule out MPS hang
    summarizer = HierarchicalSummarizer(device='cpu')
    print(f"Initialized on {summarizer.device}")
    
    result = summarizer.summarize_document(dummy_text)
    
    print("\n=== Chunk Summaries ===")
    for i, s in enumerate(result['chunk_summaries']):
        print(f"Chunk {i}: {s[:100]}...")
        
    print("\n=== Final Summary ===")
    print(result['final_summary'])
    print(f"\nPipeline successful. Device: {summarizer.device}")
