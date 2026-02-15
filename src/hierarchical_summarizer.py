import torch
from transformers import PegasusForConditionalGeneration, AutoTokenizer
from typing import List, Dict, Optional, Union, Any
import logging
from semantic_document_chunker import SemanticDocumentChunker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalSummarizer:
    """
    EXACT COPY from run_treesum_part1_500.py (Working Reference)
    Ensures 100% functional parity with the ablation study.
    """
    
    def __init__(self, 
                 model_name: str = "google/pegasus-multi_news",
                 device: Optional[str] = None,
                 batch_size: int = 4,
                 semantic_weight: float = 1.0,
                 dtype: Optional[torch.dtype] = None,
                 chunker: Optional[SemanticDocumentChunker] = None,
                 compile: bool = False):
        """
        Initialize the summarizer.
        
        Args:
            model_name: HuggingFace model hub ID.
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
            batch_size: Batch size for chunk summarization.
            dtype: torch.dtype (e.g. torch.bfloat16). Defaults to precision appropriate for device.
            chunker: Pre-initialized chunker instance.
            compile: Whether to use torch.compile() for speedup (Requires Torch 2.0+).
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
        
        # 2. Dtype Selection (Flexible for H200 BFloat16)
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = torch.float32  # Default to float32 for max compatibility

        # 3. Load Model & Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.model = PegasusForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=self.dtype
            ).to(self.device)
            # CRITICAL DIFFERENCE: NO model.eval() here in working script
            
            # Optional: Optimization with torch.compile
            if compile:
                if hasattr(torch, 'compile'):
                    logger.info("Compiling model for faster inference (Torch 2.0+)...")
                    self.model = torch.compile(self.model)
                else:
                    logger.warning("torch.compile() not available. Skipping.")
            
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
                max_tokens=1024, # Matches working script (was 1024 in broken src)
                overlap_tokens=128,
                use_semantic_coherence=True,
                adaptive_overlap=True,
                semantic_weight=semantic_weight
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
            
            # Decode
            summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            return summaries
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return [""] * len(inputs)

    def summarize_document(self, document: str) -> Dict[str, Any]:
        """
        Execute the full hierarchical pipeline on a raw document.
        
        Returns:
            Dict containing:
            - 'final_summary': The resulting summary
            - 'chunk_summaries': Intermediate summaries (for analysis)
            - 'chunks': Raw chunks
        """
        if not document.strip():
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': [], 'reduction_layers': 0}
            
        # Stage 1: Segmentation (SOTA Hybrid)
        chunks = self.chunker.chunk_document(document)
        chunk_texts = [c['text'] for c in chunks]
        
        if not chunk_texts:
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': [], 'reduction_layers': 0}
            
        # Stage 2: Map (Chunk Summarization)
        chunk_summaries = self._stage1_map_summaries(chunk_texts)
            
        # Stage 3: Reduce (Aggregation & Final Summarization)
        final_summary, concatenated_summary, reduction_layers = self._stage2_reduce_summaries(chunk_summaries)
            
        return {
            'final_summary': final_summary,
            'chunk_summaries': chunk_summaries,
            'chunks': chunks,
            'concatenated_intermediate': concatenated_summary,
            'reduction_layers': reduction_layers
        }

    def _stage1_map_summaries(self, chunk_texts: List[str]) -> List[str]:
        """
        Stage 1 (Map): Summarize each chunk independently.
        """
        chunk_summaries = []
        
        # Determine strictness based on chunk count
        local_max_len = 128 if len(chunk_texts) > 5 else 256
        
        for i in range(0, len(chunk_texts), self.batch_size):
            batch = chunk_texts[i : i + self.batch_size]
            summaries = self._generate(batch, max_length=local_max_len)
            chunk_summaries.extend(summaries)
            
        return chunk_summaries

    def _stage2_reduce_summaries(self, chunk_summaries: List[str]) -> tuple:
        """
        Stage 2 (Reduce): Recursively summarize chunk summaries until they fit valid context.
        """
        # 1. Initial Concatenation (for logging/debug)
        concatenated_intermediate = " ".join(chunk_summaries)
        
        current_summaries = chunk_summaries
        layer = 0
        
        MAX_INPUT_TOKENS = 1000 
        
        while True:
            # Check length of concatenated current level
            combined_text = " ".join(current_summaries)
            tokenized_len = len(self.tokenizer.encode(combined_text, truncation=False))
            
            logger.info(f"Reduction Layer {layer}: {len(current_summaries)} chunks, {tokenized_len} tokens")
            
            # Base Case: If it fits, generate final summary
            if tokenized_len <= MAX_INPUT_TOKENS:
                # If it's very short, just return it
                if tokenized_len < 256 and layer > 0:
                    return combined_text, concatenated_intermediate, layer
                
                # Final Pass
                final_summary_list = self._generate(
                    [combined_text], 
                    max_length=512, 
                    min_length=128
                )
                return final_summary_list[0], concatenated_intermediate, layer
            
            # Recursive Step
            if len(current_summaries) <= 1:
                final_summary_list = self._generate(
                    [current_summaries[0]], 
                    max_length=512, 
                    min_length=128
                )
                return final_summary_list[0], concatenated_intermediate, layer
                
            # Smart Grouping (Bin Packing)
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
            
            # Process final group
            if current_group:
                group_text = " ".join(current_group)
                new_summary = self._generate([group_text], max_length=256)[0]
                new_level_summaries.append(new_summary)
            
            current_summaries = new_level_summaries
            layer += 1
            
            # Safety break
            if layer > 5:
                logger.warning("Max reduction layers reached. Truncating.")
                final_text = " ".join(current_summaries)
                final_summary_list = self._generate([final_text], max_length=512)
                return final_summary_list[0], concatenated_intermediate, layer
