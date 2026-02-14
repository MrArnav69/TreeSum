import torch
from transformers import PegasusForConditionalGeneration, AutoTokenizer
from typing import List, Dict, Optional, Union, Any
import time
import logging
from tqdm import tqdm
import numpy as np
from semantic_document_chunker import SemanticDocumentChunker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalSummarizer:
    def __init__(self, 
                 model_name: str = "google/pegasus-multi_news",
                 device: Optional[str] = None,
                 batch_size: int = 4,
                 semantic_weight: float = 1.0,
                 dtype: Optional[torch.dtype] = None,
                 chunker: Optional[SemanticDocumentChunker] = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        logger.info(f"initializing hierarchicalsummarizer on {self.device}")
        
        self.dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=self.dtype
        ).to(self.device)
                
        self.batch_size = batch_size
        
        if chunker:
            self.chunker = chunker
        else:
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

    def summarize_document(self, document: str, semantic_weight: Optional[float] = None) -> Dict[str, Any]:
        if not document.strip():
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': []}
            
        chunks = self.chunker.chunk_document(document, semantic_weight=semantic_weight)
        chunk_texts = [c['text'] for c in chunks]
        
        if not chunk_texts:
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': []}
            
        chunk_summaries = self._stage1_map_summaries(chunk_texts)
        final_summary, concatenated_summary = self._stage2_reduce_summaries(chunk_summaries)
            
        return {
            'final_summary': final_summary,
            'chunk_summaries': chunk_summaries,
            'chunks': chunks,
            'concatenated_intermediate': concatenated_summary
        }

    def _stage1_map_summaries(self, chunk_texts: List[str]) -> List[str]:
        chunk_summaries = []
        local_max_len = 128 if len(chunk_texts) > 5 else 256
        for i in range(0, len(chunk_texts), self.batch_size):
            batch = chunk_texts[i : i + self.batch_size]
            summaries = self._generate(batch, max_length=local_max_len)
            chunk_summaries.extend(summaries)
        return chunk_summaries

    def _stage2_reduce_summaries(self, chunk_summaries: List[str]) -> (str, str):
        concatenated_intermediate = " ".join(chunk_summaries)
        current_summaries = chunk_summaries
        layer = 0
        max_input_tokens = 1000 
        
        while True:
            combined_text = " ".join(current_summaries)
            tokenized_len = len(self.tokenizer.encode(combined_text, truncation=False))
            logger.info(f"reduction layer {layer}: {len(current_summaries)} chunks, {tokenized_len} tokens")
            
            if tokenized_len <= max_input_tokens:
                if tokenized_len < 256 and layer > 0:
                    return combined_text, concatenated_intermediate
                
                final_summary_list = self._generate(
                    [combined_text], 
                    max_length=512, 
                    min_length=128
                )
                return final_summary_list[0], concatenated_intermediate
            
            if len(current_summaries) <= 1:
                final_summary_list = self._generate(
                    [current_summaries[0]], 
                    max_length=512, 
                    min_length=128
                )
                return final_summary_list[0], concatenated_intermediate
                
            new_level_summaries = []
            current_group = []
            current_group_len = 0
            
            for summary in current_summaries:
                s_len = len(self.tokenizer.encode(summary, truncation=False))
                if current_group_len + s_len > max_input_tokens:
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
                logger.warning("max reduction layers reached. truncating.")
                final_text = " ".join(current_summaries)
                final_summary_list = self._generate([final_text], max_length=512)
                return final_summary_list[0], concatenated_intermediate
