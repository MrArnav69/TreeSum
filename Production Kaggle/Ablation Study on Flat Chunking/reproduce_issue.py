
import torch
from transformers import PegasusForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
import logging
import json

# Minimal version of the summarizer
class FlatChunkerOverlap:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = 1024
        self.overlap = 128
        self.step = 896
    
    def chunk_document(self, document: str):
        tokens = self.tokenizer.encode(document, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), self.step):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append({'text': chunk_text})
            if i + self.max_tokens >= len(tokens):
                break
        return chunks

class HierarchicalSummarizerOverlap:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.chunker = FlatChunkerOverlap(tokenizer)
        self.device = "cpu"
    
    def _generate(self, inputs, max_length=256, min_length=64):
        if not inputs or all(not s.strip() for s in inputs):
            return [""] * len(inputs)
            
        batch = self.tokenizer(inputs, truncation=True, padding="longest", max_length=1024, return_tensors="pt")
        with torch.no_grad():
            summary_ids = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=4, # Faster for test
                max_length=max_length,
                min_length=min_length
            )
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    def summarize_document(self, document):
        chunks = self.chunker.chunk_document(document)
        print(f"Num chunks: {len(chunks)}")
        chunk_texts = [c['text'] for c in chunks]
        
        # Mapping
        chunk_summaries = []
        for text in chunk_texts:
            s = self._generate([text], max_length=128)[0]
            chunk_summaries.append(s)
            print(f"Chunk summary: {s[:100]}...")
            
        # Reducing
        combined = " ".join(chunk_summaries)
        final = self._generate([combined], max_length=512, min_length=128)[0]
        return final

print("Loading model...")
model_name = "google/pegasus-multi_news"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

summarizer = HierarchicalSummarizerOverlap(model, tokenizer)

print("Loading dataset sample...")
ds = load_dataset("Awesome075/multi_news_parquet", split="test")
doc = ds[0]['document']

print("\nSummarizing sample 0...")
summary = summarizer.summarize_document(doc)
print(f"\nFINAL SUMMARY:\n{summary}")
