#!/usr/bin/env python3

import subprocess
import sys

def install_requirements():
    packages = [
        'transformers>=4.30.0',
        'datasets>=2.14.0',
        'torch>=2.0.0',
        'tqdm>=4.65.0',
        'sentence-transformers>=2.2.2',
        'nltk>=3.8.0',
        'scipy>=1.10.0',
        'numpy>=1.24.0',
        'psutil'
    ]
    
    print("installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"warning: failed to install {package}: {e}")
    print("package installation complete\n")

try:
    install_requirements()
except Exception as e:
    print(f"installation error (continuing anyway): {e}\n")

import os
import sys
import json
import random
import torch
import numpy as np
import nltk
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))
print(f"added {src_path} to python path")

try:
    from semantic_document_chunker import SemanticDocumentChunker
    from hierarchical_summarizer import HierarchicalSummarizer
    print("successfully imported TreeSum modules from src/")
except ImportError as e:
    print(f"critical error: could not import TreeSum modules: {e}")
    sys.exit(1)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

nltk.download('punkt', quiet=True)

checkpoint_dir = Path("treesum_checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
final_output = "treesum_summaries_2500.json"

print("loading dataset...")
dataset = load_dataset("Awesome075/multi_news_parquet", split="test")

print("sampling indices (Seed 42)...")
all_indices = list(range(len(dataset)))
random.shuffle(all_indices)
selected_indices = sorted(all_indices[:2500])

print("initializing treesum (pure semantic, alpha=1.0)...")

chunker = SemanticDocumentChunker(
    semantic_weight=1.0,
    min_chunk_tokens=256,
    max_tokens=1024,
    overlap_tokens=128
)

summarizer = HierarchicalSummarizer(
    model_name="google/pegasus-multi_news",
    chunker=chunker,
    batch_size=32,
    dtype=torch.float32
)

results = []
start_time = datetime.now()

print(f"starting treesum run on {len(selected_indices)} samples...")
for i, idx in enumerate(tqdm(selected_indices)):
    sample = dataset[idx]
    doc = sample['document']
    
    try:
        output = summarizer.summarize_document(doc, semantic_weight=1.0)
        
        chunk_metadata = []
        if 'chunks' in output and output['chunks']:
            chunk_metadata = [
                {
                    'chunk_id': c.get('chunk_id'),
                    'token_count': c.get('token_count'),
                    'has_overlap': c.get('has_overlap', False)
                } for c in output['chunks']
            ]
            
        res = {
            'sample_id': idx,
            'document': doc,
            'reference_summary': sample['summary'],
            'generated_summary': output.get('final_summary', ""),
            'num_chunks': len(chunk_metadata),
            'chunk_metadata': chunk_metadata,
            'concatenated_intermediate': output.get('concatenated_intermediate', "")
        }
        results.append(res)
        
    except Exception as e:
        print(f"error on sample {idx}: {e}")
        results.append({
            'sample_id': idx,
            'error': str(e),
            'generated_summary': ""
        })
        continue
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if (i + 1) % 100 == 0:
        ckpt = checkpoint_dir / f"checkpoint_{i+1:04d}.json"
        with open(ckpt, 'w') as f:
            json.dump(results[-100:], f, indent=2)
        print(f"checkpoint saved: {ckpt}")

with open(final_output, 'w') as f:
    json.dump(results, f, indent=2)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print(f"\n{'='*60}")
print(f"treesum synthesis complete!")
print(f"total samples processed: {len(results)}")
print(f"total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
print(f"average time per sample: {duration/len(results):.2f} seconds")
print(f"final output: {final_output}")
print(f"{'='*60}")
