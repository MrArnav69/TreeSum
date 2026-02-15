import os
import sys
import subprocess

def install_requirements():
    """Install required packages for Kaggle environment"""
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
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"warning: failed to install {package}: {e}")
    print("Package installation complete\n")

# Install missing dependencies in Kaggle
install_requirements()

import json
import random
import torch
import numpy as np
import nltk
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

# Add Kaggle source path
kaggle_source = "/kaggle/input/models/mrarnav69/treesum/pytorch/default/1"
if os.path.exists(kaggle_source):
    sys.path.append(kaggle_source)
    print(f"added {kaggle_source} to python path")
else:
    # Fallback to local for development
    src_path = Path(__file__).resolve().parent.parent / "src"
    sys.path.append(str(src_path))
    print(f"added {src_path} to python path")

from semantic_document_chunker import SemanticDocumentChunker
from hierarchical_summarizer import HierarchicalSummarizer

# PART 4 CONFIGURATION
PART_ID = 4
START_SLICE = 1500
END_SLICE = 2000
final_output = f"treesum_summaries_p{PART_ID}.json"
checkpoint_dir = Path(f"treesum_checkpoints_p{PART_ID}")
checkpoint_dir.mkdir(exist_ok=True)

# Global Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

nltk.download('punkt', quiet=True)

print(f"Loading dataset for Part {PART_ID}...")
dataset = load_dataset("Awesome075/multi_news_parquet", split="test")

print("Sampling indices (Seed 42)...")
all_indices = list(range(len(dataset)))
random.shuffle(all_indices)
# Match exactly the full list, then slice
full_selected = sorted(all_indices[:2500])
selected_indices = full_selected[START_SLICE:END_SLICE]

print(f"Initializing TreeSum (P100 Optimized, Batch 8, FP32)...")
summarizer = HierarchicalSummarizer(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=8,
    dtype=torch.float32,
    semantic_weight=1.0,
    compile=False
)

results = []
start_time = datetime.now()

print(f"Starting treesum Part {PART_ID} run on {len(selected_indices)} samples...")
for i, idx in enumerate(tqdm(selected_indices)):
    sample = dataset[idx]
    doc = sample['document']
    
    try:
        output = summarizer.summarize_document(doc)
        
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
            'concatenated_intermediate': output.get('concatenated_intermediate', ""),
            'reduction_layers': output.get('reduction_layers', 0)
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
print(f"Part {PART_ID} complete!")
print(f"total time: {duration/60:.2f} minutes")
print(f"final output: {final_output}")
print(f"{'='*60}")
