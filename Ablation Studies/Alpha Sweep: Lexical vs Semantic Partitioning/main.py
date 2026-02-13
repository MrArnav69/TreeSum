import os
import sys
import json
import time
import torch
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '../../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from semantic_document_chunker import SemanticDocumentChunker
from hierarchical_summarizer import HierarchicalSummarizer

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

num_samples = 100
seed = 42
alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

output_dir = os.path.join(current_dir, 'alpha_sweep_results')
os.makedirs(output_dir, exist_ok=True)

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)

set_seed(seed)

def prepare_data():
    from datasets import load_dataset
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    indices = np.random.choice(len(dataset), num_samples, replace=False).tolist()
    selected_samples = []
    for idx in indices:
        item = dataset[int(idx)]
        selected_samples.append({
            'id': int(idx),
            'document': item['document'],
            'summary': item['summary']
        })
    return selected_samples

def run_alpha_sweep():
    samples = prepare_data()
    for alpha in alpha_values:
        summarizer = HierarchicalSummarizer(
            device=device,
            semantic_weight=alpha,
            batch_size=4
        )
        summary_export = []
        for item in tqdm(samples, desc=f"alpha={alpha:.1f}"):
            try:
                res = summarizer.summarize_document(item['document'], semantic_weight=alpha)
                pred = res['final_summary']
            except Exception as e:
                logger.error(f"error at alpha={alpha}, sample id={item['id']}: {e}")
                pred = ""
            summary_export.append({
                "sample_id": item['id'],
                "alpha": alpha,
                "document": item['document'],
                "generated_summary": pred,
                "reference_summary": item['summary']
            })
        summary_path = os.path.join(output_dir, f"summaries_alpha_{alpha:.1f}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_export, f, indent=2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    start_time = time.time()
    run_alpha_sweep()
    elapsed_time = time.time() - start_time
    logger.info(f"Alpha sweep completed in {elapsed_time:.2f} seconds")
