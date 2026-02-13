import os
import sys
import json
import time
import torch
import random
import logging
import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import evaluate

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '../../../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from hierarchical_summarizer import HierarchicalSummarizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random_seed = 42
num_samples = 1000
batch_size = 2
max_tokens = 1024
overlap_tokens = 128
checkpoint_interval = 100

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

output_dir = os.path.join(current_dir, 'results_flat_overlap')
os.makedirs(output_dir, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(random_seed)

class flatchunkeroverlap:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
    
    def _clean_document(self, text: str) -> str:
        text = re.sub(r'Enlarge this image.*?AP', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'toggle caption.*?AP', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()
    
    def chunk_document(self, document: str):
        document = self._clean_document(document)
        if not document: return []
        tokens = self.tokenizer.encode(document, add_special_tokens=False)
        chunks = []
        step = self.max_tokens - self.overlap_tokens
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append({'text': chunk_text, 'token_count': len(chunk_tokens)})
            if i + self.max_tokens >= len(tokens): break
        return chunks

def run_ablation_flat_overlap():
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    random.seed(random_seed)
    indices = random.sample(range(len(dataset)), num_samples)
    
    with open(os.path.join(output_dir, 'sample_indices.json'), 'w') as f:
        json.dump(indices, f)


    samples = dataset.select(indices)
    summarizer = HierarchicalSummarizer(device=device, batch_size=batch_size)
    summarizer.chunker = flatchunkeroverlap(summarizer.tokenizer)
    
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    results = []
    chunk_counts = []
    all_predictions = []
    all_references = []
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(samples, desc="Processing Documents")):
        try:
            output = summarizer.summarize_document(sample['document'])
            pred = output['final_summary']
            num_chunks = len(output.get('chunks', []))
        except Exception as e:
            logger.error(f"sample {indices[i]} failed: {e}")
            pred = ""
            num_chunks = 0
            
        results.append({
            'sample_id': indices[i],
            'generated_summary': pred,
            'reference_summary': sample['summary'],
            'doc_length': len(sample['document'].split())
        })

        chunk_counts.append(num_chunks)
        all_predictions.append(pred)
        all_references.append(sample['summary'])

        if (i + 1) % checkpoint_interval == 0:
            batch_num = (i + 1) // checkpoint_interval
            with open(os.path.join(output_dir, f'summaries_batch_{batch_num}.json'), 'w') as f:
                json.dump(results[i+1-checkpoint_interval:i+1], f, indent=2)
            
            with open(os.path.join(output_dir, 'progress.json'), 'w') as f:
                json.dump({"completed": i + 1, "total": len(samples)}, f)

    total_time = time.time() - start_time
    
    r_res = rouge.compute(predictions=all_predictions, references=all_references)
    b_res = bertscore.compute(
        predictions=all_predictions, references=all_references, lang="en",
        model_type="microsoft/deberta-xlarge-mnli", device=device, batch_size=8
    )
    
    r1, r2, rL, rLsum = r_res['rouge1']*100, r_res['rouge2']*100, r_res['rougeL']*100, r_res['rougeLsum']*100
    bp, br, bf1 = np.mean(b_res['precision'])*100, np.mean(b_res['recall'])*100, np.mean(b_res['f1'])*100
    
    metrics = {
        "method": "Flat_1024_128_Overlap",
        "num_samples": len(samples),
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rL,
        "rougeLsum": rLsum,
        "bertscore_precision": bp,
        "bertscore_recall": br,
        "bertscore_f1": bf1,
        "avg_chunks_per_doc": np.mean(chunk_counts),
        "total_runtime_hours": total_time / 3600
    }
    
    with open(os.path.join(output_dir, 'metrics_flat_overlap.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    with open(os.path.join(output_dir, 'summaries_flat_overlap.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_ablation_flat_overlap()
