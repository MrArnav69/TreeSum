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
import nltk
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

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

output_dir = os.path.join(current_dir, 'treesum_part1_results')
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

def run_ablation_treesum_1():
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    random.seed(random_seed)
    indices = random.sample(range(len(dataset)), num_samples)
    selected_indices = indices[:500]


    samples = dataset.select(selected_indices)
    summarizer = HierarchicalSummarizer(device=device, batch_size=batch_size, semantic_weight=1.0)
    
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    results = []
    all_predictions = []
    all_references = []
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(samples, desc="Processing Documents")):
        try:
            output = summarizer.summarize_document(sample['document'], semantic_weight=1.0)
            pred = output['final_summary']
        except Exception as e:
            logger.error(f"sample {i} failed: {e}")
            pred = ""
            
        results.append({
            'sample_id': selected_indices[i],
            'generated_summary': pred,
            'reference_summary': sample['summary'],
            'doc_length': len(sample['document'].split())
        })

        all_predictions.append(pred)
        all_references.append(sample['summary'])

    total_time = time.time() - start_time
    
    rouge_res = rouge.compute(predictions=all_predictions, references=all_references)
    bert_res = bertscore.compute(
        predictions=all_predictions, 
        references=all_references, 
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        device=device,
        batch_size=8
    )
    
    r1, r2, rL, rLsum = rouge_res['rouge1']*100, rouge_res['rouge2']*100, rouge_res['rougeL']*100, rouge_res['rougeLsum']*100
    bp, br, bf1 = np.mean(bert_res['precision'])*100, np.mean(bert_res['recall'])*100, np.mean(bert_res['f1'])*100
    
    summary_df = pd.DataFrame([{
        '': 'alpha_1.0',
        'rouge1': r1, 'rouge2': r2, 'rougeL': rL, 'rougeLsum': rLsum,
        'bertscore_precision': bp, 'bertscore_recall': br, 'bertscore_f1': bf1
    }])
    summary_df.to_csv(os.path.join(output_dir, 'treesum_part1_results.csv'), index=False)
    
    pd.DataFrame([{
        '': 'alpha_1.0',
        'rouge1': r1, 'rouge2': r2, 'rougeL': rL, 'rougeLsum': rLsum
    }]).to_csv(os.path.join(output_dir, 'rouge_results.csv'), index=False)
    
    pd.DataFrame([{
        '': 'alpha_1.0',
        'bertscore_precision': bp, 'bertscore_recall': br, 'bertscore_f1': bf1
    }]).to_csv(os.path.join(output_dir, 'bertscore_results.csv'), index=False)
    
    metadata = {
        "num_samples": len(samples),
        "alpha_values": [1.0],
        "device": device,
        "total_time_seconds": total_time,
        "time_per_sample_avg": total_time / len(samples) if len(samples) > 0 else 0
    }
    with open(os.path.join(output_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    with open(os.path.join(output_dir, 'summaries_treesum_pt1_first_500.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_ablation_treesum_1()
