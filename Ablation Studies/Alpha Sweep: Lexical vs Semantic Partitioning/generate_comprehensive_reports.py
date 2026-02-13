import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import warnings
import time
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
import torch

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'alpha_sweep_results')
output_dir = os.path.join(current_dir, 'alpha_sweep_reports')

bertscore_model = "microsoft/deberta-xlarge-mnli"
bertscore_batch_size = 8
use_mps = torch.backends.mps.is_available()

alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

os.makedirs(output_dir, exist_ok=True)

def compute_rouge_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    rouge = evaluate.load('rouge')
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not valid_pairs:
        return {k: 0.0 for k in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
    
    vp, vr = zip(*valid_pairs)
    scores = rouge.compute(predictions=vp, references=vr, use_stemmer=True)
    return {k: v * 100 for k, v in scores.items()}

def compute_bertscore_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    from bert_score import score as bert_score_fn
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not valid_pairs:
        return {k: 0.0 for k in ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']}
    
    vp, vr = zip(*valid_pairs)
    device = 'mps' if use_mps else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    P, R, F1 = bert_score_fn(
        list(vp), list(vr),
        model_type=bertscore_model,
        device=device,
        batch_size=bertscore_batch_size,
        lang='en',
        verbose=False
    )
    
    return {
        'bertscore_precision': P.mean().item() * 100,
        'bertscore_recall': R.mean().item() * 100,
        'bertscore_f1': F1.mean().item() * 100
    }

def run_analysis():
    all_results = []
    
    for alpha in alpha_values:
        file_path = os.path.join(results_dir, f"summaries_alpha_{alpha:.1f}.json")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        preds = [s['generated_summary'] for s in data]
        refs = [s['reference_summary'] for s in data]
        ids = [s['sample_id'] for s in data]
        
        r_scores = compute_rouge_metrics(preds, refs)
        b_scores = compute_bertscore_metrics(preds, refs)
        
        res = {'alpha': alpha, 'num_samples': len(data), 'sample_ids': ids}
        res.update(r_scores)
        res.update(b_scores)
        all_results.append(res)
        
    if not all_results:
        return

    df = pd.DataFrame(all_results).set_index('alpha')
    
    df.drop(columns=['sample_ids']).to_csv(os.path.join(output_dir, 'alpha_sweep_results.csv'))
    df[['rouge1', 'rouge2', 'rougeL', 'rougeLsum']].to_csv(os.path.join(output_dir, 'rouge_results.csv'))
    df[['bertscore_precision', 'bertscore_recall', 'bertscore_f1']].to_csv(os.path.join(output_dir, 'bertscore_results.csv'))
    
    df.drop(columns=['sample_ids']).to_json(os.path.join(output_dir, 'alpha_sweep_results.json'), indent=2, orient='index')
    
    ref_ids = sorted(all_results[0]['sample_ids'])
    consistent = all(sorted(r['sample_ids']) == ref_ids for r in all_results)
    with open(os.path.join(output_dir, 'validation_report.json'), 'w') as f:
        json.dump({
            'status': 'PASSED' if consistent else 'FAILED',
            'num_samples': len(ref_ids),
            'sample_ids': ref_ids
        }, f, indent=2)

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for col in ['rouge1', 'rouge2', 'rougeL']:
        ax1.plot(df.index, df[col], marker='o', label=col.upper())
    ax1.set_title('Alpha vs ROUGE')
    ax1.legend()
    
    for col in ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
        ax2.plot(df.index, df[col], marker='s', label=col.replace('bertscore_', '').title())
    ax2.set_title('Alpha vs BERTScore')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_analysis.png'))
    
    for metric, title in [('rouge1', 'ROUGE-1'), ('bertscore_f1', 'BERTScore F1')]:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[metric], marker='o', linewidth=3)
        plt.title(f'Alpha vs {title}')
        plt.savefig(os.path.join(output_dir, f'{metric}_analysis.png'))
        plt.close()

    with open(os.path.join(output_dir, 'COMPREHENSIVE_REPORT.txt'), 'w') as f:
        f.write("ALPHA SWEEP COMPREHENSIVE REPORT\n")
        f.write("="*32 + "\n\n")
        f.write(df.drop(columns=['sample_ids']).to_string())
        f.write("\n\nBest Alphas:\n")
        for col in df.columns:
            if col != 'sample_ids':
                f.write(f"{col}: {df[col].idxmax():.1f} ({df[col].max():.2f})\n")

if __name__ == "__main__":
    run_analysis()
