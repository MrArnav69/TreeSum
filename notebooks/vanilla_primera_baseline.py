#!/usr/bin/env python3

import subprocess
import sys

def install_requirements():
    """install required packages for kaggle environment"""
    packages = [
        'transformers>=4.30.0',
        'datasets>=2.14.0',
        'torch>=2.0.0',
        'tqdm>=4.65.0'
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

import json
import random
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

checkpoint_dir = Path("vanilla_primera_checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

final_output_file = "vanilla_primera_summaries_2500.json"
checkpoint_interval = 100
num_samples = 2500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")

print("loading dataset...")
dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
print(f"dataset loaded: {len(dataset)} total samples")

print("sampling indices with seed 42...")
all_indices = list(range(len(dataset)))
random.shuffle(all_indices)
selected_indices = sorted(all_indices[:num_samples])
print(f"selected {len(selected_indices)} indices")

print("loading primera model...")
model_name = "allenai/PRIMERA-multinews"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
print("model loaded successfully")

results = []
start_time = datetime.now()

print(f"\nstarting summarization of {num_samples} samples...")
for i, idx in enumerate(tqdm(selected_indices, desc="generating summaries")):
    sample = dataset[idx]
    document = sample['document']
    reference_summary = sample['summary']
    
    inputs = tokenizer(
        document,
        max_length=4096,
        truncation=True,
        padding=False,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=512,
                min_length=128,
                length_penalty=0.8,
                num_beams=8,
                early_stopping=True
            )
    
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    result_entry = {
        'sample_id': idx,
        'document': document,
        'reference_summary': reference_summary,
        'generated_summary': generated_summary,
        'doc_length': len(tokenizer.encode(document, truncation=False))
    }
    
    results.append(result_entry)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if (i + 1) % checkpoint_interval == 0:
        checkpoint_file = checkpoint_dir / f"checkpoint_{i+1:04d}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(results[-checkpoint_interval:], f, indent=2)
        print(f"\ncheckpoint saved: {checkpoint_file}")

if len(results) % checkpoint_interval != 0:
    remaining = len(results) % checkpoint_interval
    checkpoint_file = checkpoint_dir / f"checkpoint_final_{len(results):04d}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(results[-remaining:], f, indent=2)
    print(f"\nfinal partial checkpoint saved: {checkpoint_file}")

print(f"\nsaving final consolidated results to {final_output_file}...")
with open(final_output_file, 'w') as f:
    json.dump(results, f, indent=2)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print(f"\n{'='*60}")
print(f"summarization complete!")
print(f"total samples processed: {len(results)}")
print(f"total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
print(f"average time per sample: {duration/len(results):.2f} seconds")
print(f"final output: {final_output_file}")
print(f"checkpoints: {checkpoint_dir}/")
print(f"{'='*60}")
