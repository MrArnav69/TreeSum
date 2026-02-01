
import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import sys

# Config
NUM_SAMPLES = 100
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../data/golden_100_random.json')
MODEL_NAME = "google/pegasus-multi_news"
SEED = 42

def prepare_production_data():
    """
    Selects 100 random documents from the Multi-News test split.
    This provides a standard performance benchmark.
    """
    print(f"Loading Multi-News dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    print(f"Initializing tokenizer for length calculation: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Deterministic random selection
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
    
    print(f"Processing {NUM_SAMPLES} selected documents...")
    selected_samples = []
    
    for idx in indices:
        item = dataset[int(idx)]
        token_count = len(tokenizer.encode(item['document'], truncation=False))
        
        selected_samples.append({
            'id': int(idx),
            'document': item['document'],
            'summary': item['summary'],
            'token_count': token_count
        })
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(selected_samples, f, indent=2)
    
    avg_len = np.mean([x['token_count'] for x in selected_samples])
    print(f"\n✅ Successfully saved {NUM_SAMPLES} random samples to {OUTPUT_PATH}")
    print(f"Average length: {avg_len:.2f} tokens.")
    print(f"Average length: {np.mean([x['token_count'] for x in selected_samples]):.2f} tokens.")

if __name__ == "__main__":
    prepare_production_data()
