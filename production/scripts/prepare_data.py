
import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import sys

# Config
NUM_SAMPLES = 500
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../data/golden_500_longest.json')
MODEL_NAME = "google/pegasus-multi_news"

def prepare_production_data():
    """
    Selects the 500 longest documents from the Multi-News test split.
    This ensures we stress-test the hierarchical tree-reduction mechanism.
    """
    print(f"Loading Multi-News dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    print(f"Initializing tokenizer for length calculation: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Calculating token lengths for {len(dataset)} documents...")
    lengths = []
    
    # We use a subset or fast length estimation if dataset is massive, 
    # but Multi-News test is ~5.6k, so we can iterate directly.
    for i in range(len(dataset)):
        doc = dataset[i]['document']
        # Fast estimation using split if tokenizer is slow, but AutoTokenizer(Pegasus) is usually fine
        # We'll use actual token count for scientific accuracy
        token_count = len(tokenizer.encode(doc, truncation=False))
        lengths.append((i, token_count))
        
        if (i+1) % 500 == 0:
            print(f"Processed {i+1}/{len(dataset)} documents...")

    # Sort by length descending
    lengths.sort(key=lambda x: x[1], reverse=True)
    
    # Select top 500
    top_indices = [x[0] for x in lengths[:NUM_SAMPLES]]
    
    print(f"Preparing final JSON (Longest doc: {lengths[0][1]} tokens, 500th: {lengths[NUM_SAMPLES-1][1]} tokens)")
    
    selected_samples = []
    for idx in top_indices:
        item = dataset[int(idx)]
        selected_samples.append({
            'id': int(idx),
            'document': item['document'],
            'summary': item['summary'],
            'token_count': next(x[1] for x in lengths if x[0] == idx)
        })
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(selected_samples, f, indent=2)
    
    print(f"\n✅ Successfully saved 500 longest samples to {OUTPUT_PATH}")
    print(f"Average length: {np.mean([x['token_count'] for x in selected_samples]):.2f} tokens.")

if __name__ == "__main__":
    prepare_production_data()
