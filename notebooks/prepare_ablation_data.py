
import os
import json
import numpy as np
from datasets import load_dataset

# Config
NUM_SAMPLES = 20
SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../results/ablation/ablation_data.json')
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def prepare_data():
    print(f"Loading Multi-News dataset (Seed: {SEED})...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Deterministic selection
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
    
    selected_samples = []
    for idx in indices:
        item = dataset[int(idx)]
        selected_samples.append({
            'id': int(idx),
            'document': item['document'],
            'summary': item['summary']
        })
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(selected_samples, f, indent=2)
    
    print(f"Successfully saved {len(selected_samples)} samples to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_data()
