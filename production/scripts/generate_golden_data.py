
import json
import os
import numpy as np
import random
from datasets import load_dataset

def generate():
    print("🚀 Loading Multi-News dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Path to the indices used by the Flat baselines
    indices_file = "/Users/mrarnav69/Documents/TreeSum/Production Kaggle/Ablation Study on Flat Chunking/shared_sample_indices.json"
    
    if not os.path.exists(indices_file):
        print(f"❌ Error: Could not find {indices_file}")
        return

    with open(indices_file, 'r') as f:
        indices = json.load(f)
    
    print(f"📊 Mapping {len(indices)} samples from Flat study...")
    
    samples = []
    for i in indices:
        item = dataset[int(i)]
        samples.append({
            'id': int(i),
            'document': item['document'],
            'summary': item['summary']
        })
    
    out_path = "/Users/mrarnav69/Documents/TreeSum/production/data/golden_1000_shared.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✅ Created Golden Dataset: {out_path}")

if __name__ == "__main__":
    generate()
