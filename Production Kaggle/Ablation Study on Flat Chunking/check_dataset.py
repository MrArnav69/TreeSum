
from datasets import load_dataset
import json

print("Loading dataset...")
dataset = load_dataset("Awesome075/multi_news_parquet", split="test")

print(f"Dataset size: {len(dataset)}")

indices = [0, 1, 2, 10, 100]
samples = dataset.select(indices)

for i, sample in enumerate(samples):
    print(f"\n--- Sample {indices[i]} ---")
    print(f"Document header: {sample['document'][:200]}...")
    print(f"Summary: {sample['summary'][:200]}...")
