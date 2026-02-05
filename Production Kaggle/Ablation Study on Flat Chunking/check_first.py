
from datasets import load_dataset
dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
sample = dataset[0]
print(f"Sample 0 Document: {sample['document'][:500]}")
print(f"Sample 0 Summary: {sample['summary']}")
