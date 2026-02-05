
from datasets import load_dataset
dataset = load_dataset("Awesome075/multi_news_parquet", split="test")

target = "recognize the accomplishments of women in the military"
found = False
for i, sample in enumerate(dataset):
    if target in sample['summary'] or target in sample['document']:
        print(f"Found in sample {i}!")
        print(f"Summary: {sample['summary'][:200]}...")
        found = True
        break

if not found:
    print("Not found in test set.")
