import sys
import os
import random
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset

def get_indices(start, end):
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    full_selected = sorted(all_indices[:2500])
    return full_selected[start:end]

p1 = get_indices(0, 500)
p2 = get_indices(500, 1000)

print(f"Part 1 First 5: {p1[:5]}")
print(f"Part 1 Last 5: {p1[-5:]}")
print(f"Part 2 First 5: {p2[:5]}")
print(f"Part 2 Last 5: {p2[-5:]}")

# Check for overlap
overlap = set(p1).intersection(set(p2))
print(f"Overlap between P1 and P2: {len(overlap)}")

if p1[-1] < p2[0]:
    print("✓ Success: Indices are sequential and sorted correctly.")
else:
    print("✗ Failure: Indices are not sorted or contain gaps.")
