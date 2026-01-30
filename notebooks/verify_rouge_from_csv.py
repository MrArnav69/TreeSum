
import pandas as pd
import evaluate
import os

# Load the data that was just generated
csv_path = '/Users/mrarnav69/Documents/FactSum/results/test_10_samples.csv'
if not os.path.exists(csv_path):
    print("Error: CSV file not found.")
    exit(1)

df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} samples from {csv_path}")

# Load ROUGE
rouge = evaluate.load('rouge')

# Calculate Scores
results = rouge.compute(
    predictions=df['generated'].tolist(),
    references=df['reference'].tolist(),
    use_aggregator=True
)

print("\n=== INDEPENDENT VERIFICATION ===")
print(f"ROUGE-1: {results['rouge1']*100:.2f}")
print(f"ROUGE-2: {results['rouge2']*100:.2f}")
print(f"ROUGE-L: {results['rougeL']*100:.2f}")
print("================================")
