
import evaluate

rouge = evaluate.load("rouge")

garbage_summary = "– It's that time of the year again to recognize the accomplishments of women in the military and in the civilian world. To celebrate, we've pulled together some of the best photos of the men and women who have served in the US armed forces, in Iraq and Afghanistan, in the first Persian Gulf War, and in Ramadi, the capital of Abu Dhabi in the United Arab Emirates. More than a dozen of these women and men were awarded the Bronze Star, the distinction awarded to those who, in combat or in uniform, were either killed or were seriously wounded in Iraq or in Iraq in the Persian Gulf or in fighting in Iraq."

# Get some real references
from datasets import load_dataset
dataset = load_dataset("Awesome075/multi_news_parquet", split="test")

references = [dataset[i]['summary'] for i in range(100)]
predictions = [garbage_summary] * 100

results = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1 with identical garbage: {results['rouge1'] * 100:.2f}%")
