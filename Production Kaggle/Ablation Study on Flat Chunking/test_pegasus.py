
import torch
from transformers import PegasusForConditionalGeneration, AutoTokenizer

model_name = "google/pegasus-multi_news"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to("cpu")

def generate(text):
    batch = tokenizer([text], truncation=True, padding="longest", max_length=1024, return_tensors="pt")
    with torch.no_grad():
        summary_ids = model.generate(
            batch["input_ids"],
            num_beams=8,
            max_length=512,
            min_length=64,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Testing empty string:")
print(f"[{generate('')}]")

print("\nTesting 'Hello World':")
print(f"[{generate('Hello World')}]")

print("\nTesting real snippet:")
print(f"[{generate('The Secret Service prostitution scandal that threatening to overshadow the Summit of the Americas came to light after one agent refused to pay one of the women involved.')}]")
