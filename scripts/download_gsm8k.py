from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")

print(ds)
print(ds["train"][0])