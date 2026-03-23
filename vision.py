from datasets import load_dataset


dataset = load_dataset("ShubUpad/CS-132-Computer-Vision-Midterm")
print(dataset["train"][0])
