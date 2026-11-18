
from datasets import load_dataset
import pandas as pd

# Load QQP Dataset
dataset = load_dataset("nyu-mll/glue", "qqp")

# Convert training data into DataFrame
df = pd.DataFrame(dataset["train"])

# Display dataset information
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# Count duplicate and non-duplicate questions
print("\nLabel Distribution:")
print(df["label"].value_counts())

# Display label percentages
print("\nLabel Percentages:")
print(df["label"].value_counts(normalize=True) * 100)

# Display first 5 question pairs
print("\nFirst 5 Question Pairs:")
print(df[["question1", "question2", "label"]].head())
