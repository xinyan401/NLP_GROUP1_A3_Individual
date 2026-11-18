
import re
from datasets import load_dataset


# --------------------------------
# 1. Text Preprocessing Function
# --------------------------------

def preprocess_text(text):

    # Handle missing or invalid text
    if not isinstance(text, str):
        return ""

    # Convert text to lowercase
    text = text.lower()

    # Remove unnecessary punctuation and symbols
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------
# 2. Load QQP Dataset
# --------------------------------

print("Loading QQP Dataset...")

dataset = load_dataset("nyu-mll/glue", "qqp")

# Use a small sample first
df = dataset["train"].select(range(10)).to_pandas()


# --------------------------------
# 3. Apply Preprocessing
# --------------------------------

df["clean_question1"] = df["question1"].apply(preprocess_text)

df["clean_question2"] = df["question2"].apply(preprocess_text)


# --------------------------------
# 4. Display Results
# --------------------------------

print("\nOriginal and Preprocessed Questions")
print("=" * 60)

for index, row in df.iterrows():

    print(f"\nExample {index + 1}")

    print("Original Question 1:")
    print(row["question1"])

    print("Clean Question 1:")
    print(row["clean_question1"])

    print("\nOriginal Question 2:")
    print(row["question2"])

    print("Clean Question 2:")
    print(row["clean_question2"])

    print("Duplicate Label:", row["label"])

    print("-" * 60)


print("\nText Preprocessing Completed Successfully!")
