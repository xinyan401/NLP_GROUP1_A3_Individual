
import re
import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load GloVe model
print("Loading GloVe Word Embeddings...")
glove_model = api.load("glove-wiki-gigaword-50")
print("GloVe Model Loaded Successfully!")


# 2. Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 3. Convert a question into a sentence vector
def get_sentence_embedding(sentence):

    sentence = preprocess_text(sentence)
    words = sentence.split()

    word_vectors = [
        glove_model[word]
        for word in words
        if word in glove_model
    ]

    if len(word_vectors) == 0:
        return np.zeros(glove_model.vector_size)

    return np.mean(word_vectors, axis=0)


# 4. Calculate similarity for question pairs
def calculate_pair_similarity(question1, question2):

    vector1 = get_sentence_embedding(question1)
    vector2 = get_sentence_embedding(question2)

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    score = np.dot(vector1, vector2) / (norm1 * norm2)

    return float(score)


# 5. Load QQP dataset
print("\nLoading QQP Dataset...")

dataset = load_dataset("nyu-mll/glue", "qqp")

df = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(10000))
    .to_pandas()
)

print("Selected Question Pairs:", len(df))


# 6. Split data into validation and test sets
validation_df, test_df = train_test_split(
    df,
    test_size=0.5,
    random_state=42,
    stratify=df["label"]
)

print("\nDataset Split Completed")
print("Validation Samples:", len(validation_df))
print("Test Samples:", len(test_df))


# 7. Cache sentence vectors to avoid repeated calculations
print("\nGenerating Sentence Embeddings...")

embedding_cache = {}

def cached_embedding(question):

    key = preprocess_text(question)

    if key not in embedding_cache:
        embedding_cache[key] = get_sentence_embedding(key)

    return embedding_cache[key]


def get_similarity_scores(data):

    scores = []

    for q1, q2 in zip(data["question1"], data["question2"]):

        vector1 = cached_embedding(q1)
        vector2 = cached_embedding(q2)

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            score = 0.0
        else:
            score = np.dot(vector1, vector2) / (norm1 * norm2)

        scores.append(float(score))

    return np.array(scores)


validation_scores = get_similarity_scores(validation_df)
test_scores = get_similarity_scores(test_df)

print("Sentence Embeddings Generated Successfully!")


# 8. Select the best threshold using validation data
print("\nSelecting Similarity Threshold...")

y_validation = validation_df["label"].to_numpy()

best_threshold = 0.0
best_f1 = -1.0

for threshold in np.arange(0.50, 1.00, 0.005):

    predictions = (validation_scores >= threshold).astype(int)

    current_f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = float(threshold)

print("Best Threshold:", round(best_threshold, 2))
print("Validation F1-score:", round(best_f1, 4))

# Analyze similarity score distribution
print("\nSimilarity Score Distribution")
print("------------------------------")

duplicate_scores = validation_scores[y_validation == 1]
non_duplicate_scores = validation_scores[y_validation == 0]

print("\nDuplicate Question Pairs:")
print("Mean Similarity:",
      round(float(np.mean(duplicate_scores)), 4))
print("Minimum Similarity:",
      round(float(np.min(duplicate_scores)), 4))
print("Maximum Similarity:",
      round(float(np.max(duplicate_scores)), 4))

print("\nNon-Duplicate Question Pairs:")
print("Mean Similarity:",
      round(float(np.mean(non_duplicate_scores)), 4))
print("Minimum Similarity:",
      round(float(np.min(non_duplicate_scores)), 4))
print("Maximum Similarity:",
      round(float(np.max(non_duplicate_scores)), 4))


# Plot similarity score distributions
plt.figure(figsize=(10, 6))

plt.hist(
    non_duplicate_scores,
    bins=30,
    alpha=0.6,
    label="Non-Duplicate",
    density=True
)

plt.hist(
    duplicate_scores,
    bins=30,
    alpha=0.6,
    label="Duplicate",
    density=True
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Threshold = {best_threshold:.3f}"
)

plt.xlabel("Cosine Similarity Score")
plt.ylabel("Density")
plt.title("Similarity Score Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(
    "screenshots/similarity_distribution.png",
    dpi=300
)

plt.show()




# 9. Evaluate using independent test data
print("\nFinal Test Results")
print("------------------------------")

y_test = test_df["label"].to_numpy()

test_predictions = (
    test_scores >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(
    y_test, test_predictions, zero_division=0
)
recall = recall_score(
    y_test, test_predictions, zero_division=0
)
f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


# 10. Test a new question pair
print("\nNew Question Pair Prediction")
print("------------------------------")

question1 = "How can I learn Python programming?"
question2 = "What is the best way to study Python?"

new_score = calculate_pair_similarity(question1, question2)

if new_score >= best_threshold:
    prediction = "Duplicate"
else:
    prediction = "Non-Duplicate"

print("Question 1:", question1)
print("Question 2:", question2)
print("Similarity Score:", round(new_score, 4))
print("Selected Threshold:", round(best_threshold, 2))
print("Prediction:", prediction)

print("\nDuplicate Question Detection Completed!")
