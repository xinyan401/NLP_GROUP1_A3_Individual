
import re
import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
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


# 3. Load QQP dataset
print("\nLoading QQP Dataset...")
dataset = load_dataset("nyu-mll/glue", "qqp")

df = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(10000))
    .to_pandas()
)

print("Selected Question Pairs:", len(df))


# 4. Split into validation and test sets
validation_df, test_df = train_test_split(
    df,
    test_size=0.5,
    random_state=42,
    stratify=df["label"]
)

print("\nDataset Split Completed")
print("Validation Samples:", len(validation_df))
print("Test Samples:", len(test_df))


# 5. Fit TF-IDF using validation questions
print("\nFitting TF-IDF Vectorizer...")

validation_questions = (
    validation_df["question1"].fillna("").tolist()
    + validation_df["question2"].fillna("").tolist()
)

tfidf_vectorizer = TfidfVectorizer(
    preprocessor=preprocess_text,
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_vectorizer.fit(validation_questions)

feature_names = tfidf_vectorizer.get_feature_names_out()

print("TF-IDF Vectorizer Fitted Successfully!")
print("Vocabulary Size:", len(feature_names))


# 6. Convert a question into a TF-IDF weighted sentence vector
def get_weighted_embedding(sentence):

    if not isinstance(sentence, str):
        sentence = ""

    tfidf_vector = tfidf_vectorizer.transform([sentence])

    indices = tfidf_vector.indices
    weights = tfidf_vector.data

    weighted_sum = np.zeros(glove_model.vector_size)
    total_weight = 0.0

    for index, weight in zip(indices, weights):

        word = feature_names[index]

        if word in glove_model:
            weighted_sum += weight * glove_model[word]
            total_weight += weight

    if total_weight == 0:
        return np.zeros(glove_model.vector_size)

    return weighted_sum / total_weight


# 7. Cache embeddings
print("\nGenerating TF-IDF Weighted Sentence Embeddings...")

embedding_cache = {}

def cached_embedding(question):

    key = preprocess_text(question)

    if key not in embedding_cache:
        embedding_cache[key] = get_weighted_embedding(key)

    return embedding_cache[key]


# 8. Calculate similarity scores
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

print("Weighted Sentence Embeddings Generated Successfully!")


# 9. Select threshold using validation data
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

print("Best Threshold:", round(best_threshold, 3))
print("Validation F1-score:", round(best_f1, 4))


# 10. Evaluate on independent test data
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


# 11. Plot similarity score distributions
duplicate_scores = validation_scores[y_validation == 1]
non_duplicate_scores = validation_scores[y_validation == 0]

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
plt.title("TF-IDF Weighted Similarity Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(
    "screenshots/tfidf_similarity_distribution.png",
    dpi=300
)

plt.show()


# 12. Test a new question pair
print("\nNew Question Pair Prediction")
print("------------------------------")

question1 = "How can I learn Python programming?"
question2 = "What is the best way to study Python?"

new_vector1 = get_weighted_embedding(question1)
new_vector2 = get_weighted_embedding(question2)

norm1 = np.linalg.norm(new_vector1)
norm2 = np.linalg.norm(new_vector2)

if norm1 == 0 or norm2 == 0:
    new_score = 0.0
else:
    new_score = float(
        np.dot(new_vector1, new_vector2) / (norm1 * norm2)
    )

prediction = (
    "Duplicate"
    if new_score >= best_threshold
    else "Non-Duplicate"
)

print("Question 1:", question1)
print("Question 2:", question2)
print("Similarity Score:", round(new_score, 4))
print("Selected Threshold:", round(best_threshold, 3))
print("Prediction:", prediction)

print("\nTF-IDF Weighted Duplicate Detection Completed!")
