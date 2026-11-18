
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# 1. Load GloVe Word Embeddings
# ==========================================

print("Loading GloVe Word Embeddings...")

glove_model = api.load("glove-wiki-gigaword-50")

print("GloVe Model Loaded Successfully!")


# ==========================================
# 2. Text Preprocessing
# ==========================================

def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 3. Load QQP Dataset
# ==========================================

print("\nLoading QQP Dataset...")

dataset = load_dataset("nyu-mll/glue", "qqp")

df = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(10000))
    .to_pandas()
)

df = df.dropna(subset=["question1", "question2", "label"])
df = df.reset_index(drop=True)

print("Selected Question Pairs:", len(df))


# ==========================================
# 4. Train / Validation / Test Split
# ==========================================

train_df, remaining_df = train_test_split(
    df,
    test_size=0.40,
    random_state=42,
    stratify=df["label"]
)

validation_df, test_df = train_test_split(
    remaining_df,
    test_size=0.50,
    random_state=42,
    stratify=remaining_df["label"]
)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nDataset Split Completed")
print("Training Samples:", len(train_df))
print("Validation Samples:", len(validation_df))
print("Test Samples:", len(test_df))


# ==========================================
# 5. Prepare TF-IDF using Training Data Only
# ==========================================

print("\nFitting TF-IDF Vectorizer on Training Data...")

training_questions = (
    train_df["question1"].tolist()
    + train_df["question2"].tolist()
)

tfidf_vectorizer = TfidfVectorizer(
    preprocessor=preprocess_text,
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_vectorizer.fit(training_questions)

feature_names = tfidf_vectorizer.get_feature_names_out()

print("TF-IDF Fitted Successfully!")
print("Vocabulary Size:", len(feature_names))


# ==========================================
# 6. Average Word Embeddings
# ==========================================

def get_average_embedding(sentence):

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


# ==========================================
# 7. TF-IDF Weighted Word Embeddings
# ==========================================

def get_tfidf_embedding(sentence):

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


# ==========================================
# 8. Cache Sentence Embeddings
# ==========================================

average_cache = {}
tfidf_cache = {}


def cached_average_embedding(question):

    key = preprocess_text(question)

    if key not in average_cache:
        average_cache[key] = get_average_embedding(key)

    return average_cache[key]


def cached_tfidf_embedding(question):

    key = preprocess_text(question)

    if key not in tfidf_cache:
        tfidf_cache[key] = get_tfidf_embedding(key)

    return tfidf_cache[key]


# ==========================================
# 9. Calculate Cosine Similarity
# ==========================================

def cosine_score(vector1, vector2):

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(
        np.dot(vector1, vector2) / (norm1 * norm2)
    )


def calculate_scores(data, embedding_function):

    scores = []

    for q1, q2 in zip(
        data["question1"],
        data["question2"]
    ):

        vector1 = embedding_function(q1)
        vector2 = embedding_function(q2)

        scores.append(cosine_score(vector1, vector2))

    return np.array(scores)


# ==========================================
# 10. Find Best Threshold on Validation Data
# ==========================================

def find_best_threshold(scores, labels):

    best_threshold = 0.0
    best_f1 = -1.0

    for threshold in np.arange(-1.0, 1.001, 0.005):

        predictions = (scores >= threshold).astype(int)

        current_f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ==========================================
# 11. Evaluate Model on Test Data
# ==========================================

def evaluate_model(name, validation_scores, test_scores):

    validation_labels = validation_df["label"].to_numpy()
    test_labels = test_df["label"].to_numpy()

    threshold, validation_f1 = find_best_threshold(
        validation_scores,
        validation_labels
    )

    test_predictions = (
        test_scores >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        test_labels, test_predictions
    )

    precision = precision_score(
        test_labels, test_predictions,
        zero_division=0
    )

    recall = recall_score(
        test_labels, test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        test_labels, test_predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        test_labels, test_predictions
    )

    print(f"\n{name} Results")
    print("------------------------------")
    print("Best Threshold:", round(threshold, 3))
    print("Validation F1-score:", round(validation_f1, 4))
    print("Test Accuracy:", round(accuracy, 4))
    print("Test Precision:", round(precision, 4))
    print("Test Recall:", round(recall, 4))
    print("Test F1-score:", round(f1, 4))
    print("Confusion Matrix:")
    print(matrix)

    return {
        "name": name,
        "threshold": threshold,
        "validation_f1": validation_f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matrix": matrix,
        "test_predictions": test_predictions
    }


# ==========================================
# 12. Run Baseline Model
# ==========================================

print("\nGenerating Baseline Embeddings...")

baseline_validation_scores = calculate_scores(
    validation_df, cached_average_embedding
)

baseline_test_scores = calculate_scores(
    test_df, cached_average_embedding
)

print("Baseline Embeddings Generated!")

baseline_result = evaluate_model(
    "Average Word Embeddings",
    baseline_validation_scores,
    baseline_test_scores
)


# ==========================================
# 13. Run TF-IDF Weighted Model
# ==========================================

print("\nGenerating TF-IDF Weighted Embeddings...")

tfidf_validation_scores = calculate_scores(
    validation_df, cached_tfidf_embedding
)

tfidf_test_scores = calculate_scores(
    test_df, cached_tfidf_embedding
)

print("TF-IDF Embeddings Generated!")

tfidf_result = evaluate_model(
    "TF-IDF Weighted Word Embeddings",
    tfidf_validation_scores,
    tfidf_test_scores
)


# ==========================================
# 14. Compare Both Models
# ==========================================

print("\nFINAL MODEL COMPARISON")
print("==============================")

print(
    f"{'Metric':<18}"
    f"{'Average':>15}"
    f"{'TF-IDF':>15}"
)

metrics = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-score", "f1")
]

for label, key in metrics:

    print(
        f"{label:<18}"
        f"{baseline_result[key]:>15.4f}"
        f"{tfidf_result[key]:>15.4f}"
    )


# ==========================================
# 15. Save Similarity Distribution Graphs
# ==========================================

os.makedirs("screenshots", exist_ok=True)

def save_distribution_plot(
    scores,
    labels,
    threshold,
    filename,
    title
):

    duplicate_scores = scores[labels == 1]
    non_duplicate_scores = scores[labels == 0]

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
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.3f}"
    )

    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()


validation_labels = validation_df["label"].to_numpy()

save_distribution_plot(
    baseline_validation_scores,
    validation_labels,
    baseline_result["threshold"],
    "screenshots/final_average_distribution.png",
    "Average Word Embeddings Distribution"
)

save_distribution_plot(
    tfidf_validation_scores,
    validation_labels,
    tfidf_result["threshold"],
    "screenshots/final_tfidf_distribution.png",
    "TF-IDF Weighted Similarity Distribution"
)


# ==========================================
# 16. Test New Question Pair
# ==========================================

question1 = "How can I learn Python programming?"
question2 = "What is the best way to study Python?"

print("\nNEW QUESTION PAIR TEST")
print("==============================")

print("Question 1:", question1)
print("Question 2:", question2)

for result, embedding_function in [
    (baseline_result, cached_average_embedding),
    (tfidf_result, cached_tfidf_embedding)
]:

    vector1 = embedding_function(question1)
    vector2 = embedding_function(question2)

    score = cosine_score(vector1, vector2)

    prediction = (
        "Duplicate"
        if score >= result["threshold"]
        else "Non-Duplicate"
    )

    print(f"\n{result['name']}")
    print("Similarity Score:", round(score, 4))
    print("Threshold:", round(result["threshold"], 3))
    print("Prediction:", prediction)


print("\nFinal Experiment Completed Successfully!")
