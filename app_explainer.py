import re
import numpy as np
import gensim.downloader as api

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ==========================================
# 1. Load GloVe Model
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


NEGATION_WORDS = {"not", "no", "nor", "never"}

# Keep negation words out of stop words so we can detect them separately.
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS


def get_content_words(sentence):

    words = preprocess_text(sentence).split()

    return [
        word for word in words
        if word not in STOP_WORDS and word not in NEGATION_WORDS
    ]


# ==========================================
# 3. Load QQP and Fit TF-IDF
# ==========================================

print("\nLoading QQP Dataset...")

dataset = load_dataset("nyu-mll/glue", "qqp")

df = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(10000))
    .to_pandas()
)

df = df.dropna(
    subset=["question1", "question2", "label"]
).reset_index(drop=True)


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

print("TF-IDF Vectorizer Loaded Successfully!")
print("Training Vocabulary Size:", len(feature_names))


# ==========================================
# 4. Model Thresholds
# ==========================================

# Thresholds obtained from final_experiment.py
AVERAGE_THRESHOLD = 0.965
TFIDF_THRESHOLD = 0.930

# Minimum word-level similarity to display in semantic alignment
MIN_WORD_SIMILARITY = 0.50


# ==========================================
# 5. Average Word Embeddings
# ==========================================

def get_average_embedding(sentence):

    words = preprocess_text(sentence).split()

    vectors = [
        glove_model[word]
        for word in words
        if word in glove_model
    ]

    if not vectors:
        return np.zeros(glove_model.vector_size)

    return np.mean(vectors, axis=0)


# ==========================================
# 6. TF-IDF Weighted Word Embeddings
# ==========================================

def get_tfidf_embedding(sentence):

    vector = tfidf_vectorizer.transform([sentence])

    weighted_sum = np.zeros(glove_model.vector_size)
    total_weight = 0.0

    for index, weight in zip(
        vector.indices,
        vector.data
    ):

        word = feature_names[index]

        if word in glove_model:

            weighted_sum += weight * glove_model[word]
            total_weight += weight

    if total_weight == 0:
        return np.zeros(glove_model.vector_size)

    return weighted_sum / total_weight


# ==========================================
# 7. Cosine Similarity
# ==========================================

def cosine_score(vector1, vector2):

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(
        np.dot(vector1, vector2) / (norm1 * norm2)
    )


# ==========================================
# 8. Common Keyword Analysis
# ==========================================

def show_common_keywords(question1, question2):

    words1 = get_content_words(question1)
    words2 = get_content_words(question2)

    common_words = sorted(set(words1) & set(words2))

    print("\n[1] COMMON CONTENT KEYWORDS")
    print("--------------------------------")

    if not common_words:
        print("No common content keywords found.")
        return

    vector1 = tfidf_vectorizer.transform([question1])
    vector2 = tfidf_vectorizer.transform([question2])

    weights1 = {
        feature_names[i]: weight
        for i, weight in zip(
            vector1.indices,
            vector1.data
        )
    }

    weights2 = {
        feature_names[i]: weight
        for i, weight in zip(
            vector2.indices,
            vector2.data
        )
    }

    for word in common_words:

        print(
            f"{word}: "
            f"Q1 TF-IDF={weights1.get(word, 0.0):.4f}, "
            f"Q2 TF-IDF={weights2.get(word, 0.0):.4f}"
        )


# ==========================================
# 9. Semantic Word-Pair Alignment
# ==========================================

def show_semantic_pairs(question1, question2):

    words1 = set(get_content_words(question1))
    words2 = set(get_content_words(question2))

    common_words = words1 & words2

    # Only compare non-identical content words that exist in GloVe
    unique_words1 = [
        word for word in (words1 - common_words)
        if word in glove_model
    ]

    unique_words2 = [
        word for word in (words2 - common_words)
        if word in glove_model
    ]

    print("\n[2] SEMANTIC WORD-PAIR ALIGNMENT")
    print("--------------------------------")

    if not unique_words1 or not unique_words2:
        print("No unique content words left for cross-question comparison.")
        return

    candidates = []

    for word1 in unique_words1:

        for word2 in unique_words2:

            score = float(glove_model.similarity(word1, word2))

            if score >= MIN_WORD_SIMILARITY:
                candidates.append((word1, word2, score))

    candidates.sort(
        key=lambda item: item[2],
        reverse=True
    )

    used_words1 = set()
    used_words2 = set()

    selected_pairs = []

    for word1, word2, score in candidates:

        if word1 in used_words1:
            continue

        if word2 in used_words2:
            continue

        selected_pairs.append(
            (word1, word2, score)
        )

        used_words1.add(word1)
        used_words2.add(word2)

        if len(selected_pairs) == 3:
            break

    if not selected_pairs:
        print(
            f"No strong semantic word pairs found "
            f"(minimum similarity >= {MIN_WORD_SIMILARITY:.2f})."
        )
        return

    for word1, word2, score in selected_pairs:

        print(
            f"{word1} <-> {word2}: "
            f"GloVe Similarity = {score:.4f}"
        )

    print(
        "Note: These are word-level similarities between "
        "non-identical content words."
    )


# ==========================================
# 10. Out-of-Vocabulary Check
# ==========================================

def show_oov_words(question1, question2):

    print("\n[3] OUT-OF-VOCABULARY CHECK")
    print("--------------------------------")

    for label, question in [
        ("Question 1", question1),
        ("Question 2", question2)
    ]:

        words = preprocess_text(question).split()

        oov_words = sorted(
            set(
                word for word in words
                if word not in glove_model
            )
        )

        print(
            f"{label} OOV Words: "
            f"{oov_words if oov_words else 'None'}"
        )


# ==========================================
# 11. Dual-Model Prediction
# ==========================================

def show_model_results(question1, question2):

    print("\n[4] DUAL-MODEL RESULTS")
    print("================================")

    models = [
        (
            "Average Word Embeddings",
            get_average_embedding,
            AVERAGE_THRESHOLD
        ),
        (
            "TF-IDF Weighted Word Embeddings",
            get_tfidf_embedding,
            TFIDF_THRESHOLD
        )
    ]

    predictions = []

    for name, embedding_function, threshold in models:

        vector1 = embedding_function(question1)
        vector2 = embedding_function(question2)

        score = cosine_score(vector1, vector2)

        prediction = (
            "Duplicate"
            if score >= threshold
            else "Non-Duplicate"
        )

        predictions.append(prediction)

        print(f"\nModel: {name}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Threshold: {threshold:.3f}")
        print(f"Prediction: {prediction}")

    print("\nModel Agreement")
    print("--------------------------------")

    if predictions[0] == predictions[1]:

        print("Both models produced the same prediction.")

    else:

        print("The two models produced different predictions.")
        print(
            "Their sentence representation methods "
            "may capture different information."
        )


# ==========================================
# 12. Diagnostic Alerts & Model Limitation
# ==========================================

def show_diagnostic_alerts(question1, question2):

    print("\n[5] DIAGNOSTIC ALERTS & LIMITATION")
    print("--------------------------------")

    tokens1 = preprocess_text(question1).split()
    tokens2 = preprocess_text(question2).split()

    neg1 = set(tokens1) & NEGATION_WORDS
    neg2 = set(tokens2) & NEGATION_WORDS

    alert_triggered = False

    # Check 1: Negation mismatch
    if neg1 != neg2:
        print(
            "Alert [Negation Mismatch]: One question contains "
            f"negation ({neg1 if neg1 else 'None'} vs {neg2 if neg2 else 'None'})."
        )
        print(
            "Word-vector averaging often fails to capture reversed meaning."
        )
        alert_triggered = True

    # Check 2: Same content words in a different order
    content1 = get_content_words(question1)
    content2 = get_content_words(question2)

    if content1 and set(content1) == set(content2) and content1 != content2:
        print(
            "Alert [Word Order Difference]: Both questions share the exact "
            "same content words in a different order."
        )
        print(
            "Word-vector averaging ignores word order and may overestimate similarity."
        )
        alert_triggered = True

    if not alert_triggered:
        print(
            "Note: Bag-of-embeddings models do not explicitly preserve "
            "word order or deep contextual meaning."
        )


# ==========================================
# 13. Interactive Question Checker
# ==========================================

def main():

    print("\n======================================")
    print("EXPLAINABLE DUPLICATE QUESTION CHECKER")
    print("======================================")

    print("Enter 'exit' to close the application.")

    while True:

        print("\n--------------------------------------")

        question1 = input("Enter Question 1: ")

        if question1.lower().strip() == "exit":
            break

        question2 = input("Enter Question 2: ")

        if question2.lower().strip() == "exit":
            break

        if not question1.strip() or not question2.strip():

            print("Please enter two non-empty questions.")
            continue

        print("\nQUESTION 1:", question1)
        print("QUESTION 2:", question2)

        show_common_keywords(question1, question2)

        show_semantic_pairs(question1, question2)

        show_oov_words(question1, question2)

        show_model_results(question1, question2)

        show_diagnostic_alerts(question1, question2)

    print("\nApplication Closed!")


if __name__ == "__main__":
    main()