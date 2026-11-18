
import re
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load GloVe Word Embeddings
print("Loading GloVe Word Embeddings...")

glove_model = api.load("glove-wiki-gigaword-50")

print("GloVe Model Loaded Successfully!")


# 2. Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 3. Convert a sentence into a vector
def get_sentence_embedding(sentence):

    sentence = preprocess_text(sentence)
    words = sentence.split()

    word_vectors = []

    for word in words:
        if word in glove_model:
            word_vectors.append(glove_model[word])

    if len(word_vectors) == 0:
        return np.zeros(glove_model.vector_size)

    sentence_vector = np.mean(word_vectors, axis=0)

    return sentence_vector


# 4. Calculate Cosine Similarity
def calculate_similarity(question1, question2):

    vector1 = get_sentence_embedding(question1)
    vector2 = get_sentence_embedding(question2)

    # Handle empty or out-of-vocabulary questions
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0.0

    score = cosine_similarity(
        vector1.reshape(1, -1),
        vector2.reshape(1, -1)
    )[0][0]

    return float(score)


# 5. Test Question Pairs
question1 = "How can I learn Python programming?"
question2 = "What is the best way to study Python?"

question3 = "What is the capital of Malaysia?"


# 6. Display Similarity Results
print("\nCosine Similarity Test")
print("------------------------------")

score1 = calculate_similarity(question1, question2)
score2 = calculate_similarity(question1, question3)

print("\nQuestion 1:", question1)
print("Question 2:", question2)
print("Similarity Score:", round(score1, 4))

print("\nQuestion 1:", question1)
print("Question 3:", question3)
print("Similarity Score:", round(score2, 4))

print("\nCosine Similarity Test Completed!")
