
import re
import numpy as np
import gensim.downloader as api

# 1. Load pre-trained GloVe model
print("Loading GloVe Word Embeddings...")

glove_model = api.load("glove-wiki-gigaword-50")

print("GloVe Model Loaded Successfully!")


# 2. Text preprocessing
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

    # Handle sentences with no words in the GloVe vocabulary
    if len(word_vectors) == 0:
        return np.zeros(glove_model.vector_size)

    # Calculate the average of word vectors
    sentence_vector = np.mean(word_vectors, axis=0)

    return sentence_vector


# 4. Test sentence embeddings
sentence1 = "How can I learn Python programming?"
sentence2 = "What is the best way to study Python?"

vector1 = get_sentence_embedding(sentence1)
vector2 = get_sentence_embedding(sentence2)


# 5. Display results
print("\nSentence Embedding Test")
print("------------------------------")

print("\nSentence 1:", sentence1)
print("Sentence Vector:", vector1)
print("Vector Shape:", vector1.shape)

print("\nSentence 2:", sentence2)
print("Sentence Vector:", vector2)
print("Vector Shape:", vector2.shape)

print("\nSentence Embeddings Test Completed!")
