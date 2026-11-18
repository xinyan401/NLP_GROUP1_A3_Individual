
import gensim.downloader as api


# --------------------------------
# 1. Load Pre-trained GloVe Model
# --------------------------------

print("Loading GloVe Word Embeddings...")
print("Please wait. The model will be downloaded if not cached.")

glove_model = api.load("glove-wiki-gigaword-50")

print("\nGloVe Model Loaded Successfully!")


# --------------------------------
# 2. Display Model Information
# --------------------------------

print("\nModel Information")
print("---------------------------")

print("Model Name: GloVe Wiki-Gigaword-50")
print("Vector Dimensions:", glove_model.vector_size)
print("Vocabulary Size:", len(glove_model))


# --------------------------------
# 3. Test Word Embeddings
# --------------------------------

words = ["python", "programming", "computer", "student"]

print("\nWord Vector Examples")
print("---------------------------")

for word in words:

    if word in glove_model:

        vector = glove_model[word]

        print(f"\nWord: {word}")
        print("Vector:", vector)
        print("Vector Shape:", vector.shape)

    else:

        print(f"{word} is not in the vocabulary.")


# --------------------------------
# 4. Test Word Similarity
# --------------------------------

print("\nWord Similarity Test")
print("---------------------------")

similarity = glove_model.similarity("python", "programming")

print("Similarity between 'python' and 'programming':")
print(round(float(similarity), 4))


print("\nWord Embeddings Test Completed!")
