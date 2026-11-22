# BAI3243 Natural Language Processing
## Individual Assignment — A3: Word-Vector Similarity Scorer

### 1. Project Overview

This project implements a Word-Vector Similarity Scorer for semantic similarity and duplicate question detection.

The system uses pre-trained GloVe word embeddings to represent individual words as numerical vectors. It then generates sentence representations using two different methods:

- Average Word Embeddings
- TF-IDF Weighted Word Embeddings

Cosine Similarity is used to measure the similarity between two question vectors. A similarity threshold selected using validation data is applied to classify question pairs as Duplicate or Non-Duplicate.

An interactive Explainable Duplicate Question Checker is also developed to allow users to enter two questions and examine the similarity results and additional linguistic diagnostic information.

### 2. Technologies and Libraries

- Python 3.12
- Gensim
- NumPy
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Hugging Face Datasets

### 3. Dataset and Pre-trained Model

**Dataset:** Quora Question Pairs (QQP), obtained through the GLUE dataset on Hugging Face.

Dataset source:
https://huggingface.co/datasets/nyu-mll/glue

The experiment uses 10,000 selected question pairs from the training split:

| Dataset Split | Number of Samples |
|---|---:|
| Training | 6,000 |
| Validation | 2,000 |
| Testing | 2,000 |
| Total | 10,000 |

**Pre-trained Word Embeddings:** GloVe Wiki-Gigaword-50.

The model is loaded using the Gensim Downloader API. Each word is represented by a 50-dimensional vector.

### 4. Project Files

| File | Description |
|---|---|
| `main.py` | Loads and explores the QQP dataset. |
| `preprocessing.py` | Performs text preprocessing. |
| `word_embeddings.py` | Loads the pre-trained GloVe model. |
| `sentence_embeddings.py` | Generates sentence vectors using Average Word Embeddings. |
| `similarity_scorer.py` | Calculates sentence similarity using Cosine Similarity. |
| `final_experiment.py` | Implements the final experiment, including dataset splitting, both embedding methods, threshold selection and model evaluation. |
| `app_explainer.py` | Runs the interactive Explainable Duplicate Question Checker. |
| `requirements.txt` | Lists the required Python packages. |

### 5. Installation

**Step 1: Create a virtual environment**

```bash
python -m venv .venv
```

**Step 2: Activate the virtual environment**

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

**Step 3: Install the required packages**

```bash
pip install -r requirements.txt
```

### 6. Running the Project

**A. Run the Final Experiment**

```bash
python final_experiment.py
```

This program performs the final experimental evaluation of both sentence representation methods and displays their evaluation results, including Accuracy, Precision, Recall, F1-score and Confusion Matrix.

**B. Run the Interactive Application**

```bash
python app_explainer.py
```

The application allows users to enter two questions and displays:

1. Common Content Keywords
2. Semantic Word-Pair Alignment
3. Out-of-Vocabulary (OOV) Check
4. Dual-Model Results and Agreement
5. Diagnostic Alerts and Limitations

Enter `exit` to close the application.

### 7. Automatic Downloads

The following resources are downloaded or loaded automatically when the corresponding programs are executed:

- The QQP dataset is accessed through Hugging Face Datasets.
- The GloVe Wiki-Gigaword-50 model is loaded through Gensim Downloader.

An internet connection is required when these resources are not already available in the local cache.

### 8. Experimental Results

The final test-set results are summarised below:

| Evaluation Metric | Average Word Embeddings | TF-IDF Weighted Word Embeddings |
|---|---:|---:|
| Accuracy | 0.6400 | 0.5985 |
| Precision | 0.5049 | 0.4705 |
| Recall | 0.7082 | 0.7973 |
| F1-score | 0.5895 | 0.5918 |
| Best Threshold | 0.965 | 0.930 |

The TF-IDF Weighted Word Embeddings method achieved a slightly higher F1-score, while Average Word Embeddings achieved higher accuracy and precision.

### 9. Notes

- The experiment uses a selected subset of 10,000 labelled QQP training examples.
- The TF-IDF vectorizer is fitted using training questions only.
- Similarity thresholds are selected using validation data.
- Final performance is evaluated on the independent test split.
- The interactive diagnostic alerts provide additional explanations but do not modify the underlying model predictions.

---

**Course:** BAI3243 Natural Language Processing  
**Individual Task:** A3 — Word-Vector Similarity Scorer