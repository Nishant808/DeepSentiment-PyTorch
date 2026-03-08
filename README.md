# DeepSentiment-PyTorch: RNN-Based Sentiment Analysis

A robust Natural Language Processing (NLP) pipeline and Recurrent Neural Network (RNN) built from scratch using PyTorch. This project performs binary sentiment classification (Positive/Negative) on the IMDB Movie Reviews dataset, achieving over 83% accuracy on the test set.

## 🚀 Project Overview

This repository demonstrates a complete end-to-end machine learning workflow for text classification. It includes extensive text preprocessing, feature extraction using TF-IDF, and the training of a custom deep learning model. The objective is to accurately predict the underlying sentiment of highly unstructured movie reviews.

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Deep Learning Framework:** PyTorch (`torch.nn`, `torch.optim`, `DataLoader`)
* **Machine Learning:** Scikit-Learn (`TfidfVectorizer`, `LabelEncoder`, `train_test_split`)
* **NLP Processing:** NLTK (Tokenization, Stopwords removal, Porter Stemming)
* **Data Manipulation:** Pandas, NumPy, Regular Expressions (`re`)

## 🧠 Pipeline & Methodology

### 1. Data Preprocessing
Raw text data is notoriously messy. A custom cleaning pipeline was implemented to normalize the dataset (49,582 unique reviews):
* Lowercasing all text.
* Removal of URLs, HTML tags, and non-alphanumeric punctuation using Regex.
* Stopword removal using NLTK to filter out low-value semantic words.
* Word stemming (PorterStemmer) to reduce words to their base or root form.

### 2. Feature Engineering
* **TF-IDF Vectorization:** Converted the cleaned text into a sparse matrix of numerical features, limiting the vocabulary to the top 5,000 most significant words (`max_features=5000`).
* **Label Encoding:** Mapped categorical sentiments ('positive', 'negative') to binary integers (1, 0).

### 3. Model Architecture
A custom Recurrent Neural Network (RNN) tailored for sequential data processing.
* **Input Layer:** 5,000 features (TF-IDF output).
* **Hidden Layer:** RNN layer with 128 hidden units (`batch_first=True`).
* **Output Layer:** Fully connected (`nn.Linear`) layer mapping to a single output logit.
* **Loss Function:** Binary Cross Entropy with Logits (`BCEWithLogitsLoss`) for enhanced numerical stability.
* **Optimizer:** Adam Optimizer.

## 📊 Results & Performance

The model was trained for 10 epochs with a batch size of 64. 
* **Validation Split:** 80/20 Train-Test split.
* **Final Test Accuracy:** **83.20%**
* **Convergence:** The model showed steady loss reduction, stabilizing around an average training loss of 0.272 by the final epoch.

## 💻 How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/DeepSentiment-PyTorch.git](https://github.com/yourusername/DeepSentiment-PyTorch.git)
   cd DeepSentiment-PyTorch
