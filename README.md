# Fake-News-Detection

# 📰 Fake News Detection using TF-IDF and Logistic Regression

## 📌 Overview

This machine learning project aims to detect **fake news** using Natural Language Processing (NLP) techniques. The core idea is to train a classifier that can distinguish between real and fake news articles based on their textual content.

We use **TF-IDF vectorization** to convert text into numerical format and apply **Logistic Regression** as the classification model.

---

## 🧠 Problem Statement

In today's digital age, misinformation spreads rapidly. The goal of this project is to build an ML model that can automatically identify whether a news article is **fake** or **real**, helping to combat misinformation.

---

## 📊 Dataset

- The dataset contains two main columns:
  - `text`: The body/content of the news article.
  - `label`: Binary label (e.g., `1` for real news, `0` for fake news).

> You can use datasets like [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) (if not already specified in your project).

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF (TfidfVectorizer)
- Logistic Regression
- Matplotlib / Seaborn (for visualization)

---

## 🔄 Workflow

### 1. Data Preprocessing
- Cleaned text data
- Removed stopwords
- Handled missing values (if any)

### 2. Text Vectorization
- Used `TfidfVectorizer` to convert text into a sparse numerical matrix
- Parameters used:
  - `stop_words='english'`
  - `max_df=0.7` (to ignore very frequent words)

### 3. Model Training
- Split data into training and test sets
- Trained a **Logistic Regression** model on the TF-IDF matrix

### 4. Evaluation
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 📈 Results

The model successfully detects fake news with strong performance on the test set. Results may vary depending on the dataset and preprocessing steps.

---
