# Machine_Learning_Shanghai_University
# 🤖 Machine Learning 

## 📚 Overview

This repository contains multiple **Machine Learning and NLP projects** developed as part of academic practice at Shanghai University.

The main focus is **Sentiment Analysis on movie reviews**, comparing:

- Classical Machine Learning approaches (TF-IDF + Logistic Regression)
- Deep Learning / Transformer models (BERT, RoBERTa, DistilBERT)
- Model evaluation and comparison

---

## 🎯 Project Goals

- Learn how to process and clean text data
- Build classical NLP pipelines
- Train and evaluate ML models
- Use Transformer models for sentiment classification
- Compare performance between different approaches
- Understand real-world ML workflows

---

## 📂 Repository Structure


## 📌 Summary
This project demonstrates a full NLP pipeline using classical Machine Learning:
from raw text → vectorization → model training → prediction → evaluation.


---

## 🧠 Models Used

### 📌 Classical Machine Learning
- TF-IDF Vectorizer
- Logistic Regression

### 📌 Deep Learning / Transformers
- DistilBERT (SST-2)
- RoBERTa (Twitter Sentiment)
- BERT (fine-tuned models)

---

## ⚙️ Pipeline

1. Load dataset (IMDb reviews)
2. Preprocess text data
3. Convert text into numerical features (TF-IDF or tokenization)
4. Train models
5. Make predictions
6. Evaluate accuracy and performance
7. Compare results between models

---

## 📊 Results Summary

| Model                | Type            | Accuracy |
|---------------------|----------------|----------|
| Logistic Regression | Classical ML    | ~88%     |
| DistilBERT         | Transformer     | ~88%     |
| RoBERTa Twitter    | Transformer     | varies   |

---

## 💬 Example Predictions

- "This movie was amazing and unforgettable" → 👍 Positive  
- "This was a terrible boring film" → 👎 Negative  
- "It was okay but too long" → 👎 Negative  

---

## 🔥 CI/CD Pipeline (Practice_ci_cd)

This project also includes a **GitHub Actions pipeline**:

- Automatically runs on push to `main`
- Trains model when dataset changes
- Generates updated model file
- Creates GitHub release automatically

---

## 🛠️ Technologies Used

- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Hugging Face Transformers
- GitHub Actions (CI/CD)

---

## 📈 Key Learnings

- Classical ML is still very strong for sentiment analysis
- Transformers improve robustness but require more computation
- Data preprocessing is more important than model complexity
- CI/CD can automate ML workflows

---

## 📌 Author

Machine Learning student project  
Shanghai University - NLP & AI practice

---

---
