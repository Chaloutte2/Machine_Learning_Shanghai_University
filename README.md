# Machine_Learning_Shanghai_University
# 📊 Machine Learning - Sentiment Analysis Project

## 🚀 Objective
This project builds a simple sentiment analysis model that classifies movie reviews:

- 👍 **1 = Positive**
- 👎 **0 = Negative**

The goal is to learn how to apply classical Machine Learning techniques to NLP tasks.

---

## 🧠 Models Used
- TF-IDF (Text feature extraction)
- Logistic Regression (Classification model)

---

## 📂 Dataset
We use a balanced IMDb dataset containing **10,000 movie reviews**.

Each review is labeled as:
- `0` → Negative review  
- `1` → Positive review  

---

## ⚙️ Pipeline Overview

1. Load dataset  
2. Split into training and testing sets  
3. Convert text into numerical features using TF-IDF  
4. Train Logistic Regression model  
5. Make predictions  
6. Evaluate performance  
7. Analyze most important words  

---

## 📈 Results

The model achieves around:

> 🎯 **88% accuracy**

This shows that even simple models can perform well on sentiment classification tasks.

---

## 💬 Example Predictions

- `"This movie was amazing and unforgettable"` → 👍 Positive  
- `"This was a terrible boring film"` → 👎 Negative  
- `"The movie was okay but too long"` → 👎 Negative  

---

## 🔥 Most Important Words

### Positive words:
`great`, `excellent`, `amazing`, `perfect`, `love`, `wonderful`

### Negative words:
`worst`, `bad`, `boring`, `awful`, `terrible`, `waste`

---

## 🛠️ Tech Stack
- Python 🐍
- Pandas
- NumPy
- Scikit-learn

---

## 📌 Summary
This project demonstrates a full NLP pipeline using classical Machine Learning:
from raw text → vectorization → model training → prediction → evaluation.

---
