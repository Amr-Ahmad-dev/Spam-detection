# 📬 End-to-End SMS Spam Classifier with NLP & Random Forest

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-green.svg)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Random Forest](https://img.shields.io/badge/Model-Random%20Forest%20(200%20Estimators)-blue.svg)](#)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) and Machine Learning pipeline that detects and filters unsolicited SMS messages (spam) with high precision and recall. 

The project features a custom tokenization and stemming engine, automated text preprocessing, an end-to-end `scikit-learn` Pipeline, stratified evaluation, and serialized model persistence using `joblib`.

---

## 📌 Table of Contents
- [Architecture & Pipeline](#-architecture--pipeline)
- [Dataset Overview](#-dataset-overview)
- [Text Preprocessing Workflow](#-text-preprocessing-workflow)
- [Model Architecture](#-model-architecture)
- [Evaluation & Expected Performance](#-evaluation--expected-performance)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Inference & Loading Saved Model](#-inference--loading-saved-model)
- [Future Enhancements](#-future-enhancements)

---

## 🏗️ Architecture & Pipeline

```text
Raw SMS Text (v2)
       │
       ▼
[Text Cleaning & Normalization]
  ├─ Lowercase Conversion
  ├─ Regex Alpha Tokenization ([a-z]+)
  ├─ NLTK English Stopword Removal
  └─ Snowball Stemming (morphological reduction)
       │
       ▼
[Cleaned Corpus Generation] ──► Exports to `spam_cleaned.csv`
       │
       ▼
[Stratified 80/20 Train-Test Split]
       │
       ▼
[Scikit-Learn Unified Pipeline]
  ├─ Step 1: CountVectorizer() [Bag of Words feature extraction]
  └─ Step 2: RandomForestClassifier(n_estimators=200, n_jobs=-1)
       │
       ▼
[Evaluation & Diagnostic Auditing]
  ├─ Accuracy Score
  └─ Classification Report (Precision, Recall, F1)
       │
       ▼
[Model Serialization] ──► Exports production artifact: `spam_model.pkl`
