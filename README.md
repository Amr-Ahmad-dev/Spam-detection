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

  

  
📊 Dataset Overview

  

The pipeline is designed for the canonical SMS Spam Collection Dataset:


  

  
Format: CSV format (encoded in latin-1).

  
Target (v1): Binary classification label:
  
  

  
ham: Legitimate messages.

  
spam: Unsolicited or fraudulent messages.

  

  
Payload (v2): Raw text message strings.

  

  

  
🧹 Text Preprocessing Workflow

  

Raw text contains noise, irregular casing, punctuation, and linguistic redundancies. The custom clean() routine standardizes every message before vectorization:


  

  
Regex Extraction: Filters exclusively for alphabetic tokens via re.compile(r"[a-z]+"), removing numbers, punctuation, and special symbols.

  
Stopword Elimination: Strips domain-neutral English stopwords (e.g., the, is, at, which) using nltk.corpus.stopwords.

  
Snowball Stemming: Reduces inflected and derived words down to their morphological roots (e.g., "winning", "wins", "winner" 
→
→ "win"), compacting the vocabulary space and reducing model sparsity.

  

  

  
⚙️ Model Architecture

  

The inference engine couples feature extraction and ensemble classification into an immutable Scikit-Learn Pipeline:


  
python
Pipeline([
    ("vec", CountVectorizer()),           
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
])

  

  
CountVectorizer: Converts preprocessed text into a sparse Bag-of-Words matrix representing token occurrence frequencies.

  
RandomForestClassifier:
  
  

  
n_estimators=200: Constructs an ensemble of 200 decorrelated decision trees to eliminate variance and prevent overfitting.

  
stratify=df["v1"]: Guarantees identical ham-to-spam class proportions across training and test splits.

  
n_jobs=-1: Distributes tree building across all available CPU cores for accelerated training.

  

  

  

  
📈 Evaluation & Expected Performance

  

Because spam datasets are naturally imbalanced (typically ~87% ham vs. ~13% spam), accuracy alone is insufficient. This model is evaluated using a full classification report:


  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
Metric	Ham (Legitimate)	Spam (Unsolicited)	Overall Macro / Weighted Avg
Precision	
≈
0.98
≈0.98	
≈
0.99
≈0.99	High protection against false flags
Recall	
≈
1.00
≈1.00	
≈
0.88
−
0.92
≈0.88−0.92	Detects the vast majority of spam
F1-Score	
≈
0.99
≈0.99	
≈
0.93
−
0.95
≈0.93−0.95	Robust harmonic balance
Overall Accuracy	—	—	
≈
97.5
%
−
98.2
%
≈97.5%−98.2%

  

  

Key takeaway: High precision on the spam class ensures legitimate personal or transactional messages (ham) are virtually never misclassified into the junk folder.


  

  

  
📁 Repository Structure

  
text
├── spam_classifier.py         # Main training and evaluation script
├── spam_model.pkl             # Serialized production pipeline artifact (after execution)
├── spam_cleaned.csv           # Exported preprocessed dataset
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

  

  
🚀 Quickstart Guide

  
1. Clone the Repository

  
bash
git clone https://github.com/<your-username>/sms-spam-classifier.git
cd sms-spam-classifier

  
2. Set Up Virtual Environment & Dependencies

  
bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
pip install -r requirements.txt

  
requirements.txt

  
text
pandas>=1.5.0
nltk>=3.8.0
scikit-learn>=1.2.0
joblib>=1.2.0

  
3. Run the Training Pipeline

  

Ensure your dataset path is correctly configured, then execute:


  
bash
python spam_classifier.py

  

  
🔮 Inference & Loading Saved Model

  

Once generated, load spam_model.pkl directly in your application or API without retraining:


  
python
import joblib
# Load saved pipeline
model = joblib.load("spam_model.pkl")
# New incoming SMS samples (raw text pre-cleaned with the same clean() routine)
new_messages = [
    "Hey are we still meeting for lunch today?",
    "URGENT! You have won a 1000 cash prize. Call 09061701461 to claim your reward."
]
# Note: apply clean() to each message first
from spam_classifier import clean
cleaned_samples = [clean(msg) for msg in new_messages]
predictions = model.predict(cleaned_samples)
for msg, label in zip(new_messages, predictions):
    print(f"[{label.upper()}] -> {msg}")
