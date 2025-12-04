# Information Retrieval System - Homework 3 (CS 516)

## Overview
This project implements a **local Information Retrieval (IR) system** using Python. Users can search a collection of text documents (`Articles.csv`) using **TF-IDF vectorization** and **cosine similarity**. The system supports both a **command-line interface (CLI)** and a **Streamlit web interface**.

---

## Features
- Data preprocessing: cleaning, normalization, tokenization, stopword removal
- Indexing: TF-IDF vectorization using scikit-learn, saved to disk
- Retrieval: Cosine similarity scoring
- Interfaces: CLI and Streamlit
- Robust CSV handling with multiple encodings

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <https://github.com/Nafil1516/IR-assignment-.git>
cd IR_ASSIGNMENT

Download NLTK Data

Run the following in Python:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

Run the Project

CLI mode: python src/main.py
Streamlit web app: streamlit run src/app.py(USE THIS ONE)

Project Structure
IR_ASSIGNMENT/
│ Help Files/
│   └─ README.md
│   └─ requirements.txt
├─ data/
│   └─ Articles.csv
│
├─ indexes/                  
│
├─ src/
│   ├─ main.py               # CLI
│   ├─ app.py                # Streamlit UI
│   ├─ preprocessing.py
│   ├─ indexer.py
│   └─ retriever.py
