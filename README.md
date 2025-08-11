 📰 Fake News Detection using NLP & Streamlit

A **machine learning project** that detects whether a given news article is **Fake** or **Real** using **Natural Language Processing (NLP)** and a **Passive Aggressive Classifier**.  
The app is built with **Streamlit** for an interactive and user-friendly interface.

---

 📌 Overview

Fake news has become one of the biggest challenges in the digital era.  
This project aims to classify news articles as **Fake** or **Real** using textual data from the headlines and article content.

We use:
- **TF-IDF Vectorization** for text feature extraction  
- **Passive Aggressive Classifier** for classification  
- **Streamlit** for a live, browser-based UI  

---

 📊 Dataset

We used the **Fake and Real News Dataset** from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

The dataset contains two CSV files:

| File Name   | Size  | Description |
|-------------|-------|-------------|
| `Fake.csv`  | ~59 MB | Contains **12,999** fake news articles |
| `True.csv`  | ~51 MB | Contains **12,999** real news articles |

**Columns in each file:**
- `title` → Headline of the news article
- `text` → Full article content
- `subject` → News category (politics, world, etc.)
- `date` → Date published

We label:
- **Fake news** → `label = 0`
- **Real news** → `label = 1`
