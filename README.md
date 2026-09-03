# movie-genreclassification
# 🎬 Movie Genre Classification

Predict the **genre of a movie from its script/dialogue** using Natural Language Processing and Machine Learning.

## 📌 Project Overview

This project classifies movies into different genres based on approximately **1,000 characters of their script**.

The dataset contains **22,579 movie-script samples** with three columns:

* `id` — Movie/sample identifier
* `text` — Movie script/dialogue
* `genre` — Target movie genre

The project performs text cleaning and preprocessing before converting the scripts into numerical features and training a **Multinomial Naive Bayes** classifier.

## 🎭 Genres

The classifier works with 9 genres:

* Other
* Action
* Adventure
* Comedy
* Drama
* Horror
* Romance
* Sci-Fi
* Thriller

## 🔄 Workflow

```text
Movie Scripts
     │
     ▼
Data Loading
     │
     ▼
Data Exploration
     │
     ▼
Genre Encoding
     │
     ▼
Text Cleaning
     │
     ├── Remove special characters
     ├── Convert to lowercase
     ├── Tokenize words
     ├── Remove stopwords
     └── Porter Stemming
     │
     ▼
Text Feature Extraction
     │
     ▼
Train / Test Split
     │
     ▼
Multinomial Naive Bayes
     │
     ▼
Genre Prediction
     │
     ▼
Evaluation
```

The text preprocessing uses regular expressions, English stopwords from NLTK, and the Porter Stemmer.

## 🧠 Machine Learning Model

### Multinomial Naive Bayes

The project uses `MultinomialNB` from Scikit-learn for multi-class text classification.

The processed dataset is split into:

* **Training set:** 18,063 samples
* **Test set:** 4,516 samples
* **Features:** 10,000

## 📊 Results

The Multinomial Naive Bayes classifier achieved:

> **89.57% Test Accuracy**

A confusion matrix is also generated to examine classification performance across the nine genres.

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* WordCloud
* Google Colab

## 📂 Project Structure

```text
Movie-Genre-Classification/
│
├── Movies Genre Classifier.ipynb
├── README.md
└── dataset/
    └── kaggle_movie_train.csv
```

> The notebook currently loads the dataset from a Google Drive path, so the dataset path should be updated if running outside Google Colab.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Movie-Genre-Classification
```

### 2. Install dependencies

```bash
pip install numpy pandas nltk scikit-learn matplotlib seaborn wordcloud
```

### 3. Download the NLTK stopwords

```python
import nltk
nltk.download('stopwords')
```

### 4. Open the notebook

Run:

```text
Movies Genre Classifier.ipynb
```

The notebook was created using a **Python 3 / Google Colab** environment.

## 📚 Dataset

The project uses the **Movie Genre Classification** dataset from Kaggle.

Dataset reference included in the original notebook:

https://www.kaggle.com/c/moviegenres/overview

## 🔮 Future Improvements


