# Movie Recommendation from User Mood Using NLP Text Classification

This project is an NLP-based movie recommendation system. The user enters a mood or movie preference in natural language, and the system predicts suitable movie genres using NLP text classification. The predicted genres are then used to recommend movies from the TMDB 5000 Movies dataset.

Example inputs:

```text
I want something dark and mysterious
I want to cry
I want explosions and fights
I want something funny and romantic
I want space and robots
```

---

## Project Goal

The goal is to build a complete NLP pipeline that can understand user-written mood text and map it to movie genres.

Main NLP task:

```text
Natural language mood/preference sentence -> Movie genre prediction
```

Full system flow:

```text
User mood text
-> Text preprocessing
-> TF-IDF vectorization
-> Linear SVM genre classifier
-> Top predicted genres
-> Movie filtering and ranking
-> Movie recommendations
```

---

## Features

- Cleans natural language mood text.
- Uses TF-IDF vectorization for NLP feature extraction.
- Uses Linear SVM for text classification.
- Uses calibrated probabilities to return the top predicted genres.
- Parses TMDB movie genres from JSON-like strings.
- Recommends movies that match the predicted genres.
- Displays movie title, genres, rating, vote count, popularity, and overview.
- Evaluates the model using accuracy, precision, recall, F1-score, and a confusion matrix.

---

## NLP Techniques Used

### 1. Text Preprocessing

The text is cleaned before training and prediction:

- Convert text to lowercase.
- Remove punctuation.
- Remove non-letter characters.
- Remove extra spaces.
- Remove missing or empty rows.

Example:

```text
"I want something dark and mysterious!"
-> "i want something dark and mysterious"
```

### 2. TF-IDF Vectorization

TF-IDF converts text into numerical features that the machine learning model can understand.

Example:

```text
i want to see explosions
```

Important feature:

```text
explosions
```

Expected genre:

```text
Action
```

### 3. Text Classification

A Linear SVM classifier learns the relationship between TF-IDF features and movie genres.

Example:

```text
i want to cry -> Drama
make me laugh -> Comedy
space robots and aliens -> Sci-Fi
detective clues and secrets -> Mystery
```

### 4. Top-N Genre Prediction

Instead of predicting only one genre, the system returns the top predicted genres using calibrated probability scores.

Example:

```text
Input: I want something funny and romantic

Output:
Comedy: 0.52
Romance: 0.31
Drama: 0.08
```

---

## Project Structure

```text
NLP-Project/
|
├── data/
│   ├── raw/
│   │   ├── mood_genre_dataset.csv
│   │   └── tmdb_5000_movies.csv
│   └── processed/
│       ├── cleaned_mood_genre_dataset.csv
│       └── cleaned_movies.csv
|
├── models/
│   ├── mood_genre_model.pkl
│   └── tfidf_vectorizer.pkl
|
├── results/
│   ├── test_data.csv
│   ├── classification_report.txt
│   ├── metrics.txt
│   └── confusion_matrix.png
|
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── recommend.py
|
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset Files

The project needs two datasets.

### 1. Mood-to-Genre Dataset

Path:

```text
data/raw/mood_genre_dataset.csv
```

Expected columns:

```text
moodText,genre
```

Example:

```csv
moodText,genre
i want to cry,Drama
i want to see explosions,Action
i want something funny,Comedy
i want space and robots,Sci-Fi
i want to solve a mystery,Mystery
```

### 2. TMDB 5000 Movies Dataset

Path:

```text
data/raw/tmdb_5000_movies.csv
```

Important columns:

```text
title
genres
overview
popularity
vote_average
vote_count
```

Original TMDB genre format:

```text
[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 878, "name": "Science Fiction"}]
```

After preprocessing:

```text
Action, Adventure, Sci-Fi
```

---

## Requirements

```text
pandas
numpy
scikit-learn
matplotlib
streamlit
joblib
nltk
```

Install:

```powershell
pip install -r requirements.txt
```

If there is a disk/cache issue:

```powershell
pip install --no-cache-dir -r requirements.txt
```

---

## Environment Setup

Open PowerShell or CMD inside the main project folder:

```powershell
cd D:\Github\NLP-Project
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it in PowerShell:

```powershell
.\venv\Scripts\activate
```

Activate it in CMD:

```cmd
venv\Scripts\activate
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
```

Install requirements:

```powershell
pip install -r requirements.txt
```

---

## How to Run the Project

Run all commands from the main project folder.

### Step 1: Preprocess the datasets

```powershell
python src/data_preprocessing.py
```

Expected outputs:

```text
data/processed/cleaned_mood_genre_dataset.csv
data/processed/cleaned_movies.csv
```

### Step 2: Train the NLP model

```powershell
python src/train_model.py
```

Expected outputs:

```text
models/mood_genre_model.pkl
models/tfidf_vectorizer.pkl
results/test_data.csv
```

### Step 3: Evaluate the model

```powershell
python src/evaluate_model.py
```

Expected outputs:

```text
results/classification_report.txt
results/metrics.txt
results/confusion_matrix.png
```

### Step 4: Test prediction in terminal

```powershell
python src/predict.py
```

Example input:

```text
i want to cry
```

Example output:

```text
Top predicted genres:
Drama: 0.62
Romance: 0.18
Comedy: 0.08
```

### Step 5: Test recommendation in terminal

```powershell
python src/recommend.py
```

Example input:

```text
Action
```

### Step 6: Run the Streamlit app

```powershell
python -m streamlit run app.py
```

The app should open at a local address similar to:

```text
http://localhost:8501
```

---

## Correct Running Order

For a fresh run:

```powershell
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py
python -m streamlit run app.py
```

If you change the mood dataset, rerun:

```powershell
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py
```

Then restart the Streamlit app.

---

## What Each File Does

| File | Purpose |
|---|---|
| `src/data_preprocessing.py` | Cleans the mood dataset and TMDB movie dataset. |
| `src/train_model.py` | Trains the TF-IDF + Linear SVM NLP classifier. |
| `src/evaluate_model.py` | Evaluates the model and saves metrics/confusion matrix. |
| `src/predict.py` | Loads the model and predicts top genres from user mood text. |
| `src/recommend.py` | Recommends movies based on predicted genres. |
| `app.py` | Streamlit web interface for the final demo. |
| `requirements.txt` | Lists required Python packages. |

---

## Model Files

After training, the project creates two `.pkl` files:

```text
models/mood_genre_model.pkl
models/tfidf_vectorizer.pkl
```

`mood_genre_model.pkl` stores the trained Linear SVM genre classification model.

`tfidf_vectorizer.pkl` stores the trained TF-IDF vectorizer. It remembers the vocabulary and feature mapping used during training.

These files are saved so the app does not retrain the model every time it runs.

Prediction flow:

```text
User text
-> Load tfidf_vectorizer.pkl
-> Convert text into TF-IDF features
-> Load mood_genre_model.pkl
-> Predict top genres
```

---

## Recommendation Logic

After the NLP model predicts genres, the recommendation system:

1. Loads `cleaned_movies.csv`.
2. Filters movies that contain the predicted genre.
3. Removes movies with low vote counts.
4. Scores movies based on genre matches.
5. Sorts movies by genre score, vote average, vote count, and popularity.
6. Returns the top recommendations.

Important rule:

```text
Recommended movies should include at least one predicted genre.
```

---

## Example Test Inputs

```text
i want to cry
i want to see explosions
i want something funny and romantic
i want space and robots
i want something dark and mysterious
i want to be scared
make me laugh
detective clues and secrets
i want a sweet love story
i want a big adventure
```







