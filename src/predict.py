import re
import joblib
from pathlib import Path

modelsPath = Path("models")

modelPath = modelsPath / "mood_genre_model.pkl"
vectorizerPath = modelsPath / "tfidf_vectorizer.pkl"


def cleanText(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def loadModel():
    if not modelPath.exists():
        raise FileNotFoundError(f"Model not found: {modelPath}")

    if not vectorizerPath.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizerPath}")

    model = joblib.load(modelPath)
    vectorizer = joblib.load(vectorizerPath)

    return model, vectorizer


def predictGenres(moodText, topN=3):
    model, vectorizer = loadModel()

    cleanedText = cleanText(moodText)
    vectorizedText = vectorizer.transform([cleanedText])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorizedText)[0]
        classes = model.classes_

        topResults = sorted(zip(classes, probabilities),key=lambda item: item[1],reverse=True)[:topN]

        return topResults

    predictedGenre = model.predict(vectorizedText)[0]
    return [(predictedGenre, 1.0)]


def main():
    moodText = input("Enter your mood or movie preference: ")

    predictedGenres = predictGenres(moodText)

    print("Top predicted genres:")
    for genre, probability in predictedGenres:
        print(f"{genre}: {probability:.2f}")


if __name__ == "__main__":
    main()