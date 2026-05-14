import pandas as pd
import joblib
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


modelsPath = Path("models")
resultsPath = Path("results")

modelPath = modelsPath / "mood_genre_model.pkl"
vectorizerPath = modelsPath / "tfidf_vectorizer.pkl"
testDataPath = resultsPath / "test_data.csv"

classificationReportPath = resultsPath / "classification_report.txt"
metricsPath = resultsPath / "metrics.txt"
confusionMatrixPath = resultsPath / "confusion_matrix.png"


def loadFiles():
    if not modelPath.exists():
        raise FileNotFoundError(f"Model not found: {modelPath}")

    if not vectorizerPath.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizerPath}")

    if not testDataPath.exists():
        raise FileNotFoundError(f"Test data not found: {testDataPath}")

    model = joblib.load(modelPath)
    vectorizer = joblib.load(vectorizerPath)
    testDataFrame = pd.read_csv(testDataPath)

    return model, vectorizer, testDataFrame


def evaluateModel():
    resultsPath.mkdir(parents=True, exist_ok=True)

    model, vectorizer, testDataFrame = loadFiles()

    xTest = testDataFrame["moodText"].astype(str)
    yTest = testDataFrame["trueGenre"].astype(str)

    xTestVectorized = vectorizer.transform(xTest)
    predictions = model.predict(xTestVectorized)

    accuracy = accuracy_score(yTest, predictions)

    report = classification_report(yTest, predictions)

    with open(classificationReportPath, "w", encoding="utf-8") as file:
        file.write(report)

    with open(metricsPath, "w", encoding="utf-8") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")

    labels = sorted(yTest.unique())

    matrix = confusion_matrix(yTest, predictions, labels=labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    display.plot(ax=ax, xticks_rotation=45)
    plt.title("Mood Genre Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusionMatrixPath)
    plt.close()

    print("Model evaluation completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification report saved to: {classificationReportPath}")
    print(f"Metrics saved to: {metricsPath}")
    print(f"Confusion matrix saved to: {confusionMatrixPath}")


if __name__ == "__main__":
    evaluateModel()