import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score


processedDataPath = Path("data/processed")
modelsPath = Path("models")
resultsPath = Path("results")

datasetPath = processedDataPath / "cleaned_mood_genre_dataset.csv"
modelPath = modelsPath / "mood_genre_model.pkl"
vectorizerPath = modelsPath / "tfidf_vectorizer.pkl"
testDataPath = resultsPath / "test_data.csv"


def loadDataset():
    if not datasetPath.exists():
        raise FileNotFoundError(f"Dataset not found: {datasetPath}")

    dataFrame = pd.read_csv(datasetPath)

    requiredColumns = ["moodText", "genre"]
    missingColumns = [column for column in requiredColumns if column not in dataFrame.columns]

    if missingColumns:
        raise ValueError(f"Missing columns: {missingColumns}")

    dataFrame = dataFrame.dropna(subset=["moodText", "genre"])
    dataFrame["moodText"] = dataFrame["moodText"].astype(str)
    dataFrame["genre"] = dataFrame["genre"].astype(str)

    return dataFrame


def trainModel():
    modelsPath.mkdir(parents=True, exist_ok=True)
    resultsPath.mkdir(parents=True, exist_ok=True)

    dataFrame = loadDataset()

    x = dataFrame["moodText"]
    y = dataFrame["genre"]

    xTrain, xTest, yTrain, yTest = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 3),
        stop_words="english",
        sublinear_tf=True
    )

    xTrainVectorized = vectorizer.fit_transform(xTrain)
    xTestVectorized = vectorizer.transform(xTest)

    baseModel = LinearSVC(
        class_weight="balanced",
        C=1.0,
        max_iter=5000
    )

    model = CalibratedClassifierCV(
        estimator=baseModel,
        cv=3
    )

    model.fit(xTrainVectorized, yTrain)

    predictions = model.predict(xTestVectorized)
    accuracy = accuracy_score(yTest, predictions)

    joblib.dump(model, modelPath)
    joblib.dump(vectorizer, vectorizerPath)

    testDataFrame = pd.DataFrame({
        "moodText": xTest,
        "trueGenre": yTest,
        "predictedGenre": predictions
    })

    testDataFrame.to_csv(testDataPath, index=False)

    print("Linear SVM model trained successfully.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {modelPath}")
    print(f"Vectorizer saved to: {vectorizerPath}")
    print(f"Test data saved to: {testDataPath}")


if __name__ == "__main__":
    trainModel()