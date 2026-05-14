import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    xTrainVectorized = vectorizer.fit_transform(x_train)
    yTrainVectorized = vectorizer.transform(y_train)
    
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    
    model.fit(xTrainVectorized, y_train)
    
    predictions = model.predict(vectorizer.transform(x_test))
    accuracy = accuracy_score(y_test, predictions)
    
    joblib.dump(model, modelPath)
    joblib.dump(vectorizer, vectorizerPath)
    
    testDataFrame = pd.DataFrame({"moodText": x_test, "trueGenre": y_test, "predictedGenre": predictions})
    
    testDataFrame.to_csv(testDataPath, index=False)
    
    print("Model trained successfully.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {modelPath}")
    print(f"Vectorizer saved to: {vectorizerPath}")
    print(f"Test data saved to: {testDataPath}")


if __name__ == "__main__":
    trainModel()
    
    
     