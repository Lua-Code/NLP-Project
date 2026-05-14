import pandas as pd
import re
import ast
from pathlib import Path

rawDataPath = Path("data/raw")
processedDataPath = Path("data/processed")

moodDatasetPath = rawDataPath / "mood_genre_dataset.csv"
moviesDatasetPath = rawDataPath / "tmdb_5000_movies.csv"

cleanedMoodDatasetPath = processedDataPath / "cleaned_mood_genre_dataset.csv"
cleanedMoviesDatasetPath = processedDataPath / "cleaned_movies.csv"

#just cleaning up the text data, not doing any feature engineering or anything like that
def cleanText(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extractGenres(genresText):
    try:
        genresList = ast.literal_eval(genresText)
        genreNames = [genre["name"] for genre in genresList]
        genreNames = ["Sci-Fi" if genre == "Science Fiction" else genre for genre in genreNames]
        return ", ".join(genreNames)
    except:
        return ""
    

def preprocessMoodDataset():
    moodDf = pd.read_csv(moodDatasetPath)

    moodDf = moodDf.dropna(subset=["moodText", "genre"])
    moodDf["moodText"] = moodDf["moodText"].apply(cleanText)
    moodDf["genre"] = moodDf["genre"].astype(str).str.strip()

    moodDf = moodDf[moodDf["moodText"] != ""]
    moodDf = moodDf[moodDf["genre"] != ""]

    moodDf.to_csv(cleanedMoodDatasetPath, index=False)

    print("Cleaned mood dataset saved successfully.")
    print("Rows:", len(moodDf))
    print("Genres:", moodDf["genre"].unique())

def preprocessMoviesDataset():
    moviesDf = pd.read_csv(moviesDatasetPath)

    neededColumns = ["title", "genres", "overview", "popularity", "vote_average", "vote_count"]
    moviesDf = moviesDf[neededColumns]

    moviesDf = moviesDf.dropna(subset=["title", "genres"])
    moviesDf["genres"] = moviesDf["genres"].apply(extractGenres)

    moviesDf = moviesDf[moviesDf["genres"] != ""]
    moviesDf["overview"] = moviesDf["overview"].fillna("No overview available.")

    moviesDf["title"] = moviesDf["title"].astype(str).str.strip()
    moviesDf["overview"] = moviesDf["overview"].astype(str).str.strip()

    moviesDf.to_csv(cleanedMoviesDatasetPath, index=False)

    print("Cleaned movies dataset saved successfully.")
    print("Rows:", len(moviesDf))

def main():
    processedDataPath.mkdir(parents=True, exist_ok=True)

    preprocessMoodDataset()
    preprocessMoviesDataset()


if __name__ == "__main__":
    main()


