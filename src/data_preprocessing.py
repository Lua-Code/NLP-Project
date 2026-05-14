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


def cleanText(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extractGenres(genresText):
    try:
        genresList = ast.literal_eval(genresText)

        genreNames = []

        for genre in genresList:
            name = genre.get("name", "").strip()

            if name == "Science Fiction":
                name = "Sci-Fi"

            if name:
                genreNames.append(name)

        return ", ".join(genreNames)

    except Exception:
        return ""


def preprocessMoodDataset():
    moodDf = pd.read_csv(moodDatasetPath)

    moodDf.columns = moodDf.columns.str.strip()

    moodDf = moodDf.dropna(subset=["moodText", "genre"])

    moodDf["moodText"] = moodDf["moodText"].apply(cleanText)
    moodDf["genre"] = moodDf["genre"].astype(str).str.strip()

    moodDf = moodDf[moodDf["moodText"] != ""]
    moodDf = moodDf[moodDf["genre"] != ""]

    moodDf.to_csv(cleanedMoodDatasetPath, index=False)

    print("Cleaned mood dataset saved successfully.")
    print("Mood rows:", len(moodDf))
    print("Mood genres:", sorted(moodDf["genre"].unique()))


def preprocessMoviesDataset():
    moviesDf = pd.read_csv(moviesDatasetPath)

    moviesDf.columns = moviesDf.columns.str.strip()

    neededColumns = [
        "title",
        "genres",
        "overview",
        "popularity",
        "vote_average",
        "vote_count"
    ]

    missingColumns = [
        column for column in neededColumns
        if column not in moviesDf.columns
    ]

    if missingColumns:
        raise ValueError(f"Missing columns in TMDB dataset: {missingColumns}")

    moviesDf = moviesDf[neededColumns]

    moviesDf = moviesDf.dropna(subset=["title", "genres"])

    moviesDf["genres"] = moviesDf["genres"].apply(extractGenres)

    moviesDf = moviesDf[moviesDf["genres"] != ""]

    moviesDf["overview"] = moviesDf["overview"].fillna("No overview available.")

    moviesDf["title"] = moviesDf["title"].astype(str).str.strip()
    moviesDf["overview"] = moviesDf["overview"].astype(str).str.strip()

    moviesDf["popularity"] = pd.to_numeric(moviesDf["popularity"], errors="coerce").fillna(0)
    moviesDf["vote_average"] = pd.to_numeric(moviesDf["vote_average"], errors="coerce").fillna(0)
    moviesDf["vote_count"] = pd.to_numeric(moviesDf["vote_count"], errors="coerce").fillna(0)

    moviesDf.to_csv(cleanedMoviesDatasetPath, index=False)

    print("Cleaned movies dataset saved successfully.")
    print("Movie rows:", len(moviesDf))
    print("Sample cleaned genres:")
    print(moviesDf[["title", "genres"]].head())


def main():
    processedDataPath.mkdir(parents=True, exist_ok=True)

    preprocessMoodDataset()
    preprocessMoviesDataset()

    print("Data preprocessing completed.")


if __name__ == "__main__":
    main()