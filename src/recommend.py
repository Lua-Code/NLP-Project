import pandas as pd
from pathlib import Path

processedDataPath = Path("data/processed")
moviesDatasetPath = processedDataPath / "cleaned_movies.csv"


relatedGenres = {
    "Action": ["Action", "Adventure", "Thriller"],
    "Adventure": ["Adventure", "Action", "Fantasy"],
    "Animation": ["Animation", "Family", "Comedy"],
    "Comedy": ["Comedy", "Animation", "Romance"],
    "Drama": ["Drama", "Romance"],
    "Horror": ["Horror", "Thriller"],
    "Mystery": ["Mystery", "Thriller", "Crime"],
    "Romance": ["Romance", "Drama", "Comedy"],
    "Sci-Fi": ["Sci-Fi", "Action", "Adventure"],
    "Thriller": ["Thriller", "Mystery", "Action", "Crime"]
}


def loadMovies():
    if not moviesDatasetPath.exists():
        raise FileNotFoundError(f"Movies dataset not found: {moviesDatasetPath}")

    moviesDf = pd.read_csv(moviesDatasetPath)

    requiredColumns = [
        "title",
        "genres",
        "overview",
        "popularity",
        "vote_average",
        "vote_count"
    ]

    missingColumns = [
        column for column in requiredColumns
        if column not in moviesDf.columns
    ]

    if missingColumns:
        raise ValueError(f"Missing columns in movies dataset: {missingColumns}")

    return moviesDf


def genreMatches(movieGenres, genresToSearch):
    movieGenres = str(movieGenres).lower()

    for genre in genresToSearch:
        if genre.lower() in movieGenres:
            return True

    return False


def recommendMovies(predictedGenre, numberOfMovies=5):
    moviesDf = loadMovies()

    predictedGenreLower = predictedGenre.lower()

    filteredMovies = moviesDf[
        moviesDf["genres"].apply(
            lambda movieGenres: predictedGenreLower in str(movieGenres).lower()
        )
    ].copy()

    filteredMovies = filteredMovies[filteredMovies["vote_count"] >= 100]

    if filteredMovies.empty:
        return filteredMovies

    relatedGenresList = relatedGenres.get(predictedGenre, [predictedGenre])

    def calculateGenreScore(movieGenres):
        movieGenresLower = str(movieGenres).lower()
        score = 0

        for genre in relatedGenresList:
            if genre.lower() in movieGenresLower:
                score += 1

        return score

    filteredMovies["genre_score"] = filteredMovies["genres"].apply(calculateGenreScore)

    recommendedMovies = filteredMovies.sort_values(
        by=["genre_score", "vote_average", "vote_count", "popularity"],
        ascending=[False, False, False, False]
    ).head(numberOfMovies)

    return recommendedMovies[
        ["title", "genres", "overview", "vote_average", "vote_count", "popularity"]
    ]

def recommendMoviesFromGenres(predictedGenres, numberOfMovies=5):
    moviesDf = loadMovies()

    genreNames = []

    for item in predictedGenres:
        if isinstance(item, tuple):
            genreNames.append(item[0])
        else:
            genreNames.append(item)

    genreNamesLower = [genre.lower() for genre in genreNames]

    filteredMovies = moviesDf[moviesDf["genres"].apply(lambda movieGenres: any(genre in str(movieGenres).lower()for genre in genreNamesLower))].copy()

    filteredMovies = filteredMovies[filteredMovies["vote_count"] >= 100]

    if filteredMovies.empty:
        return filteredMovies

    def calculateGenreScore(movieGenres):
        movieGenresLower = str(movieGenres).lower()
        score = 0

        for genre in genreNamesLower:
            if genre in movieGenresLower:
                score += 1

        return score

    filteredMovies["genre_score"] = filteredMovies["genres"].apply(calculateGenreScore)

    recommendedMovies = filteredMovies.sort_values(by=["genre_score", "vote_average", "vote_count", "popularity"],ascending=[False, False, False, False]).head(numberOfMovies)

    return recommendedMovies[
        ["title", "genres", "overview", "vote_average", "vote_count", "popularity"]
    ]

def main():
    predictedGenre = input("Enter predicted genre: ")

    recommendations = recommendMovies(predictedGenre)

    print(f"\nRecommended movies for genre: {predictedGenre}\n")

    for _, movie in recommendations.iterrows():
        print(movie["title"])
        print(f"Genres: {movie['genres']}")
        print(f"Rating: {movie['vote_average']} | Votes: {movie['vote_count']}")
        print(f"Overview: {movie['overview']}")
        print("-" * 60)


if __name__ == "__main__":
    main()