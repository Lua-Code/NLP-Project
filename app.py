import streamlit as st

from src.predict import predictGenres
from src.recommend import recommendMoviesFromGenres


st.set_page_config(
    page_title="Movie Mood Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation from User Mood")
st.write(
    "Enter your mood or movie preference in natural language, "
    "and the NLP model will predict a suitable movie genre."
)

moodText = st.text_area(
    "How are you feeling? What type of movie do you want?",
    placeholder="Example: I want something dark and mysterious"
)

numberOfMovies = st.slider(
    "Number of movie recommendations",
    min_value=3,
    max_value=10,
    value=5
)

if st.button("Recommend Movies"):
    if moodText.strip() == "":
        st.warning("Please enter a mood or movie preference first.")
    else:
        predictedGenres = predictGenres(moodText, topN=3)

        st.subheader("Top Predicted Genres")

        for genre, probability in predictedGenres:
            st.write(f"**{genre}** — {probability:.2f}")

        mainGenre = predictedGenres[0][0]
        st.success(f"Main predicted genre: {mainGenre}")

        recommendations = recommendMoviesFromGenres(
            predictedGenres,
            numberOfMovies=numberOfMovies
        )

        st.subheader("Recommended Movies")

        if recommendations.empty:
            st.error("No movies found for this genre.")
        else:
            for _, movie in recommendations.iterrows():
                with st.container():
                    st.markdown(f"### {movie['title']}")
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(f"**Rating:** {movie['vote_average']} / 10")
                    st.write(f"**Votes:** {int(movie['vote_count'])}")
                    st.write(f"**Popularity:** {movie['popularity']}")
                    st.write(movie["overview"])
                    st.divider()