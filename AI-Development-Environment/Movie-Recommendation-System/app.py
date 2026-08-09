import gradio as gr
import pandas as pd

from recommender import MovieRecommender


# -----------------------------------------
# Load recommendation system
# -----------------------------------------

print("Loading recommendation system...")

recommender = MovieRecommender()

print("Recommendation system ready.")


# -----------------------------------------
# Load available users and genres
# -----------------------------------------

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

users = sorted(
    ratings["userId"].unique().tolist()
)

genres = set()

for movie_genres in movies["genres"]:

    for genre in movie_genres.split("|"):

        if genre != "(no genres listed)":
            genres.add(genre)

genre_choices = ["All Genres"] + sorted(genres)


# -----------------------------------------
# Recommendation function
# -----------------------------------------

def get_recommendations(
    user_id,
    genre,
    number_of_recommendations
):

    try:

        user_id = int(user_id)

        number_of_recommendations = int(
            number_of_recommendations
        )

        recommendations = (
            recommender.recommend_movies(
                user_id=user_id,
                n=number_of_recommendations,
                genre=genre
            )
        )

        if recommendations.empty:

            return "No recommendations found."

        # Format results
        output = []

        for index, row in recommendations.iterrows():

            movie_title = row["title"]
            movie_genres = row["genres"]
            predicted_rating = row[
                "predicted_rating"
            ]
            reason = row["reason"]

            output.append(
                f"""
### {index + 1}. {movie_title}

**Genres:** {movie_genres}

**Predicted Rating:** ⭐ {predicted_rating:.2f}

**Why this was recommended:** {reason}
"""
            )

        return "\n---\n".join(output)

    except Exception as error:

        return (
            f"Unable to generate recommendations: "
            f"{error}"
        )


# -----------------------------------------
# Build Gradio interface
# -----------------------------------------

with gr.Blocks(
    title="AI Movie Recommendation System"
) as demo:

    gr.Markdown(
        """
# 🎬 AI Movie Recommendation System

Discover movies personalized to your rating history
using **SVD-based collaborative filtering**.

The system can also filter recommendations by genre
and provide an explanation for why each movie was
recommended.
"""
    )

    with gr.Row():

        user_dropdown = gr.Dropdown(
            choices=users,
            label="Select User",
            value=users[0]
        )

        genre_dropdown = gr.Dropdown(
            choices=genre_choices,
            label="Preferred Genre",
            value="All Genres"
        )

        recommendation_count = gr.Slider(
            minimum=1,
            maximum=20,
            value=10,
            step=1,
            label="Number of Recommendations"
        )

    recommend_button = gr.Button(
        "🎥 Get Recommendations"
    )

    results = gr.Markdown(
        label="Recommendations"
    )

    recommend_button.click(
        fn=get_recommendations,
        inputs=[
            user_dropdown,
            genre_dropdown,
            recommendation_count
        ],
        outputs=results
    )

    gr.Markdown(
        """
### About the system

Recommendations are generated using an SVD
collaborative-filtering model trained on MovieLens
user-rating data.

The predicted rating is a model estimate and should
not be interpreted as a guaranteed rating.
"""
    )


# -----------------------------------------
# Launch application
# -----------------------------------------

if __name__ == "__main__":

    demo.launch()