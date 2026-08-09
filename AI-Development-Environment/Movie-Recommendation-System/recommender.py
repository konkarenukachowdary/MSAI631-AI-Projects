import pandas as pd
from collections import Counter

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


class MovieRecommender:

    def __init__(self):

        # -----------------------------
        # Load datasets
        # -----------------------------
        self.ratings = pd.read_csv("data/ratings.csv")
        self.movies = pd.read_csv("data/movies.csv")

        print("Ratings loaded:", self.ratings.shape)
        print("Movies loaded:", self.movies.shape)

        # -----------------------------
        # Prepare Surprise dataset
        # -----------------------------
        reader = Reader(rating_scale=(0.5, 5.0))

        data = Dataset.load_from_df(
            self.ratings[
                ["userId", "movieId", "rating"]
            ],
            reader
        )

        # -----------------------------
        # Train/test split
        # -----------------------------
        self.trainset, self.testset = train_test_split(
            data,
            test_size=0.20,
            random_state=42
        )

        # -----------------------------
        # SVD model
        # -----------------------------
        self.model = SVD(
            n_factors=100,
            n_epochs=20,
            random_state=42
        )

        # -----------------------------
        # Train model
        # -----------------------------
        print("\nTraining SVD model...")

        self.model.fit(self.trainset)

        print("Training completed.")

    # =====================================================
    # MODEL EVALUATION
    # =====================================================

    def evaluate(self):

        print("\nEvaluating model...")

        predictions = self.model.test(
            self.testset
        )

        rmse = accuracy.rmse(
            predictions,
            verbose=False
        )

        mae = accuracy.mae(
            predictions,
            verbose=False
        )

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return rmse, mae

    # =====================================================
    # USER PREFERENCE ANALYSIS
    # =====================================================

    def get_user_preferred_genres(
        self,
        user_id,
        minimum_rating=4.0
    ):

        # Get movies rated highly by the user
        user_ratings = self.ratings[
            (self.ratings["userId"] == user_id)
            &
            (self.ratings["rating"] >= minimum_rating)
        ]

        # Add movie information
        user_movies = user_ratings.merge(
            self.movies,
            on="movieId"
        )

        genre_counts = Counter()

        for genres in user_movies["genres"]:

            for genre in genres.split("|"):

                if genre != "(no genres listed)":
                    genre_counts[genre] += 1

        return genre_counts

    # =====================================================
    # GENERATE EXPLANATION
    # =====================================================

    def explain_recommendation(
        self,
        user_id,
        movie_genres
    ):

        preferred_genres = self.get_user_preferred_genres(
            user_id
        )

        recommended_genres = set(
            movie_genres.split("|")
        )

        matching_genres = [
            genre
            for genre in recommended_genres
            if genre in preferred_genres
        ]

        if matching_genres:

            return (
                "Matches genres you have "
                "rated highly: "
                + ", ".join(matching_genres)
            )

        return (
            "Recommended based on "
            "your learned rating preferences"
        )

    # =====================================================
    # TOP-N RECOMMENDATIONS
    # =====================================================

    def recommend_movies(
        self,
        user_id,
        n=10,
        genre=None
    ):

        # Movies already rated by user
        watched_movies = set(
            self.ratings[
                self.ratings["userId"] == user_id
            ]["movieId"]
        )

        # Candidate movies
        candidate_movies = self.movies[
            ~self.movies["movieId"].isin(
                watched_movies
            )
        ].copy()

        # Optional genre filtering
        if genre and genre != "All Genres":
            candidate_movies = candidate_movies[
                candidate_movies["genres"].str.contains(
                    genre,
                    case=False,
                    na=False
                )
            ]

        predictions = []

        # Predict rating for every candidate
        for movie_id in candidate_movies["movieId"]:

            predicted_rating = self.model.predict(
                user_id,
                movie_id
            ).est

            predictions.append(
                (
                    movie_id,
                    predicted_rating
                )
            )

        # Sort highest predicted ratings first
        predictions.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Top N
        top_predictions = predictions[:n]

        recommendations = pd.DataFrame(
            top_predictions,
            columns=[
                "movieId",
                "predicted_rating"
            ]
        )

        # Add movie metadata
        recommendations = recommendations.merge(
            self.movies,
            on="movieId"
        )

        # Add explanations
        recommendations["reason"] = (
            recommendations["genres"]
            .apply(
                lambda genres:
                self.explain_recommendation(
                    user_id,
                    genres
                )
            )
        )

        return recommendations[
            [
                "movieId",
                "title",
                "genres",
                "predicted_rating",
                "reason"
            ]
        ]


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    recommender = MovieRecommender()

    # Evaluate model
    recommender.evaluate()

    # Generate recommendations
    print(
        "\nTop 10 recommendations for User 1:"
    )

    recommendations = (
    recommender.recommend_movies(
        user_id=1,
        n=10,
        genre="Sci-Fi"
    )
)

    print(
        recommendations.to_string(
            index=False
        )
    )