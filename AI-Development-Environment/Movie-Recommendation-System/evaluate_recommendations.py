import pandas as pd
from collections import Counter

from recommender import MovieRecommender


def calculate_genre_diversity(recommendations):
    """
    Calculate the number of unique genres represented
    in the recommendation list.
    """

    genres = set()

    for movie_genres in recommendations["genres"]:
        for genre in movie_genres.split("|"):
            if genre != "(no genres listed)":
                genres.add(genre)

    return len(genres), sorted(genres)


def main():

    print("Loading recommendation system...\n")

    recommender = MovieRecommender()

    # Generate recommendations for User 1
    recommendations = recommender.recommend_movies(
        user_id=1,
        n=10,
        genre="Sci-Fi"
    )

    print("\nCurrent Recommendations:")
    print(
        recommendations[
            [
                "title",
                "genres",
                "predicted_rating"
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------
    # Diversity measurement
    # -----------------------------------------

    diversity_count, genres = calculate_genre_diversity(
        recommendations
    )

    print("\nRecommendation Diversity")
    print("------------------------")
    print("Unique genres represented:", diversity_count)
    print("Genres:", ", ".join(genres))

    # -----------------------------------------
    # Genre frequency
    # -----------------------------------------

    genre_counter = Counter()

    for movie_genres in recommendations["genres"]:

        for genre in movie_genres.split("|"):

            if genre != "(no genres listed)":
                genre_counter[genre] += 1

    print("\nGenre Frequency")
    print("--------------")

    for genre, count in genre_counter.most_common():
        print(f"{genre}: {count}")


if __name__ == "__main__":
    main()