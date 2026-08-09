# AI Movie Recommendation System

## Project Overview

The AI Movie Recommendation System is an interactive movie recommendation application developed using artificial intelligence and machine learning techniques.

The system uses collaborative filtering with Singular Value Decomposition (SVD) to learn patterns from user-movie ratings and generate personalized movie recommendations.

The project was developed using the MovieLens dataset and provides an interactive Gradio-based graphical user interface.

The system was inspired by the SVD-based movie recommendation approach provided in the course learning materials. The implementation was extended with genre filtering, recommendation explanations, and a user-friendly interactive interface.

---

## Objectives

The main objectives of the project are to:

- Implement an AI-based movie recommendation system.
- Apply collaborative filtering using SVD.
- Train and evaluate a recommendation model using user-rating data.
- Generate personalized Top-N movie recommendations.
- Allow users to filter recommendations by genre.
- Provide human-readable explanations for recommendations.
- Develop an interactive graphical user interface.
- Evaluate the recommendation model using RMSE and MAE.
- Demonstrate how an existing recommendation approach can be extended.

---

## Recommendation Approach

The system uses collaborative filtering based on Singular Value Decomposition (SVD).

The MovieLens ratings dataset contains interactions between users and movies. The SVD algorithm learns latent patterns from these interactions and predicts ratings for movies that a user has not previously rated.

The recommendation process is:

1. Load MovieLens rating data.
2. Prepare the user-movie-rating dataset.
3. Split the data into training and testing sets.
4. Train an SVD collaborative filtering model.
5. Predict ratings for movies not previously rated by the selected user.
6. Rank movies according to predicted ratings.
7. Apply optional genre filtering.
8. Generate explanations based on the user's highly rated genres.
9. Display recommendations through the Gradio interface.

---

## Dataset

The project uses the MovieLens Latest Small dataset provided by GroupLens.

The dataset contains user ratings and movie metadata.

The primary files used by this project are:

- `ratings.csv`
- `movies.csv`
- `tags.csv`

The ratings data contains:

- User ID
- Movie ID
- Rating
- Timestamp

The movie metadata contains:

- Movie ID
- Movie title
- Genres

The dataset contains 100,836 ratings and 610 users. The movie catalog contains 9,742 movies, while 9,724 movies appear in the ratings data.

---

## Technologies Used

| Technology      | Purpose                             |
| --------------- | ----------------------------------- |
| Python          | Application development             |
| Pandas          | Data processing                     |
| NumPy           | Numerical computation               |
| SciPy           | Scientific computing                |
| Scikit-learn    | Machine learning utilities          |
| Scikit-Surprise | SVD recommendation model            |
| Gradio          | Graphical user interface            |
| MovieLens       | Recommendation dataset              |
| Git/GitHub      | Version control and project sharing |

---

## System Architecture

```text
                 MovieLens Dataset
                        |
                        v
                 Data Preparation
                        |
                        v
              User-Movie Rating Data
                        |
                        v
              SVD Collaborative Model
                        |
                        v
                Predicted Ratings
                        |
                +-------+-------+
                |               |
                v               v
        Genre Filtering   User Preferences
                |               |
                +-------+-------+
                        |
                        v
                 Top-N Ranking
                        |
                        v
             Recommendation Explanation
                        |
                        v
                  Gradio GUI
```
