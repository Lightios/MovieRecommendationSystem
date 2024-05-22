import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


import numpy as np
import pandas as pd


def get_movie_recommendations(user_id, model, user_movie_matrix, movies, n_recommendations=10):
    # Sprawdzenie, czy użytkownik istnieje w macierzy
    if user_id not in user_movie_matrix.index:
        raise ValueError(f'User ID {user_id} not found in the user-movie matrix.')

    # Ekstrakcja danych użytkownika
    user_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)

    # Transformacja danych użytkownika
    user_transformed = model.named_steps['svd'].transform(user_ratings)

    # Rekonstrukcja ocen użytkownika
    user_reconstructed = model.named_steps['svd'].inverse_transform(user_transformed)

    # Różnica między rekonstrukcją a rzeczywistymi ocenami
    diff = user_reconstructed - user_ratings

    # Tworzenie ramki danych z różnicami
    diff_df = pd.DataFrame(diff.flatten(), index=user_movie_matrix.columns, columns=['diff'])

    # Filtrowanie filmów, które użytkownik nie ocenił
    unseen_movies = diff_df[user_movie_matrix.loc[user_id] == 0]

    # Sortowanie filmów według różnicy w ocenach (im wyższa różnica, tym wyższa rekomendacja)
    recommendations = unseen_movies.sort_values(by='diff', ascending=False).head(n_recommendations)

    # Łączenie rekomendacji z tytułami filmów
    recommended_movie_ids = recommendations.index
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]

    return recommended_movies


def get_movie_id_by_title(title, movies):
    # Szukaj filmu w DataFrame `movies`
    movie = movies[movies['title'].str.contains(title, case=False, na=False)]

    # Sprawdź, czy znaleziono film
    if movie.empty:
        raise ValueError(f'No movie found with title containing: {title}')

    # Jeśli jest więcej niż jeden wynik, zwróć wszystkie pasujące ID
    movie_ids = movie['movieId'].tolist()
    return movie_ids


def predict_rating(user_id, movie_id, model, user_movie_matrix, user_means):
    # Sprawdzenie, czy użytkownik istnieje w macierzy
    if user_id not in user_movie_matrix.index:
        raise ValueError(f'User ID {user_id} not found in the user-movie matrix.')

    # Sprawdzenie, czy film istnieje w macierzy
    if movie_id not in user_movie_matrix.columns:
        raise ValueError(f'Movie ID {movie_id} not found in the user-movie matrix.')

    # Ekstrakcja danych użytkownika
    user_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)

    # Transformacja danych użytkownika
    user_transformed = model.named_steps['svd'].transform(user_ratings)

    # Rekonstrukcja ocen użytkownika
    user_reconstructed = model.named_steps['svd'].inverse_transform(user_transformed)

    # Dodanie średniej oceny użytkownika
    predicted_rating = user_reconstructed[0, user_movie_matrix.columns.get_loc(movie_id)] + user_means[user_id]

    return predicted_rating


def filter_movies(movies, phrase):
    return movies[movies['title'].str.contains(str(phrase))]['title'].tolist()
