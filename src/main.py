import pandas as pd
import pickle

from src.model.model_functions import load_model, get_movie_id_by_title, predict_rating, update_user_ratings, \
    get_movie_recommendations
from src.ui.app import MovieRecommendationApp

# Załaduj model i dane
# with open('src/model/svd_model.pkl', 'rb') as f:
#     svd = pickle.load(f)
# user_movie_matrix = pd.read_pickle('src/model/user_movie_matrix.pkl')
#
# movies = pd.read_pickle('src/model/movies.pkl')
# Załaduj model i dane
# Załaduj model i dane

# Ścieżki do plików
model_path = 'model/svd_model.pkl'
user_movie_matrix_path = 'model/user_movie_matrix.pkl'
movies_path = 'model/movies.pkl'
user_means_path = 'model/user_means.pkl'

# Ładowanie modelu
model = load_model(model_path)


# Przykładowy ID użytkownika i tytuł filmu
user_movie_matrix = pd.read_pickle(user_movie_matrix_path)
user_means = pd.read_pickle(user_means_path)
movies = pd.read_pickle(movies_path)

# Przykładowy ID użytkownika i tytuł filmu
# user_id = 20230  # Zakładamy, że użytkownik o ID 1000 jest nowy
# movie_title = "Harry Potter and the Half-Blood Prince"  # Zakładamy, że film o takim tytule istnieje
#
# #
# # Movie title: Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001), Movie ID: 4896
# # Movie title: Harry Potter and the Chamber of Secrets (2002), Movie ID: 5816
# # Movie title: Harry Potter and the Prisoner of Azkaban (2004), Movie ID: 8368
# # Movie title: Harry Potter and the Goblet of Fire (2005), Movie ID: 40815
# # Movie title: Harry Potter and the Order of the Phoenix (2007), Movie ID: 54001
# # Movie title: Harry Potter and the Half-Blood Prince (2009), Movie ID: 69844
# # Przykładowe nowe oceny wprowadzone przez użytkownika
# # new_ratings = {
# #     4896: 1,
# #     5816: 2,
# #     8368: 1,
# #     40815: 1,
# #     54001: 1,
# # }  # Przykładowe ID filmów i oceny
# new_ratings = {
#     4896: 4.5,
#     5816: 5,
#     8368: 5.0,
#     40815: 4.0,
#     54001: 4.5,
# }  # Przykładowe ID filmów i oceny
#
# # Aktualizacja ocen użytkownika
# user_movie_matrix, user_movie_matrix_normalized, user_means = update_user_ratings(user_id, new_ratings,
#                                                                                   user_movie_matrix, user_means)
#
# # Przewidywanie oceny
# try:
#     # Uzyskaj ID filmu na podstawie tytułu
#     movie_ids = get_movie_id_by_title(movie_title, movies)
#
#     # Jeśli jest więcej niż jeden wynik, wybierz pierwszy (można to dostosować)
#     movie_id = movie_ids[0]
#
#     # Przewiduj ocenę dla użytkownika dla wybranego filmu
#     predicted_rating = predict_rating(user_id, movie_id, model, user_movie_matrix_normalized, user_means)
#     print(f'Przewidywana ocena użytkownika {user_id} dla filmu "{movie_title}": {predicted_rating}')
#
#     # Pobieranie rekomendacji dla użytkownika
#     recommendations = get_movie_recommendations(user_id, model, user_movie_matrix_normalized, movies, user_means)
#     print(f'Rekomendacje dla użytkownika {user_id}:')
#     print(recommendations)
# except ValueError as e:
#     print(e)
#


if __name__ == '__main__':
    MovieRecommendationApp(model, user_movie_matrix, user_movie_matrix_normalized, movies, user_means).run()
