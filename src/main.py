import pandas as pd
import pickle

from src.model.model_functions import load_model, get_movie_id_by_title, predict_rating
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
user_id = 1
movie_title = "Toy Story"  # Zakładamy, że film o takim tytule istnieje

try:
    # Uzyskaj ID filmu na podstawie tytułu
    movie_ids = get_movie_id_by_title(movie_title, movies)

    # Jeśli jest więcej niż jeden wynik, wybierz pierwszy (można to dostosować)
    movie_id = movie_ids[0]

    # Przewiduj ocenę dla użytkownika dla wybranego filmu
    predicted_rating = predict_rating(user_id, movie_id, model, user_movie_matrix, user_means)
    print(f'Przewidywana ocena użytkownika {user_id} dla filmu "{movie_title}": {predicted_rating}')
except ValueError as e:
    print(e)



if __name__ == '__main__':
    MovieRecommendationApp(model, user_movie_matrix, movies, user_means).run()
