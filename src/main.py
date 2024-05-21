import pandas as pd
import pickle

from src.ui.app import MovieRecommendationApp

# Załaduj model i dane
with open('src/model/svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)
user_movie_matrix = pd.read_pickle('src/model/user_movie_matrix.pkl')
movies = pd.read_pickle('src/model/movies.pkl')


if __name__ == '__main__':
    MovieRecommendationApp(svd, user_movie_matrix, movies).run()
