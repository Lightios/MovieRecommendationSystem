from ui.app import MovieRecommendationApp

from data_loader import *


if __name__ == '__main__':
    MovieRecommendationApp(user_movie_matrix, user_movie_matrix_normalized, movies, user_means, len_tags, len_genres).run()
