import pandas as pd

user_movie_matrix_path = 'src/model/user_movie_matrix.pkl'
user_movie_matrix_normalized_path = 'src/model/user_movie_matrix_normalized.pkl'
movies_path = 'src/model/movies.pkl'
user_means_path = 'src/model/user_means.pkl'
len_tags_genres_path = 'src/model/len_tags_genres.pkl'
movie_means_path = 'src/model/movie_means.pkl'

user_movie_matrix = pd.read_pickle(user_movie_matrix_path)
user_movie_matrix_normalized = pd.read_pickle(user_movie_matrix_normalized_path)
user_means = pd.read_pickle(user_means_path)
movies = pd.read_pickle(movies_path)
lengths = pd.read_pickle(len_tags_genres_path)
len_tags = lengths[0].values[0]
len_genres = lengths[0].values[1]