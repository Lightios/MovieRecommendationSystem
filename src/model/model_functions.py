def get_movie_recommendations(user_id, user_movie_matrix, svd, movies, num_recommendations=10):
    user_idx = user_id - 1  # Indeksy w macierzy zaczynają się od 0
    user_ratings = user_movie_matrix.iloc[user_idx].values
    user_svd = svd.transform(user_ratings.reshape(1, -1))
    user_reconstructed = svd.inverse_transform(user_svd).flatten()
    unrated_movies_indices = np.where(user_ratings == 0)[0]
    predicted_ratings = user_reconstructed[unrated_movies_indices]
    top_indices = np.argsort(predicted_ratings)[::-1][:num_recommendations]
    top_movie_ids = user_movie_matrix.columns[unrated_movies_indices[top_indices]]
    return movies[movies['movieId'].isin(top_movie_ids)]


def predict_movie_rating(user_id, movie_title, user_movie_matrix, svd, movies):
    if user_id not in user_movie_matrix.index:
        return None

    user_idx = user_id - 1
    user_ratings = user_movie_matrix.iloc[user_idx].values
    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
    movie_idx = user_movie_matrix.columns.get_loc(movie_id)

    user_svd = svd.transform(user_ratings.reshape(1, -1))
    user_reconstructed = svd.inverse_transform(user_svd).flatten()

    predicted_rating = user_reconstructed[movie_idx]
    return predicted_rating


def filter_movies(movies, phrase):
    return movies[movies['title'].str.contains(str(phrase))]['title'].tolist()

