import numpy as np
import pandas as pd


def get_movie_id_by_title(title, movies):
    # Szukaj filmu w DataFrame `movies`
    movie = movies[movies['title'].str.contains(title, case=False, na=False)]

    # Sprawdź, czy znaleziono film
    if movie.empty:
        raise ValueError(f'No movie found with title containing: {title}')

    # Jeśli jest więcej niż jeden wynik, zwróć wszystkie pasujące ID
    movie_ids = movie['movieId'].tolist()
    return movie_ids


def get_movie_title_by_id(movie_id, movies):
    # Szukaj filmu w DataFrame `movies`
    movie = movies[movies['movieId'] == movie_id]

    # Sprawdź, czy znaleziono film
    if movie.empty:
        raise ValueError(f'No movie found with ID: {movie_id}')

    # Zwróć tytuł filmu
    return movie['title'].values[0]


def filter_movies(movies, phrase):
    dataframe = movies[movies['title'].str.contains(str(phrase))]
    return dataframe['title'].tolist(), dataframe['movieId'].tolist()
    # return movies[movies['title'].str.contains(str(phrase))]['title'].tolist()


def update_user_ratings(user_id, new_ratings, user_movie_matrix, user_movie_matrix_normalized, user_means):
    # Do macierzy user_movie_matrix dodajemy srednią ocenę uzytkownika z user_means
    user_movie_matrix = user_movie_matrix.add(user_means, axis=0)

    # dodaj średnią ocen do user_movie_matrix_normalized dla user_id z user_means
    user_movie_matrix_normalized = user_movie_matrix_normalized.add(user_means, axis=0)

    # Dodanie nowych ocen do macierzy
    for movie_id, rating in new_ratings.items():
        user_movie_matrix.loc[user_id, movie_id] = rating
        # dla filmu o id movie_id oblicz średnią ocenę
        movie_mean = user_movie_matrix.loc[:, movie_id].mean()
        # znajdz indeksy wierszy gdzie ocena uzytkownika jest NaN dla filmu o id movie_id
        nan_indexes = user_movie_matrix.loc[user_movie_matrix[movie_id].isnull()].index
        # dla tych indeksów wstaw średnią ocenę filmu do user_movie_matrix_normalized
        user_movie_matrix_normalized.loc[nan_indexes, movie_id] = movie_mean

    # Aktualizacja średniej oceny użytkownika
    user_means[user_id] = user_movie_matrix.loc[user_id].mean()

    # Odejmij średnie ocen z user_means od user_movie_matrix_normalized
    user_movie_matrix_normalized = user_movie_matrix_normalized.sub(user_means, axis=0)

    # Zapisz macierz użytkownik-film i normalizowane średnie
    user_movie_matrix_normalized.to_pickle('src/model/user_movie_matrix_normalized.pkl')
    user_means.to_pickle('src/model/user_means.pkl')
    user_movie_matrix.to_pickle('src/model/user_movie_matrix.pkl')

    return user_movie_matrix, user_movie_matrix_normalized, user_means


def convert_dict_to_dataframe(dictionary):
    return pd.DataFrame(list(dictionary.items()), columns=['movieId', 'rating'])

#  last_movie_ratings - dataframe postaci:
#  last_movie_ratings = pd.DataFrame()
#  last_movie_ratings['movieId'] = [136020, 1, 166461, 2948, 24]
#  last_movie_ratings['rating'] = [4, 5, 5, 4, 4]
#
# zwraca tablice z movieId filmów
def get_predictions(user_id, last_movie_ratings, movies, len_genres, len_tags, user_movie_matrix,
                    user_movie_matrix_normalized, user_means):
    avg_movie = get_average_movie(last_movie_ratings, movies, len_genres, len_tags)
    result = get_closest_movie(avg_movie, user_id, movies, user_movie_matrix, 20)
    return get_svd_prediction(user_id, result, user_movie_matrix, movies, user_movie_matrix_normalized, user_means)


def get_average_movie(last_movie_ratings, movies, len_genres, len_tags):
    if len(last_movie_ratings) == 0:
        return None

    last_movie_ratings['tagId'] = last_movie_ratings['movieId'].apply(
        lambda x: movies.loc[movies['movieId'] == x, 'tagId'].values[0])

    # jeśli średnia ocen jest mniejsza niż 2.5 to zwróć tagi odwrotne do tagów filmów
    if last_movie_ratings['rating'].max() <= 2.5:
        # get sum of tags
        tags_sum = np.zeros(len_genres)
        for tag in last_movie_ratings['tagId']:
            tags_sum = tags_sum + np.array(tag[:len_genres])

        # if tags_sum[i] > 0 then make it 0 and if tags_sum[i] == 0 then make it 1
        tags_sum = [0 if i > 0 else 1 for i in tags_sum]
        return tags_sum + [0] * len_tags
    else:
        # delete films with rating < 2
        last_movie_ratings = last_movie_ratings[last_movie_ratings['rating'] > 2]
        # from each rating subtract 2
        last_movie_ratings['rating'] = last_movie_ratings['rating'] - 2

        ratings_sum = last_movie_ratings['rating'].sum()
        average_tags = np.zeros(len_genres + len_tags)
        for id in last_movie_ratings['movieId']:
            movie_tags = movies.loc[movies['movieId'] == id, 'tagId'].values[0]
            average_tags = average_tags + np.array(movie_tags) * \
                           last_movie_ratings.loc[last_movie_ratings['movieId'] == id, 'rating'].values[0] / ratings_sum
        return average_tags


def get_closest_movie(tags_array, user_id, movies, user_movie_matrix, n=20):
    # copy movieIDs and tags
    movies_copy = movies[['movieId', 'tagId']].copy()

    # delete movies that user has already rated
    movies_copy = movies_copy[~movies_copy['movieId'].isin(user_movie_matrix.loc[user_id].dropna().index)]

    # calculate distance to given average tags
    movies_copy['dist'] = movies_copy['tagId'].apply(lambda x: np.linalg.norm(np.array(x) - np.array(tags_array)))

    # sort by distance
    movies_copy = movies_copy.sort_values('dist')

    # add genres to movies_copy - debug
    # movies_copy['genres'] = movies_copy['tagId'].apply(lambda x: [genres[i] for i in range(len(genres)) if x[i] == 1])

    # get n closest movies
    return movies_copy.head(n)['movieId'].values


def get_svd_prediction(userId, tags_result, user_movie_matrix, movies, user_movie_matrix_normalized, user_means):
    # wykonaj SVD
    u, s, vt = np.linalg.svd(user_movie_matrix_normalized, full_matrices=False)

    # reduce dimensionality to 30 biggest singular values
    k = 30
    indexes = np.argsort(s)[::-1][:k]
    u_k = u[:, indexes]
    s_k = np.diag(s[indexes])
    vt_k = vt[indexes, :]

    # reconstruct matrix
    user_movie_matrix_reconstructed = np.dot(u_k, np.dot(s_k, vt_k))
    # add user means to get original ratings
    user_movie_matrix_reconstructed = user_movie_matrix_reconstructed + user_means.values[:, np.newaxis]
    user_movie_matrix_reconstructed = pd.DataFrame(user_movie_matrix_reconstructed, columns=user_movie_matrix.columns,
                                                   index=user_movie_matrix.index)

    # get predictions for movieIDs from result for user 1
    predictions = user_movie_matrix_reconstructed.loc[userId, tags_result]
    # add predictions to result
    tags_result = pd.DataFrame(tags_result, columns=['movieId'])
    tags_result['prediction'] = predictions.values
    # sort by prediction
    tags_result = tags_result.sort_values('prediction', ascending=False)
    # add titles
    tags_result = tags_result.merge(movies, on='movieId', how='left')
    # get top 4 prediction movieID
    tags_result = tags_result.head(4)['movieId'].values
    return tags_result
