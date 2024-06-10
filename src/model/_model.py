import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import pickle

# Wczytanie danych
ratings = pd.read_csv('src/ml-latest-small/ratings.csv')
movies = pd.read_csv('src/ml-latest-small/movies.csv')
tags = pd.read_csv('src/ml-latest-small/tags.csv')

# Usunięcie zbędnych danych - tagów które pojawiają sie tylko raz, oraz kolumn z timestampami i userId
tags['tag'] = tags['tag'].str.lower()
tags = tags.groupby('tag').filter(lambda x: len(x) > 1)
tags = tags.drop(columns=['userId', 'timestamp'])

# Przyporządkowanie każdemu tagowi unikalnego id
tags['tagId'] = tags['tag'].astype('category').cat.codes

# Zrobienie listy wszystkich tagów i posortowanie po id
all_tags = tags['tagId'].unique()
all_tags = np.sort(all_tags)
print (all_tags) #ewntualnie len(all_tags) 

# Do każdego filmu dodaj listę rozmiaru all_tags z wartościami od 0 do 1 w zależności od tego czy film ma dany tag
movie_tags = tags.groupby('movieId')['tagId'].apply(list).reset_index()
movie_tags['tagId'] = movie_tags['tagId'].apply(lambda x: [2 if i in x else 0 for i in all_tags])

# Połącz dane filmów z tagami - jesli film nie ma tagów to wstaw listę zer
movies = movies.merge(movie_tags, on='movieId', how='left')
movies['tagId'] = movies['tagId'].fillna(movies['tagId'].apply(lambda x: [0]*len(all_tags)))

#Zrób listę gatunków
genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']

print(genres)

#Rozdziel movies['genres'] - podziałką jest '|'
movies['genres'] = movies['genres'].str.split('|')

#zamien movies['genres'] na listę 0 i 1 w zależności od tego czy film ma dany gatunek
movies['genres_temp'] = movies['genres'].apply(lambda x: [1 if i in x else 0 for i in genres])

#dodaj listę z gatunkami na początku listy tagID
movies['tagId'] = movies['genres_temp'] + movies['tagId']
movies.drop(columns=['genres_temp', 'genres'], inplace=True)

print(movies)

# get movieID which tagID is closest to given tagID
def get_closest_movie(tags_array, n = 20):
    #copy movies
    movies_copy = movies.copy()
    
    #calculate distance to given tag_temp - make them arraylike
    movies_copy['dist'] = movies_copy['tagId'].apply(lambda x: np.linalg.norm(np.array(x) - np.array(tags_array)))

    #sort by distance
    movies_copy = movies_copy.sort_values('dist')

    # add genres to movies_copy
    movies_copy['genres'] = movies_copy['tagId'].apply(lambda x: [genres[i] for i in range(len(genres)) if x[i] == 1])

    #get n closest movies
    return movies_copy.head(n)['movieId'].values


#  my_movies - dataframe postaci:
#  my_movies = pd.DataFrame()
#  my_movies['movieId'] = [136020, 1, 166461, 2948, 24]
#  my_movies['rating'] = [4, 5, 5, 4, 4]
#
# zwraca tablice z tagami uśrednionymi na podstawie ocen filmów
def get_average_movie(my_movies):
    if len(my_movies) == 0:
        return None
    
    my_movies['tagId'] = my_movies['movieId'].apply(lambda x: movies.loc[movies['movieId'] == x, 'tagId'].values[0])

    # jeśli średnia ocen jest mniejsza niż 2.5 to zwróć tagi odwrotne do tagów filmów
    if my_movies['rating'].max() <= 2.5:
        #get sum of tags
        tags_sum = np.zeros(len(genres))
        for tag in my_movies['tagId']:
            tags_sum = tags_sum + np.array(tag[:len(genres)])
        
        #if tags_sum[i] > 0 then make it 0 and if tags_sum[i] == 0 then make it 1
        tags_sum = [0 if i > 0 else 1 for i in tags_sum]
        return tags_sum + [0]*len(all_tags)
    else:
        #delete films with rating < 2
        my_movies = my_movies[my_movies['rating'] > 2]
        #from each rating subtract 2
        my_movies['rating'] = my_movies['rating'] - 2

        ratings_sum = my_movies['rating'].sum()
        average_tags = np.zeros(len(genres) + len(all_tags))
        for id in my_movies['movieId']:
            movie_tags = movies.loc[movies['movieId'] == id, 'tagId'].values[0]
            average_tags = average_tags + np.array(movie_tags) * my_movies.loc[my_movies['movieId'] == id, 'rating'].values[0] / ratings_sum
        return average_tags


my_movies = pd.DataFrame()
my_movies['movieId'] = [167746, 167746, 167746,167746, 167746]
my_movies['rating'] = [4, 5, 5, 4, 4]

print (my_movies)
#print(get_average_movie(my_movies))
avg_movie = get_average_movie(my_movies)
result = get_closest_movie(avg_movie, 50)



# Tworzenie macierzy użytkownik-film
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
print(user_movie_matrix)

###
def get_svd_prediction(userId, tags_result):
    # Normalizacja ocen - zamiast NaN wstawmy średnią ocenę filmu
    user_movie_matrix_normalized = user_movie_matrix.fillna(user_movie_matrix.mean(axis=0)).fillna(0)
    # Normalizacja ocen - odejmowanie średniej oceny użytkownika
    user_means = user_movie_matrix_normalized.mean(axis=1)
    user_movie_matrix_normalized = user_movie_matrix_normalized.sub(user_means, axis=0).fillna(0)

    #wykonaj SVD
    u, s, vt = np.linalg.svd(user_movie_matrix_normalized, full_matrices=False)

    #reduce dimensionality to 30 biggest singular values
    k = 30
    indexes = np.argsort(s)[::-1][:k]
    u_k = u[:, indexes]
    s_k = np.diag(s[indexes])
    vt_k = vt[indexes, :]

    #reconstruct matrix
    user_movie_matrix_reconstructed = np.dot(u_k, np.dot(s_k, vt_k))
    #add user means to get original ratings
    user_movie_matrix_reconstructed = user_movie_matrix_reconstructed + user_means.values[:, np.newaxis]
    user_movie_matrix_reconstructed = pd.DataFrame(user_movie_matrix_reconstructed, columns=user_movie_matrix.columns, index=user_movie_matrix.index)


    #get predictions for movieIDs from result for user 1
    predictions = user_movie_matrix_reconstructed.loc[userId, tags_result]
    #add predictions to result
    tags_result = pd.DataFrame(tags_result, columns=['movieId'])
    tags_result['prediction'] = predictions.values
    #sort by prediction
    tags_result = tags_result.sort_values('prediction', ascending=False)
    #add titles
    tags_result = tags_result.merge(movies, on='movieId', how='left')
    #get top 4 prediction movieID
    tags_result = tags_result.head(4)['movieId'].values
    return tags_result
    ###

print (get_svd_prediction(1, result))















#compare predictions for test_indexes to real values from user_movie_matrix

#print(test_indexes)


##get 10 top rated movies from user 1 from user_movie_matrix
#user1_ratings = user_movie_matrix.loc[252].sort_values(ascending=False).head(10)
##to each movieID get title from movies
#user1_ratings = user1_ratings.reset_index().merge(movies, on='movieId', how='left')
#print(user1_ratings)
#
##get 10 top rated movies from user 1 from user_movie_matrix_reconstructed
#user1_ratings_reconstructed = user_movie_matrix_reconstructed.loc[252].sort_values(ascending=False).head(10)
##to each movieID get title from movies
#user1_ratings_reconstructed = user1_ratings_reconstructed.reset_index().merge(movies, on='movieId', how='left')
#print(user1_ratings_reconstructed)

# # Normalizacja ocen - odejmowanie średniej oceny użytkownika
# user_means = user_movie_matrix.mean(axis=1)
# user_movie_matrix_normalized = user_movie_matrix.sub(user_means, axis=0).fillna(0)

# # Zapisz macierz użytkownik-film i normalizowane średnie
# #user_movie_matrix_normalized.to_pickle('user_movie_matrix_normalized.pkl')
# #user_means.to_pickle('user_means.pkl')
# #movies.to_pickle('movies.pkl')

# # Split danych na zbiór treningowy i testowy
# X_train, X_test = train_test_split(user_movie_matrix_normalized, test_size=0.2, random_state=42)

# # Pipeline modelu
# pipeline = Pipeline(steps=[
#     ('svd', TruncatedSVD(random_state=42))
# ])

# # Custom scoring function
# def reconstruction_error(estimator, X):
#     # Pipeline expects 2D array, ensure input is correct
#     if isinstance(X, pd.DataFrame):
#         X = X.values
#     X_transformed = estimator.named_steps['svd'].transform(X)
#     X_reconstructed = estimator.named_steps['svd'].inverse_transform(X_transformed)
#     return mean_squared_error(X, X_reconstructed)

# # Siatka parametrów dla GridSearchCV
# param_grid = {
#     'svd__n_components': [10, 20, 30, 40, 50]
# }

# # GridSearchCV
# model = Sequential()
# model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='relu'))

# model.compile(loss='mean_squared_error', optimizer='adam')

# model.fit(X_train, X_train, epochs=100, batch_size=10, verbose=2)

# model.evaluate(X_test, X_test)
# Najlepsze parametry
# best_params = grid_search.best_params_
# print(f'Best parameters: {best_params}')

# # Najlepszy model
# best_pipeline = grid_search.best_estimator_

# # Ewaluacja modelu
# X_train_transformed = best_pipeline.named_steps['svd'].transform(X_train)
# X_train_reconstructed = best_pipeline.named_steps['svd'].inverse_transform(X_train_transformed)
# train_mse = mean_squared_error(X_train, X_train_reconstructed)

# X_test_transformed = best_pipeline.named_steps['svd'].transform(X_test)
# X_test_reconstructed = best_pipeline.named_steps['svd'].inverse_transform(X_test_transformed)
# test_mse = mean_squared_error(X_test, X_test_reconstructed)

# print(f'Train Mean Squared Error: {train_mse}')
# print(f'Test Mean Squared Error: {test_mse}')

# Zapis najlepszego modelu
#with open('svd_model_best.pkl', 'wb') as f:
#    pickle.dump(best_pipeline, f)