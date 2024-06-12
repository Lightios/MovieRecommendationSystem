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

#Zapisz len(all_tags) do pickle
#len(all_tags).to_pickle('len_tags.pkl')

# Do każdego filmu dodaj listę rozmiaru all_tags z wartościami od 0 do 1 w zależności od tego czy film ma dany tag
movie_tags = tags.groupby('movieId')['tagId'].apply(list).reset_index()
movie_tags['tagId'] = movie_tags['tagId'].apply(lambda x: [2 if i in x else 0 for i in all_tags])

# Połącz dane filmów z tagami - jesli film nie ma tagów to wstaw listę zer
movies = movies.merge(movie_tags, on='movieId', how='left')
movies['tagId'] = movies['tagId'].fillna(movies['tagId'].apply(lambda x: [0]*len(all_tags)))

#Zrób listę gatunków
genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']

#len(genres).to_pickle('len_genres.pkl')
#zrób 2-elementowy dataframe z len(all_tags) i len(genres)
pd.DataFrame([len(all_tags), len(genres)]).to_pickle('src/model/len_tags_genres.pkl')

#Rozdziel movies['genres'] - podziałką jest '|'
movies['genres'] = movies['genres'].str.split('|')

#zamien movies['genres'] na listę 0 i 1 w zależności od tego czy film ma dany gatunek
movies['genres_temp'] = movies['genres'].apply(lambda x: [1 if i in x else 0 for i in genres])

#dodaj listę z gatunkami na początku listy tagID
movies['tagId'] = movies['genres_temp'] + movies['tagId']
movies.drop(columns=['genres_temp', 'genres'], inplace=True)

movies.to_pickle('src/model/movies.pkl')

# Tworzenie macierzy użytkownik-film
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_movie_matrix.to_pickle('src/model/user_movie_matrix.pkl')

# Normalizacja ocen - zamiast NaN wstawmy średnią ocenę filmu
user_movie_matrix_normalized = user_movie_matrix.fillna(user_movie_matrix.mean(axis=0)).fillna(0)
# Normalizacja ocen - odejmowanie średniej oceny użytkownika
user_means = user_movie_matrix_normalized.mean(axis=1)
user_movie_matrix_normalized = user_movie_matrix_normalized.sub(user_means, axis=0).fillna(0)

# Zapisz macierz użytkownik-film i normalizowane średnie
user_movie_matrix_normalized.to_pickle('src/model/user_movie_matrix_normalized.pkl')
user_means.to_pickle('src/model/user_means.pkl')

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