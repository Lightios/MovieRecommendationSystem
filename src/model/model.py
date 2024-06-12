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

# Do każdego filmu dodaj listę rozmiaru all_tags z wartościami od 0 do 1 w zależności od tego czy film ma dany tag
movie_tags = tags.groupby('movieId')['tagId'].apply(list).reset_index()
movie_tags['tagId'] = movie_tags['tagId'].apply(lambda x: [2 if i in x else 0 for i in all_tags])

# Połącz dane filmów z tagami - jesli film nie ma tagów to wstaw listę zer
movies = movies.merge(movie_tags, on='movieId', how='left')
movies['tagId'] = movies['tagId'].fillna(movies['tagId'].apply(lambda x: [0]*len(all_tags)))

#Zrób listę gatunków
genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']

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