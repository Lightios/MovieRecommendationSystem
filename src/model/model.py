import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Wczytanie danych
ratings = pd.read_csv('../ml-latest-small/ratings.csv')
movies = pd.read_csv('../ml-latest-small/movies.csv')

# Tworzenie macierzy użytkownik-film
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Normalizacja ocen - odejmowanie średniej oceny użytkownika
user_means = user_movie_matrix.mean(axis=1)
user_movie_matrix_normalized = user_movie_matrix.sub(user_means, axis=0).fillna(0)

# Zapisz macierz użytkownik-film i normalizowane średnie
user_movie_matrix_normalized.to_pickle('user_movie_matrix_normalized.pkl')
user_means.to_pickle('user_means.pkl')
movies.to_pickle('movies.pkl')

# Split danych na zbiór treningowy i testowy
X_train, X_test = train_test_split(user_movie_matrix_normalized, test_size=0.2, random_state=42)

# Pipeline modelu
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler(with_mean=False)),
    ('svd', TruncatedSVD(n_components=20, random_state=42))
])

# Trening modelu
pipeline.fit(X_train)

# Ewaluacja modelu
X_train_transformed = pipeline.named_steps['svd'].transform(X_train)
X_train_reconstructed = pipeline.named_steps['svd'].inverse_transform(X_train_transformed)
train_mse = mean_squared_error(X_train, X_train_reconstructed)

X_test_transformed = pipeline.named_steps['svd'].transform(X_test)
X_test_reconstructed = pipeline.named_steps['svd'].inverse_transform(X_test_transformed)
test_mse = mean_squared_error(X_test, X_test_reconstructed)

print(f'Train Mean Squared Error: {train_mse}')
print(f'Test Mean Squared Error: {test_mse}')

# Zapis modelu
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
