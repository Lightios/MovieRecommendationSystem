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
    ('svd', TruncatedSVD(random_state=42))
])

# Custom scoring function
def reconstruction_error(estimator, X):
    # Pipeline expects 2D array, ensure input is correct
    if isinstance(X, pd.DataFrame):
        X = X.values
    X_transformed = estimator.named_steps['svd'].transform(X)
    X_reconstructed = estimator.named_steps['svd'].inverse_transform(X_transformed)
    return mean_squared_error(X, X_reconstructed)

# Siatka parametrów dla GridSearchCV
param_grid = {
    'svd__n_components': [10, 20, 30, 40, 50]
}

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=make_scorer(reconstruction_error, greater_is_better=False), n_jobs=-1)
grid_search.fit(X_train)

# Najlepsze parametry
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

# Najlepszy model
best_pipeline = grid_search.best_estimator_

# Ewaluacja modelu
X_train_transformed = best_pipeline.named_steps['svd'].transform(X_train)
X_train_reconstructed = best_pipeline.named_steps['svd'].inverse_transform(X_train_transformed)
train_mse = mean_squared_error(X_train, X_train_reconstructed)

X_test_transformed = best_pipeline.named_steps['svd'].transform(X_test)
X_test_reconstructed = best_pipeline.named_steps['svd'].inverse_transform(X_test_transformed)
test_mse = mean_squared_error(X_test, X_test_reconstructed)

print(f'Train Mean Squared Error: {train_mse}')
print(f'Test Mean Squared Error: {test_mse}')

# Zapis najlepszego modelu
with open('svd_model_best.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)
