import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Wczytaj dane
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Przygotowanie macierzy użytkownik-film
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Podział danych na zbiory treningowe i testowe
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Przygotowanie macierzy treningowej
train_user_movie_matrix = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
train_matrix = csr_matrix(train_user_movie_matrix.values)

# Definicja funkcji do oceny modelu
def calculate_rmse(svd, test_data, train_user_movie_matrix):
    user_movie_matrix = train_user_movie_matrix.copy()
    test_users = test_data['userId'].unique()
    test_movies = test_data['movieId'].unique()

    # Filtruj test_users i test_movies, które są w train_user_movie_matrix
    test_users = [user for user in test_users if user in train_user_movie_matrix.index]
    test_movies = [movie for movie in test_movies if movie in train_user_movie_matrix.columns]

    user_movie_matrix = user_movie_matrix.loc[test_users, test_movies]
    original_ratings = test_data.set_index(['userId', 'movieId']).loc[test_users, test_movies]['rating']

    # Uzupełnij macierz brakującymi ocenami
    predictions = svd.inverse_transform(svd.transform(user_movie_matrix.values))

    pred_ratings = pd.DataFrame(predictions, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

    # Filtruj oceny tylko dla tych, które były w zbiorze testowym
    pred_ratings = pred_ratings.stack().loc[original_ratings.index]

    rmse = np.sqrt(mean_squared_error(original_ratings, pred_ratings))
    return rmse

# Tworzenie pipelinu
svd = TruncatedSVD(random_state=42)
pipeline = Pipeline([('svd', svd)])

# Ustalanie zakresu parametrów do optymalizacji
param_grid = {
    'svd__n_components': [20, 50, 100],
    'svd__algorithm': ['randomized', 'arpack'],
    'svd__n_iter': [5, 10, 20],
    'svd__tol': [0.0, 0.01, 0.1]
}

# Grid Search
gs = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
gs.fit(train_matrix)

# Najlepsze parametry
print("Best RMSE score attained: ", -gs.best_score_)
print("Best parameters: ", gs.best_params_)

# Trenowanie finalnego modelu z najlepszymi parametrami
best_svd = TruncatedSVD(
    n_components=gs.best_params_['svd__n_components'],
    algorithm=gs.best_params_['svd__algorithm'],
    n_iter=gs.best_params_['svd__n_iter'],
    tol=gs.best_params_['svd__tol'],
    random_state=42
)
best_svd.fit(train_matrix)

# Zapisz najlepszy model
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(best_svd, f)

# Zapisz macierz użytkownik-film i filmy
user_movie_matrix.to_pickle('user_movie_matrix.pkl')
movies.to_pickle('movies.pkl')
