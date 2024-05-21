from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
import pandas as pd
import numpy as np
import pickle

# Załaduj model i dane
with open('svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)
user_movie_matrix = pd.read_pickle('user_movie_matrix.pkl')
movies = pd.read_pickle('movies.pkl')


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


class MovieRecommendationApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Dark"
        return self.root

    def search_movies(self):
        text = self.root.ids.year_input.text
        if text:
            movie_titles = filter_movies(movies, text)
        else:
            movie_titles = movies['title'].tolist()

        menu_items = [{"text": title, "on_release": lambda x=title: self.set_movie(x)} for title in movie_titles]

        self.root.ids.movie_dropdown.dropdown_menu = MDDropdownMenu(
            caller=self.root.ids.movie_dropdown,
            items=menu_items,
            width_mult=4
        )
        self.root.ids.movie_dropdown.text = 'Select a Movie'
        self.root.ids.movie_dropdown.dropdown_menu.open()

    def set_movie(self, title):
        self.root.ids.dropdown_text.text = title
        self.root.ids.movie_dropdown.dropdown_menu.dismiss()

    def submit_rating(self):
        user_id_text = self.root.ids.user_id_input.text
        if user_id_text:
            user_id = int(user_id_text)
        else:
            # Assign new ID
            user_id = user_movie_matrix.index.max() + 1

        movie_title = self.root.ids.dropdown_text.text
        rating = float(self.root.ids.rating_input.text)

        movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]

        if user_id in user_movie_matrix.index:
            user_movie_matrix.loc[user_id, movie_id] = rating
        else:
            new_user_ratings = pd.Series(0, index=user_movie_matrix.columns)
            new_user_ratings[movie_id] = rating
            user_movie_matrix.loc[user_id] = new_user_ratings

        self.root.ids.result_label.text = f'Rating submitted for {movie_title}\nUser ID: {user_id}'

    def show_recommendations(self):
        user_id = int(self.root.ids.user_id_input.text)
        recommendations = get_movie_recommendations(user_id, user_movie_matrix, svd, movies)
        recommended_titles = recommendations['title'].tolist()
        self.root.ids.result_label.text = 'Recommended Movies:\n' + '\n'.join(recommended_titles)

    def predict_rating(self):
        user_id = int(self.root.ids.user_id_input.text)
        movie_title = self.root.ids.movie_dropdown.text
        predicted_rating = predict_movie_rating(user_id, movie_title, user_movie_matrix, svd, movies)
        if predicted_rating is None:
            self.root.ids.result_label.text = f'User ID {user_id} does not exist. Please submit a rating first.'
        else:
            # Scale the predicted rating
            predicted_rating = max(0.5, min(5.0, predicted_rating))
            self.root.ids.result_label.text = f'Predicted rating for {movie_title}: {predicted_rating:.2f}'


if __name__ == '__main__':
    MovieRecommendationApp().run()
