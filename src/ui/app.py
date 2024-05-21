import pandas as pd
from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu

from src.model.model_functions import filter_movies, get_movie_recommendations, predict_movie_rating


class MovieRecommendationApp(MDApp):
    def __init__(self, svd, user_movie_matrix, movies, **kwargs):
        super().__init__(**kwargs)
        self.svd = svd
        self.user_movie_matrix = user_movie_matrix
        self.movies = movies

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Dark"
        return self.root

    def search_movies(self):
        text = self.root.ids.year_input.text
        if text:
            movie_titles = filter_movies(self.movies, text)
        else:
            movie_titles = self.movies['title'].tolist()

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
            user_id = self.user_movie_matrix.index.max() + 1

        movie_title = self.root.ids.dropdown_text.text
        rating = float(self.root.ids.rating_input.text)

        movie_id = self.movies[self.movies['title'] == movie_title]['movieId'].values[0]

        if user_id in self.user_movie_matrix.index:
            self.user_movie_matrix.loc[user_id, movie_id] = rating
        else:
            new_user_ratings = pd.Series(0, index=self.user_movie_matrix.columns)
            new_user_ratings[movie_id] = rating
            self.user_movie_matrix.loc[user_id] = new_user_ratings

        self.root.ids.result_label.text = f'Rating submitted for {movie_title}\nUser ID: {user_id}'

    def show_recommendations(self):
        user_id = int(self.root.ids.user_id_input.text)
        recommendations = get_movie_recommendations(user_id, self.user_movie_matrix, self.svd, self.movies)
        recommended_titles = recommendations['title'].tolist()
        self.root.ids.result_label.text = 'Recommended Movies:\n' + '\n'.join(recommended_titles)

    def predict_rating(self):
        user_id = int(self.root.ids.user_id_input.text)
        movie_title = self.root.ids.movie_dropdown.text
        predicted_rating = predict_movie_rating(user_id, movie_title, self.user_movie_matrix, self.svd, self.movies)
        if predicted_rating is None:
            self.root.ids.result_label.text = f'User ID {user_id} does not exist. Please submit a rating first.'
        else:
            # Scale the predicted rating
            predicted_rating = max(0.5, min(5.0, predicted_rating))
            self.root.ids.result_label.text = f'Predicted rating for {movie_title}: {predicted_rating:.2f}'
