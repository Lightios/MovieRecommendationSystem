import pandas as pd
from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import MDSnackbar, MDSnackbarText

from src.model.model_functions import get_movie_recommendations, get_movie_id_by_title, \
    filter_movies, predict_rating, update_user_ratings


class MovieRecommendationApp(MDApp):
    def __init__(self, svd, user_movie_matrix, movies, user_means, **kwargs):
        super().__init__(**kwargs)
        self.svd = svd
        self.user_movie_matrix = user_movie_matrix
        self.movies = movies
        self.user_means = user_means
        self.user_id = 2000
        self.user_ratings = {}

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Dark"
        return self.root

    def search_movies(self):
        text = self.root.ids.title_input.text
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
        movie_title = self.get_title()
        if movie_title:
            rating = float(self.root.ids.rating_input.text)

            movie_id = self.movies[self.movies['title'] == movie_title]['movieId'].values[0]
            self.user_ratings[movie_id] = rating
            text = f'Rating {rating} submitted for {movie_title}'

            MDSnackbar(
                MDSnackbarText(
                    text=text,
                ),
            ).open()

    def update_user_ratings(self):
        if not self.user_ratings:
            MDSnackbar(
                MDSnackbarText(
                    text='Please submit a rating first.',
                ),
            ).open()
            return False

        self.user_movie_matrix, self.user_movie_matrix_normalized, self.user_means = update_user_ratings(self.user_id, self.user_ratings, self.user_movie_matrix, self.user_means)
        return True

    def show_recommendations(self):
        # user_id_text = self.root.ids.user_id_input.text
        # if user_id_text:
        #     user_id = int(user_id_text)
        # else:
        #     # Assign new ID
        #     user_id = self.user_movie_matrix.index.max() + 1
        if self.update_user_ratings():
            recommendations = get_movie_recommendations(self.user_id, self.svd, self.user_movie_matrix_normalized, self.movies, self.user_means)
            recommended_titles = recommendations['title'].tolist()
            self.root.ids.result_label.text = 'Recommended Movies:\n' + '\n'.join(recommended_titles)

    def predict_rating(self):
        if self.update_user_ratings():
            # user_id = int(self.root.ids.user_id_input.text)
            movie_title = self.get_title()
            if not movie_title:
                return

            movie_title = movie_title.split(' (')[0]
            movie_id = get_movie_id_by_title(movie_title, self.movies)[0]

            predicted_rating = predict_rating(self.user_id, movie_id, self.svd, self.user_movie_matrix_normalized, self.user_means)
            if predicted_rating is None:
                self.root.ids.result_label.text = f'User ID {self.user_id} does not exist. Please submit a rating first.'
            else:
                # Scale the predicted rating
                predicted_rating = max(0.5, min(5.0, predicted_rating))
                self.root.ids.result_label.text = f'Predicted rating for {movie_title}: {predicted_rating:.2f}'

    def get_title(self):
        title = self.root.ids.dropdown_text.text
        if title == 'Select a Movie':
            MDSnackbar(
                MDSnackbarText(
                    text='Please select a movie first.',
                ),
            ).open()
            return False

        return title
