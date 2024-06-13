import pandas as pd
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.widget import Widget
from kivymd.app import MDApp
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialogButtonContainer, MDDialog, MDDialogHeadlineText, MDDialogSupportingText, \
    MDDialogContentContainer
from kivymd.uix.divider import MDDivider
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDListItem, MDListItemSupportingText
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import MDSnackbar, MDSnackbarText

# from model.model_functions import get_movie_recommendations, get_movie_id_by_title, filter_movies, predict_rating, update_user_ratings
# from ui.widgets.movie_card import MovieCard

from model.model_functions import filter_movies, convert_dict_to_dataframe, get_movie_title_by_id, get_predictions, update_user_ratings, \
    get_movie_id_by_title
from ui.widgets.movie_card import MovieCard

Builder.load_file('ui/widgets/movie_card.kv')


class MovieRecommendationApp(MDApp):
    selected_movie = StringProperty()

    def __init__(self, user_movie_matrix, user_movie_matrix_normalized, movies, user_means, len_tags, len_genres, **kwargs):
        super().__init__(**kwargs)
        self.user_movie_matrix = user_movie_matrix
        self.user_movie_matrix_normalized = user_movie_matrix_normalized
        self.movies = movies
        self.user_means = user_means
        self.len_tags = len_tags
        self.len_genres = len_genres
        self.user_id = 1
        self.user_ratings = {}
        self.selected_movie_id = None
        self.selected_card = None

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Dark"
        return self.root

    def search_movies(self):
        text = self.root.ids.title_input.text
        if text:
            movie_titles, movie_ids = filter_movies(self.movies, text)
        else:
            return

        self.root.ids.movies_stack.clear_widgets()
        for title, movie_id in zip(movie_titles, movie_ids):
            rating = self.user_ratings.get(movie_id, None)
            card = MovieCard(movie_title=title, movie_id=movie_id, rating=rating)
            self.root.ids.movies_stack.add_widget(card)

    def set_movie(self, title):
        self.root.ids.dropdown_text.text = title
        self.root.ids.movie_dropdown.dropdown_menu.dismiss()

    def submit_rating(self):
        movie_title = self.get_title()
        if movie_title:
            rating = float(self.root.ids.rating_input.text)
            if rating < 0.5 or rating > 5.0:
                MDSnackbar(
                    MDSnackbarText(
                        text='Rating must be between 0.5 and 5.0',
                    ),
                ).open()
                return

            movie_id = self.movies[self.movies['title'] == movie_title]['movieId'].values[0]
            self.user_ratings[movie_id] = rating
            text = f'Rating {rating} submitted for {movie_title}'
            self.root.ids.rating_label.text = str(rating)
            self.selected_card.set_rating(rating)
            MDSnackbar(
                MDSnackbarText(
                    text=text,
                ),
            ).open()
            print(self.user_ratings)

    def update_user_ratings(self):
        if not self.user_ratings:
            MDSnackbar(
                MDSnackbarText(
                    text='Please submit a rating first.',
                ),
            ).open()
            return False
        self.user_movie_matrix, self.user_movie_matrix_normalized, self.user_means = update_user_ratings(self.user_id, self.user_ratings, self.user_movie_matrix, self.user_movie_matrix_normalized, self.user_means)
        return True

    def show_recommendations(self):
        if self.update_user_ratings():
            last_movie_ratings = convert_dict_to_dataframe(self.user_ratings)
            recommendations = get_predictions(self.user_id, last_movie_ratings, self.movies, self.len_genres, self.len_tags, self.user_movie_matrix, self.user_movie_matrix_normalized, self.user_means)
            recommended_titles = []
            for movie_id in recommendations:
                title = get_movie_title_by_id(movie_id, self.movies)
                recommended_titles.append(title)

            content_items = []
            # Loop over the recommended_titles
            for title in recommended_titles:
                # Create a new MDListItem for each title
                item = MDListItem(
                    MDListItemSupportingText(
                        text=title,
                    )
                )
                # Add the item and a divider to the list
                content_items.append(item)
                content_items.append(MDDivider())

            self.dialog = MDDialog(
                # -----------------------Headline text-------------------------
                MDDialogHeadlineText(
                    text="Recommended Movies",
                ),
                # -----------------------Supporting text-----------------------
                MDDialogSupportingText(
                    text="Our model recommends the following movies for you:",
                ),
                # -----------------------Custom content------------------------
                MDDialogContentContainer(
                    *content_items,
                    orientation="vertical",
                ),
                # ---------------------Button container------------------------
                MDDialogButtonContainer(
                    Widget(),
                    MDButton(
                        MDButtonText(text="Close"),
                        style="text",
                        on_release=lambda x: self.dialog.dismiss(),
                    ),
                ),
                # -------------------------------------------------------------
            )
            self.dialog.open()

    def predict_rating(self):
        if self.update_user_ratings():
            # user_id = int(self.root.ids.user_id_input.text)
            movie_title = self.get_title()
            if not movie_title:
                return

            movie_title = movie_title.split(' (')[0]
            movie_id = get_movie_id_by_title(movie_title, self.movies)[0]

            predicted_rating = predict_rating(self.user_id, movie_id, self.svd, self.user_movie_matrix_normalized,
                                              self.user_means)
            if predicted_rating is None:
                self.root.ids.result_label.text = f'User ID {self.user_id} does not exist. Please submit a rating first.'
            else:
                # Scale the predicted rating
                predicted_rating = max(0.5, min(5.0, predicted_rating))

                self.dialog = MDDialog(
                    # -----------------------Headline text-------------------------
                    MDDialogHeadlineText(
                        text="Predicted Rating",
                    ),
                    # -----------------------Supporting text-----------------------
                    MDDialogSupportingText(
                        text=f'Predicted rating for {movie_title}: {predicted_rating:.2f}',
                    ),
                    # ---------------------Button container------------------------
                    MDDialogButtonContainer(
                        Widget(),
                        MDButton(
                            MDButtonText(text="Close"),
                            style="text",
                            on_release=lambda x: self.dialog.dismiss(),
                        ),
                    ),
                    # -------------------------------------------------------------
                )
                self.dialog.open()

                # self.root.ids.result_label.text = f'Predicted rating for {movie_title}: {predicted_rating:.2f}'

    def get_title(self):
        title = self.selected_movie
        if title == '':
            MDSnackbar(
                MDSnackbarText(
                    text='Please select a movie first.',
                ),
            ).open()
            return False

        return title

    def select_movie(self, card: MovieCard):
        self.selected_card = card
        self.selected_movie = card.movie_title
        self.selected_movie_id = card.movie_id

        if card.movie_id in self.user_ratings:
            self.root.ids.rating_label.text = str(self.user_ratings[card.movie_id])
        else:
            self.root.ids.rating_label.text = 'Not rated yet'
