from kivy.app import App
from kivy.properties import StringProperty, ListProperty, NumericProperty
from kivymd.uix.card import MDCard


class MovieCard(MDCard):
    movie_title = StringProperty()
    movie_id = NumericProperty()
    card_color = ListProperty([1, 1, 1, 1])  # white color

    def __init__(self, rating, **kwargs):
        super().__init__(**kwargs)
        self.movie_title = kwargs.get('movie_title')
        self.movie_id = kwargs.get('movie_id')
        self.rating = rating

        if self.rating is None:
            self.ids.rating_label.text = f'Not rated yet'
        else:
            self.ids.rating_label.text = f'Your rating: {self.rating}'

    def on_release(self):
        self.card_color = [1, 0, 0, 1]  # red color
        app = App.get_running_app()
        app.select_movie(self)

    def set_rating(self, rating):
        self.rating = rating
        self.ids.rating_label.text = f'Your rating: {rating}'
