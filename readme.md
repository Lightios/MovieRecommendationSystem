# Movie Recommendation System

## Opis projektu

Ten projekt to system rekomendacji filmów, który jest zbudowany na podstawie algorytmu SVD (Singular Value Decomposition). System rekomenduje filmy na podstawie ocen użytkowników dla różnych filmów. Użytkownik może również przeszukiwać filmy po tytule i dodawać swoje oceny dla filmów.

## Wymagania

Aby uruchomić ten projekt, potrzebujesz zależności zawartych w requirements.txt

## Uruchomienie projektu

Aby uruchomić ten projekt, wykonaj następujące kroki:

1. Sklonuj repozytorium na swoje lokalne urządzenie.
2. Przejdź do katalogu projektu.
3. Zainstaluj wymagane zależności.
4. Uruchom plik `main.py` za pomocą Pythona.

```bash
git clone [<repo_url](https://github.com/Lightios/MovieRecommendationSystem)>
cd <project_directory>
pip install -r requirements.txt
python src/main.py
```

## Przykłady użycia

Po uruchomieniu aplikacji, możesz przeszukiwać filmy wpisując tytuł filmu w pole tekstowe i naciskając przycisk "Search Movies by title". Wyniki wyszukiwania zostaną wyświetlone poniżej.

Możesz wybrać film, klikając na kartę filmu. Wybrany film zostanie podświetlony.

Możesz dodać ocenę dla wybranego filmu, wpisując ocenę w pole tekstowe i naciskając przycisk "Submit Rating". Twoja ocena zostanie dodana do systemu.

Aby uzyskać rekomendacje filmów, naciśnij przycisk "Get Recommendations". System wygeneruje listę filmów, które mogą Ci się spodobać na podstawie Twoich ocen.

Ten projekt jest dostępny na licencji MIT. Zobacz plik `LICENSE` dla szczegółów.
