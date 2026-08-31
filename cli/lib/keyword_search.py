import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        preprocessed_query = process_text(query)
        preprocessed_title = process_text(movie["title"])
        if preprocessed_query in preprocessed_title:
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def process_text(text: str) -> str:
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization

    return text
