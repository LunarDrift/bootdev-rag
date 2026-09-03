import math
import os
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH,
    load_movies,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)  # maps tokens -> sets of document ids
        self.docmap: dict[int, dict] = {}  # maps document ids -> full document objects
        self.term_frequencies = defaultdict(Counter)  # maps document ids -> Counters
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie
            # Concatenate the title and description
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)

        except FileNotFoundError as e:
            print(f"Could not find file: {e}")
            return

    def get_documents(self, term: str) -> list[int]:
        ids = self.index.get(term, set())
        return sorted(list(ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)

    def get_term_frequencies(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    index.load()
    query_tokens = tokenize_text(query)
    results = []
    seen = set()

    for query in query_tokens:
        matching_doc_ids = index.get_documents(query)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = index.docmap[doc_id]
            results.append(doc)

            if len(results) >= limit:
                return results

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


def load_stopwords(file: str = STOPWORDS_PATH) -> list[str]:
    with open(file, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)

    # Remove stopwords
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)

    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))

    return stemmed_words


def tokenize_single_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    return tokens[0]


def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_term_frequencies(doc_id, tokenize_single_term(term))


def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    token = tokenize_single_term(term)
    idf = math.log((len(index.docmap) + 1) / (len(index.get_documents(token)) + 1))
    return idf
