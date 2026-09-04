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

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        """Calculate the Inverse Document Frequency of term.
        IDF Formula: math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        +1s are to prevent division-by-zero when term doesn't appear in any documents
        """
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[term])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        """TF-IDF = TF * IDF"""
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        """
        Gets the BM25 IDF for a given term - assuming term is a single token.
        BM25 IDF Formula:
        IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        Where:
         - N = total number of documents
         - df = document frequency
        """
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)


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
    return index.get_tf(doc_id, tokenize_single_term(term))


def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    token = tokenize_single_term(term)
    return index.get_idf(token)


def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_tf_idf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(tokenize_single_term(term))
