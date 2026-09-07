import os
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from lib.search_utils import MOVIE_EMBEDDINGS_PATH, Movie, load_movies


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


EmbeddingArray = NDArray[Any]


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: EmbeddingArray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> EmbeddingArray:
        if not text.strip():
            raise ValueError("input text is empty")

        input = [text]  # model.encode needs a list for batching
        embedding = self.model.encode(input)
        return embedding[0]  # only care about the first element

    def build_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        movie_strings: list[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search = SemanticSearch()
    movies = load_movies()
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query(query: str):
    search = SemanticSearch()
    embeddings = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embeddings[:3]}")
    print(f"Shape: {embeddings.shape}")
