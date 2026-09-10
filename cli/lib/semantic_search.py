import os
from typing import Any, TypedDict

import numpy as np
from lib.search_utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    MOVIE_EMBEDDINGS_PATH,
    Movie,
    load_movies,
)
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


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

    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SemanticSearchResult]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call 'load_or_create_embeddings' first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call 'load_or_create_embeddings' first."
            )

        query_embedding = self.generate_embedding(query)
        similarities: list[tuple[float, Movie]] = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results: list[SemanticSearchResult] = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )
        return results


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


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


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    search = SemanticSearch()
    documents = load_movies()
    search.load_or_create_embeddings(documents)

    results = search.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    words = text.split()
    chunks = []

    num_words = len(words)
    i = 0
    while i < num_words:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size

    return chunks


def chunk_text(query: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    chunks = fixed_size_chunking(query, chunk_size)
    print(f"Chunking {len(query)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
