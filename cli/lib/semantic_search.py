from typing import Any

from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

EmbeddingArray = NDArray[Any]


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str) -> EmbeddingArray:
        if not text.strip():
            raise ValueError("input text is empty")

        input = [text]  # model.encode needs a list for batching
        embedding = self.model.encode(input)
        return embedding[0]  # only care about the first element


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
