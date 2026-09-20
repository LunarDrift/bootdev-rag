import os
from typing import Any

from lib.search_utils import (
    DOCUMENT_PREVIEW_LENGTH,
    Movie,
    SearchResult,
    format_search_result,
    load_movies,
)
from lib.semantic_search import cosine_similarity
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(
        self, documents: list[Movie] | None = None, model_name="clip-ViT-B-32"
    ) -> None:
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.model = SentenceTransformer(model_name)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str) -> NDArray[Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)
        image_embedding = self.model.encode([image], show_progress_bar=True)
        return image_embedding[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[SearchResult]:
        image_embedding = self.embed_image(image_path)
        similarities: list[tuple[int, float]] = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in similarities[:limit]:
            doc = self.documents[idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )
        return results


def verify_image_embedding(image_path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(
    image_path: str = "data/paddington.jpeg", limit: int = 5
) -> dict[str, str | list[SearchResult]]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    movies = load_movies()
    mm_search = MultimodalSearch(movies)
    return mm_search.search_with_image(image_path, limit)
