import os
from typing import Any

from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(
        self, documents: list[object] | None = None, model_name="clip-ViT-B-32"
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str) -> NDArray[Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)
        image_embedding = self.model.encode([image])
        return image_embedding[0]


def verify_image_embedding(image_path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
