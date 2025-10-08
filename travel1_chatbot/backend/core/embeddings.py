from __future__ import annotations

from typing import Iterable, List, Optional

from sentence_transformers import SentenceTransformer
from services.config import settings

_MODEL_SINGLETON: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Return a process-wide singleton SentenceTransformer model.
    Avoids reloading the model multiple times across modules.
    """
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _MODEL_SINGLETON


# Backward-compat: export a module-level model instance
embedding_model: SentenceTransformer = get_embedding_model()


def embed_text(text: str) -> List[float]:
    """Encode a single text into an embedding list[float]."""
    return get_embedding_model().encode(text).tolist()


def embed_batch(texts: Iterable[str]) -> List[List[float]]:
    """Encode a batch of texts. Returns list of embeddings as lists of floats."""
    return get_embedding_model().encode(list(texts)).tolist()


class EmbeddingGenerator:
    """Compatibility wrapper with helpful batch API."""

    def __init__(self):
        self.model = get_embedding_model()

    def generate_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def generate_batch(self, texts: Iterable[str]) -> List[List[float]]:
        return self.model.encode(list(texts)).tolist()

