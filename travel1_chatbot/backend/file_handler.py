from text_extractor import extract_segments
from embeddings import embedding_model
from backend.database import QdrantDB
from typing import List, Dict


class FileHandler:
    def __init__(self):
        self.db = QdrantDB()
        # No per-file tracking; queries will search across all stored chunks

    def _split_long_text(self, text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
        if len(text) <= max_chars:
            return [text.strip()]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    def chunk_and_embed(self, file_path: str) -> List[tuple]:
        segments = extract_segments(file_path)
        embeddings_and_metadata: List[tuple] = []

        for seg in segments:
            base_meta: Dict = {
                "text_chunk": None,  # will set per sub-chunk
                "source": seg.get("source"),
                "page": seg.get("page"),
            }
            for sub in self._split_long_text(seg.get("text", "")):
                if not sub:
                    continue
                vec = embedding_model.encode(sub).tolist()
                meta = {**base_meta, "text_chunk": sub}
                embeddings_and_metadata.append((vec, meta))

        return embeddings_and_metadata

    def process_and_store(self, file_path: str, collection_name: str | None = None):
        # Build embeddings with source/page-aware segments
        embeddings_and_metadata = self.chunk_and_embed(file_path)
        if not embeddings_and_metadata:
            raise ValueError("[Error] No text extracted from file.")

        # Store new embeddings with metadata in Qdrant (collection_name ignored here)
        self.db.store_chunks(embeddings_and_metadata)

        return {"stored_chunks": len(embeddings_and_metadata)}

