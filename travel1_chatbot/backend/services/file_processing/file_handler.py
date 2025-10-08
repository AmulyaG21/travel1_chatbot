from services.file_processing.text_extractor import extract_segments
from core.embeddings import embedding_model
from services.utils.qdrant_storage import QdrantStorage
from services.file_processing.chunker import Chunker
from typing import List, Dict


class FileHandler:
    def __init__(self):
        self.storage = QdrantStorage()
        self.chunker = Chunker()
        # No per-file tracking; queries will search across all stored chunks

    def chunk_and_embed(self, file_path: str) -> List[Dict]:
        segments = extract_segments(file_path)
        sub_chunks: List[Dict] = self.chunker.chunk_segments(segments)
        documents: List[Dict] = []
    
        for item in sub_chunks:
            text = item.get("text_chunk") or ""
        if not text:
            continue
        doc = {
            "text": text,
            "metadata": {
                "source": item.get("source"),
                "page": item.get("page"),
            }
        }
        documents.append(doc)
    return documents

def process_and_store(self, file_path: str, collection_name: str | None = None):
    documents = self.chunk_and_embed(file_path)
    if not documents:
        raise ValueError("[Error] No text extracted from file.")
    
    self.storage.add_documents(documents)  # ✅ Fixed method call
    return {"stored_chunks": len(documents)}