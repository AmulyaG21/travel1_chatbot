import logging
import uuid
from typing import List, Optional, Dict, Any

import numpy as np
from qdrant_client.models import PointStruct

from core.database import QdrantDB
from services.config import settings

logger = logging.getLogger("QdrantStorage")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

class QdrantStorage:
    """Handles storing and retrieving documents from Qdrant vector store."""

    def __init__(self, collection_name: str = None):
        self.db = QdrantDB()
        self.collection_name = collection_name or self.db.collection_name
        self.ensure_collection()

    def ensure_collection(self):
        """Ensure the collection exists."""
        if not self.db.collection_exists(self.collection_name):
            logger.info(f"Creating collection: {self.collection_name}")
            self.db.create_collection(self.collection_name)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return

        points = []
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue

            # Generate embedding
            embedding = self.db.embedder.encode(text).tolist()
            
            point = {
                "id": doc.get("id", str(uuid.uuid4())),
                "vector": {self.db.vector_name: embedding},
                "payload": {
                    "text": text,
                    "metadata": doc.get("metadata", {})
                }
            }
            points.append(point)

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.db.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        logger.info(f"Added {len(points)} documents to collection '{self.collection_name}'")

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        return self.db.search(
            query=query_text,
            collection_name=self.collection_name,
            top_k=top_k,
            score_threshold=score_threshold
        )

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.db.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return self.db.get_collection_info(self.collection_name)
