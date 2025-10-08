import logging
import time
import ssl
import certifi
import uuid
from typing import Optional, List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, Distance

try:
    from qdrant_client.models import SearchRequest, NamedVector
except ImportError:
    SearchRequest = None
    NamedVector = None

from services.config import settings

logger = logging.getLogger("QdrantDB")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

class QdrantDB:
    def __init__(self, use_ssl_context: bool = False):
        """
        Initialize QdrantDB connection.

        Args:
            use_ssl_context (bool): If True, use custom SSL context 
                                  (for self-signed/local Qdrant). 
                                  Default False (Qdrant Cloud safe).
        """
        logger.info(f"[INFO] Connecting to Qdrant at {settings.QDRANT_URL}...")

        try:
            if use_ssl_context:
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                self.client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=getattr(settings, "QDRANT_API_KEY", None),
                    timeout=60.0,
                    https=True,
                )
                logger.info("[INFO] Connected to Qdrant using custom SSL context (DEV MODE)")
            else:
                self.client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=getattr(settings, "QDRANT_API_KEY", None),
                    timeout=60.0,
                )
                logger.info("[INFO] Connected to Qdrant using standard HTTPS")

        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to Qdrant: {str(e)}")
            raise

        self.collection_name = getattr(settings, "QDRANT_COLLECTION", "travel1_chatbot")
        self.vector_name = "default"
        model_name = getattr(settings, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)
        self.vector_size = self.embedder.get_sentence_embedding_dimension()

        # Ensure collection exists
        self.ensure_collection(self.collection_name)

    def ensure_collection(self, collection_name: Optional[str] = None):
        """Ensure collection exists, create if it doesn't."""
        collection_name = collection_name or self.collection_name
        try:
            if not self.collection_exists(collection_name):
                logger.info(f"Creating collection: {collection_name}")
                self.create_collection(collection_name)
            else:
                logger.debug(f"Collection exists: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def create_collection(self, collection_name: str = None, recreate: bool = False) -> None:
        """Create a new collection with the specified name."""
        collection_name = collection_name or self.collection_name
        logger.info(f"Creating collection: {collection_name}")

        if recreate:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
        else:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
        
        self.collection_name = collection_name
        logger.info(f"Collection '{collection_name}' created successfully")

    def delete_collection(self, collection_name: str = None) -> None:
        """Delete the specified collection."""
        collection_name = collection_name or self.collection_name
        logger.info(f"Deleting collection: {collection_name}")
        self.client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully")

    def add_documents(self, documents: List[Dict[str, Any]], collection_name: str = None) -> None:
        """Add documents to the collection."""
        collection_name = collection_name or self.collection_name
        logger.info(f"Adding {len(documents)} documents to {collection_name}")

        points = []
        for doc in documents:
            embedding = self.embedder.encode(doc["text"]).tolist()
            
            point = {
                "id": doc.get("id", str(uuid.uuid4())),
                "vector": {self.vector_name: embedding},
                "payload": {
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {})
                }
            }
            points.append(point)

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )

        logger.info(f"Successfully added {len(documents)} documents to {collection_name}")

    def search(
        self,
        query: str,
        collection_name: str = None,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        collection_name = collection_name or self.collection_name
        logger.info(f"Searching in {collection_name} for: {query[:50]}...")

        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Perform search
        search_params = {
            "collection_name": collection_name,
            "query_vector": (self.vector_name, query_embedding),
            "limit": top_k,
            "score_threshold": score_threshold,
            "with_payload": True,
            "with_vectors": False
        }

        if SearchRequest is not None and hasattr(self.client, 'search_batch'):
            search_request = SearchRequest(
                vector=NamedVector(name=self.vector_name, vector=query_embedding),
                limit=top_k,
                with_payload=True
            )
            results = self.client.search_batch(
                collection_name=collection_name,
                requests=[search_request]
            )
            results = results[0]  # Get first (and only) search result
        else:
            results = self.client.search(**search_params)

        # Format results
        formatted_results = []
        for hit in results:
            formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload.get("metadata", {})
            })

        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get information about the collection."""
        collection_name = collection_name or self.collection_name
        return self.client.get_collection(collection_name=collection_name)

    def collection_exists(self, collection_name: str = None) -> bool:
        """Check if a collection exists."""
        collection_name = collection_name or self.collection_name
        try:
            self.get_collection_info(collection_name)
            return True
        except Exception:
            return False
