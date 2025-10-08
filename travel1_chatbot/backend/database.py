import inspect
import logging
import time 
import uuid
from typing import List, Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Optional imports (may not exist on some qdrant-client versions)
try:
    from qdrant_client.models import SearchRequest, NamedVector
except Exception:
    SearchRequest = None
    NamedVector = None

# Some installations expose http models; not required but used if available
try:
    from qdrant_client.http.models import PayloadSchemaType
except Exception:
    PayloadSchemaType = None

from config import settings

logger = logging.getLogger("QdrantDB")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class QdrantDB:
    def __init__(self):
        logger.info(f"[INFO] Connecting to Qdrant at {settings.QDRANT_URL}...")
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=getattr(settings, "QDRANT_API_KEY", None),
            timeout=60.0
        )

        # Use collection name from configuration; defaults to "default"
        self.collection_name = getattr(settings, "QDRANT_COLLECTION", "default")
        # If your collection was created with a named vector, keep that name here.
        # The common default for many examples is "default".
        self.vector_name = "default"

        # Embedder used for query encoding (and to derive vector dimension)
        # Use the same model as ingestion for consistent vector size
        model_name = getattr(settings, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)
        try:
            self.vector_size = int(self.embedder.get_sentence_embedding_dimension())
        except Exception:
            # Fallback to common size for MiniLM if API not available
            self.vector_size = 384

        # Ensure the collection exists (best-effort) with the correct vector size
        self._ensure_collection_exists()

        logger.info("[INFO] QdrantDB initialized successfully.")

        # Search path controls
        self._search_mode = None  # "sdk" | "rest" once determined
        self._force_rest = bool(getattr(settings, "QDRANT_FORCE_REST", False))

    def _ensure_collection_exists(self):
        """Create collection (named vector) if it doesn't exist, and try to create payload index for file_id.
        Also, if an existing collection has an incompatible vector schema (no named vector 'default'), recreate it.
        """
        try:
            exists = False
            try:
                exists = self.client.collection_exists(self.collection_name)
            except Exception as e:
                logger.debug("collection_exists() check failed: %s", e)

            if not exists:
                logger.info(f"[INFO] Creating collection '{self.collection_name}' with vector '{self.vector_name}'...")
                # recreate_collection ensures collection and named vector config
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config={self.vector_name: VectorParams(size=self.vector_size, distance=Distance.COSINE)}
                )
                time.sleep(0.2)
            else:
                # Validate existing schema has our named vector
                try:
                    info = self.client.get_collection(self.collection_name)
                    # qdrant-client may expose either dict-like or attribute objects
                    vectors = None
                    if hasattr(info, "config") and getattr(info.config, "params", None):
                        vectors = getattr(info.config.params, "vectors", None)
                    if vectors is None and isinstance(getattr(info, "result", None), dict):
                        vectors = info.result.get("config", {}).get("params", {}).get("vectors")

                    needs_recreate = False
                    if isinstance(vectors, dict):
                        # Named vectors are represented as dict mapping
                        if self.vector_name not in vectors:
                            needs_recreate = True
                    else:
                        # Unnamed vector config -> incompatible with our named vector usage
                        needs_recreate = True

                    if needs_recreate:
                        logger.info(
                            "[INFO] Existing collection schema incompatible; recreating with named vector '%s'...",
                            self.vector_name,
                        )
                        self.client.recreate_collection(
                            collection_name=self.collection_name,
                            vectors_config={self.vector_name: VectorParams(size=self.vector_size, distance=Distance.COSINE)},
                        )
                        time.sleep(0.2)
                    else:
                        logger.info(f"[INFO] Collection '{self.collection_name}' already exists with proper schema.")
                except Exception as e:
                    logger.debug("Schema inspection failed (continuing): %s", e)
                    # Best effort: proceed without recreating

            # Create payload index for file_id (best-effort)
            if PayloadSchemaType is not None:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="file_id",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    logger.info("[INFO] Payload index created for 'file_id'.")
                except Exception as e:
                    logger.debug("create_payload_index() failed (continuing): %s", e)
        except Exception as e:
            logger.exception("Error in _ensure_collection_exists: %s", e)

    def ensure_collection(self, collection_name: Optional[str] = None):
        """Ensure a given collection exists with the correct named-vector schema.
        If not provided, uses the default self.collection_name.
        """
        name = collection_name or self.collection_name
        try:
            if not self.client.collection_exists(name):
                logger.info("[INFO] Creating collection '%s' on demand...", name)
                self.client.recreate_collection(
                    collection_name=name,
                    vectors_config={self.vector_name: VectorParams(size=self.vector_size, distance=Distance.COSINE)}
                )
                time.sleep(0.2)
        except Exception as e:
            logger.exception("ensure_collection('%s') failed: %s", name, e)

    def clear_collection(self):
        """Delete and recreate the collection with the expected named vector schema."""
        try:
            logger.info("[INFO] Clearing collection '%s'...", self.collection_name)
            try:
                self.client.delete_collection(self.collection_name)
            except Exception as e:
                logger.debug("delete_collection() failed or not present: %s", e)
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={self.vector_name: VectorParams(size=self.vector_size, distance=Distance.COSINE)},
            )
            logger.info("[INFO] Collection recreated.")
        except Exception as e:
            logger.exception("Failed to clear/recreate collection: %s", e)

    def collection_info(self):
        """Return basic collection information for debugging (safe to call)."""
        try:
            info = self.client.get_collection(self.collection_name)
            # Try to convert to plain dict for JSON friendliness
            if hasattr(info, "dict"):
                return info.dict()
            if isinstance(getattr(info, "result", None), dict):
                return info.result
            return str(info)
        except Exception as e:
            logger.exception("collection_info failed: %s", e)
            return {"error": str(e)}

    def count_points(self) -> int:
        """Return total number of points in the collection (best-effort)."""
        try:
            # Prefer count API if available; otherwise scroll limited
            try:
                res = self.client.count(collection_name=self.collection_name, exact=True)
                # qdrant-client may return object or dict
                if hasattr(res, "count"):
                    return int(res.count)
                if isinstance(getattr(res, "result", None), dict):
                    return int(res.result.get("count", 0))
            except Exception:
                pass

            # Fallback: use scroll and count manually (may be slower on large sets)
            total = 0
            next_offset = None
            while True:
                batch = self.client.scroll(
                    collection_name=self.collection_name,
                    with_payload=False,
                    with_vectors=False,
                    limit=1000,
                    offset=next_offset,
                )
                # qdrant-client returns (points, next_offset)
                points, next_offset = batch
                total += len(points or [])
                if not next_offset:
                    break
            return total
        except Exception as e:
            logger.exception("count_points failed: %s", e)
            return -1

    def store_chunks(self, embeddings_and_metadata: List[tuple], collection_name: Optional[str] = None):
        """
        embeddings_and_metadata: list of (vector, meta) where meta must include 'text_chunk'
        Stores using named vector format: vector={self.vector_name: vec}
        """
        BATCH_SIZE = 50
        points = []

        # Ensure target collection exists (default or custom)
        target_collection = collection_name or self.collection_name
        self.ensure_collection(target_collection)

        for vec, meta in embeddings_and_metadata:
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()

            if not isinstance(vec, (list, tuple)):
                logger.warning("Skipping invalid vector (not list/tuple): %s", type(vec))
                continue

            text_chunk = meta.get("text_chunk", "")
            source = meta.get("source")
            page = meta.get("page")

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={self.vector_name: vec},
                payload={"text_chunk": text_chunk, "source": source, "page": page}
            )
            points.append(point)

        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            try:
                self.client.upsert(collection_name=target_collection, points=batch)
            except Exception as e:
                logger.exception("Upsert failed for batch: %s", e)

        logger.info(f"[✅ INFO] Stored {len(points)} embeddings in Qdrant collection '{target_collection}'.")

    def _normalize_results_from_sdk(self, results):
        out = []
        for r in results:
            payload = getattr(r, "payload", None)
            if payload is None and isinstance(r, dict):
                payload = r.get("payload", {})
            score = getattr(r, "score", None) if not isinstance(r, dict) else r.get("score")
            out.append({
                "score": float(score) if score is not None else None,
                "text_chunk": (payload.get("text_chunk") if payload else None),
                "source": (payload.get("source") if payload else None),
                "page": (payload.get("page") if payload else None),
            })
        return out

    def _normalize_results_from_rest(self, json_data):
        # REST returns top-level 'result' list of points
        hits = json_data.get("result") or json_data.get("data") or []
        out = []
        for hit in hits:
            payload = hit.get("payload") or {}
            score = hit.get("score") or hit.get("dist") or None
            out.append({
                "score": float(score) if score is not None else None,
                "text_chunk": payload.get("text_chunk"),
                "source": payload.get("source"),
                "page": payload.get("page"),
            })
        return out

    def _http_search(self, qvec: List[float], top_k: int, filter_obj):
        """
        Direct REST fallback to /collections/{col}/points/search.
        Tries multiple body shapes:
          1) NamedVectorStruct: {"vector": {"name": vector_name, "vector": [...]}, "limit": top_k, "filter": ...}
          2) {"vector": [...], "using": vector_name, "limit": top_k, "filter": ...}
          3) {"vector": [...], "vector_name": vector_name, "limit": top_k, "filter": ...}
          4) Dict-keyed (older variants): {"vector": {vector_name: [...]}, ...}
        """
        base = settings.QDRANT_URL.rstrip("/")
        url = f"{base}/collections/{self.collection_name}/points/search"
        headers = {"Content-Type": "application/json"}
        api_key = getattr(settings, "QDRANT_API_KEY", None)
        if api_key:
            # Qdrant Cloud expects 'api-key' header
            headers["api-key"] = api_key

        # Convert filter_obj to plain dict if needed
        filt = None
        if filter_obj is not None:
            # If it's SDK Filter object, try .dict() or convert
            try:
                filt = filter_obj.dict()
            except Exception:
                filt = filter_obj if isinstance(filter_obj, dict) else None

        # Attempt A: NamedVectorStruct (preferred for Qdrant Cloud REST)
        body_a = {"vector": {"name": self.vector_name, "vector": qvec}, "limit": top_k}
        if filt is not None:
            body_a["filter"] = filt

        try:
            logger.debug("HTTP fallback: trying NamedVectorStruct to %s", url)
            resp = requests.post(url, json=body_a, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning("REST search (NamedVectorStruct) failed: %s %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
            data = resp.json()
            return self._normalize_results_from_rest(data)
        except Exception as e:
            logger.warning("HTTP NamedVectorStruct attempt failed: %s", e)

        # Attempt B: raw vector + using field (current REST naming)
        body_b = {"vector": qvec, "using": self.vector_name, "limit": top_k}
        if filt is not None:
            body_b["filter"] = filt

        try:
            logger.debug("HTTP fallback: trying vector + using body to %s", url)
            resp = requests.post(url, json=body_b, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning("REST search (using) failed: %s %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
            data = resp.json()
            return self._normalize_results_from_rest(data)
        except Exception as e:
            logger.warning("HTTP using attempt failed: %s", e)

        # Attempt C: raw vector + vector_name field (some deployments)
        body_c = {"vector": qvec, "vector_name": self.vector_name, "limit": top_k}
        if filt is not None:
            body_c["filter"] = filt
        try:
            logger.debug("HTTP fallback: trying vector + vector_name body to %s", url)
            resp = requests.post(url, json=body_c, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning("REST search (vector_name) failed: %s %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
            data = resp.json()
            return self._normalize_results_from_rest(data)
        except Exception as e:
            logger.warning("HTTP vector_name attempt failed: %s", e)

        # Attempt D: dict-keyed (older variants)
        body_d = {"vector": {self.vector_name: qvec}, "limit": top_k}
        if filt is not None:
            body_d["filter"] = filt
        try:
            logger.debug("HTTP fallback: trying dict-keyed body to %s", url)
            resp = requests.post(url, json=body_d, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning("REST search (dict-keyed) failed: %s %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
            data = resp.json()
            return self._normalize_results_from_rest(data)
        except Exception as e:
            logger.warning("HTTP dict-keyed attempt failed: %s", e)

        # No luck
        return []

    def search(self, query_text: str, top_k: int = 5, file_id: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Search pipeline:
         - encode with embedder
         - try several client.search call shapes
         - if all client attempts fail, fall back to HTTP REST call
        Returns list of dicts: {"score","text_chunk"}
        """
        if not isinstance(query_text, str):
            query_text = str(query_text)

        # Encode
        try:
            qvec = self.embedder.encode(query_text)
            if isinstance(qvec, np.ndarray):
                qvec = qvec.tolist()
        except Exception as e:
            logger.exception("Embedding encoding failed: %s", e)
            return []

        # file_id filter removed; search across all data
        filter_obj = None

        # Determine target collection and ensure it exists (no-op if present)
        target_collection = collection_name or self.collection_name
        self.ensure_collection(target_collection)

        # If forced or previously determined to use REST, go straight to REST
        if self._force_rest or self._search_mode == "rest":
            try:
                rest_res = self._http_search(qvec=qvec, top_k=top_k, filter_obj=filter_obj)
                if rest_res:
                    self._search_mode = "rest"
                    return rest_res
            except Exception as e:
                logger.exception("HTTP fallback (forced/cached) failed: %s", e)
                return []

        # Inspect client.search signature to see accepted kwargs
        try:
            sig = inspect.signature(self.client.search)
            valid_params = set(sig.parameters.keys())
        except Exception:
            valid_params = set()

        attempts = []
        # Variant 1: Try SearchRequest/NamedVector if available
        if SearchRequest is not None and NamedVector is not None:
            try:
                attempts.append("SearchRequest/NamedVector")
                sr = SearchRequest(vector=NamedVector(name=self.vector_name, vector=qvec),
                                   limit=top_k,
                                   filter=filter_obj)
                res = self.client.search(collection_name=target_collection, search_request=sr)
                self._search_mode = "sdk"
                return self._normalize_results_from_sdk(res)
            except Exception as e:
                logger.debug("SearchRequest attempt failed: %s", e)

        # Variant 2: try dict-keyed query_vector ({"default": [...]})
        try:
            attempts.append("query_vector dict keyed")
            kw = {"collection_name": target_collection, "query_vector": {self.vector_name: qvec}, "limit": top_k}
            if "query_filter" in valid_params:
                kw["query_filter"] = filter_obj
            elif "filter" in valid_params:
                kw["filter"] = filter_obj
            res = self.client.search(**kw)
            self._search_mode = "sdk"
            return self._normalize_results_from_sdk(res)
        except Exception as e:
            logger.debug("dict-keyed attempt failed: %s", e)

        # Variant 3: try raw list + vector_name kwarg (if client supports vector_name param)
        try:
            attempts.append("raw list + vector_name kwarg")
            kw = {"collection_name": target_collection, "query_vector": qvec, "limit": top_k}
            if "vector_name" in valid_params:
                kw["vector_name"] = self.vector_name
            if "with_payload" in valid_params:
                kw["with_payload"] = True
            if "query_filter" in valid_params:
                kw["query_filter"] = filter_obj
            elif "filter" in valid_params:
                kw["filter"] = filter_obj
            res = self.client.search(**kw)
            self._search_mode = "sdk"
            return self._normalize_results_from_sdk(res)
        except Exception as e:
            logger.debug("raw-list+vector_name attempt failed: %s", e)

        # Variant 4: try raw list + query_filter (some versions)
        try:
            attempts.append("raw list + query_filter")
            kw = {"collection_name": target_collection, "query_vector": qvec, "limit": top_k}
            if "with_payload" in valid_params:
                kw["with_payload"] = True
            kw["query_filter"] = filter_obj
            res = self.client.search(**kw)
            self._search_mode = "sdk"
            return self._normalize_results_from_sdk(res)
        except Exception as e:
            logger.debug("raw list + query_filter attempt failed: %s", e)

        # Variant 5: some client versions use 'vector' instead of 'query_vector'
        try:
            attempts.append("vector param (dict keyed)")
            kw = {"collection_name": target_collection, "vector": {self.vector_name: qvec}, "limit": top_k}
            if "with_payload" in valid_params:
                kw["with_payload"] = True
            if "query_filter" in valid_params:
                kw["query_filter"] = filter_obj
            elif "filter" in valid_params:
                kw["filter"] = filter_obj
            res = self.client.search(**kw)
            self._search_mode = "sdk"
            return self._normalize_results_from_sdk(res)
        except Exception as e:
            logger.debug("vector param (dict keyed) attempt failed: %s", e)

        try:
            attempts.append("vector param + vector_name")
            kw = {"collection_name": target_collection, "vector": qvec, "limit": top_k}
            if "vector_name" in valid_params:
                kw["vector_name"] = self.vector_name
            if "with_payload" in valid_params:
                kw["with_payload"] = True
            if "query_filter" in valid_params:
                kw["query_filter"] = filter_obj
            elif "filter" in valid_params:
                kw["filter"] = filter_obj
            res = self.client.search(**kw)
            self._search_mode = "sdk"
            return self._normalize_results_from_sdk(res)
        except Exception as e:
            logger.debug("vector param + vector_name attempt failed: %s", e)

        # Variant 6: use 'using' kwarg to select named vector (newer SDKs)
        try:
            attempts.append("raw list + using kwarg")
            kw = {"collection_name": target_collection, "query_vector": qvec, "limit": top_k}
            if "using" in valid_params:
                kw["using"] = self.vector_name
            if "with_payload" in valid_params:
                kw["with_payload"] = True
            if "query_filter" in valid_params:
                kw["query_filter"] = filter_obj
            elif "filter" in valid_params:
                kw["filter"] = filter_obj
            res = self.client.search(**kw)
            self._search_mode = "sdk"
            return self._normalize_results_from_sdk(res)
        except Exception as e:
            logger.debug("raw list + using kwarg attempt failed: %s", e)

        if self._search_mode != "rest":
            logger.warning("All client.search attempts failed. Tried: %s. Falling back to HTTP REST.", attempts)

        # Final fallback: HTTP REST calls against target collection via REST helper (uses default in URL)
        try:
            rest_res = self._http_search(qvec=qvec, top_k=top_k, filter_obj=None)
            if rest_res:
                self._search_mode = "rest"
                return rest_res
        except Exception as e:
            logger.exception("HTTP fallback failed: %s", e)

        logger.error("All search attempts failed.")
        return []
