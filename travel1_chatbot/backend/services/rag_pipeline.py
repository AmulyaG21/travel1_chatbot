# from qdrant_storage import QdrantSearch
# from gemini_api import GeminiClient
# import logging
# from typing import Dict, List, Optional, Generator
# from travel_api import TravelAPIClient
# from redis import redis_client

# logger = logging.getLogger("RAGPipeline")
# if not logger.handlers:
#     h = logging.StreamHandler()
#     h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
#     logger.addHandler(h)
# logger.setLevel(logging.INFO)


# class RAGPipeline:
#     def __init__(self):
#         self.search = QdrantStorage()
#         self.gemini_client = GeminiClient()
#         self.travel_client = TravelAPIClient()
#         # Simple in-memory conversation history per conversation_id
#         # {conversation_id: [{"role":"user"|"assistant","content": str}, ...]}
#         self._conversations: Dict[str, List[Dict[str, str]]] = {}
#         # How many prior messages to include
#         self._history_max_messages = 6

#     def _build_prompt(self, query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
#         history_text = ""
#         if history:
#             # Include only recent messages for brevity
#             trimmed = history[-self._history_max_messages :]
#             lines = []
#             for turn in trimmed:
#                 role = turn.get("role", "user")
#                 content = turn.get("content", "")
#                 lines.append(f"{role}: {content}")
#             history_text = "\n".join(lines)

#         prompt = (
#             "You are an assistant that answers questions using the provided context. "
#             "Be concise and return useful, accurate information based on the context. "
#             "If the answer is not contained in the context, say you don't know.\n\n"
#         )
#         if history_text:
#             prompt += f"Conversation so far:\n{history_text}\n\n"
#         prompt += (
#             f"Context:\n{context}\n\n"
#             f"Question: {query}\n\n"
#             "Answer:"
#         )
#         return prompt

#     def _update_history(self, conversation_id: Optional[str], user: str, assistant: Optional[str] = None):
#         if not conversation_id:
#             return
#         convo = self._conversations.setdefault(conversation_id, [])
#         if user:
#             convo.append({"role": "user", "content": user})
#         if assistant:
#             convo.append({"role": "assistant", "content": assistant})
#         # Trim
#         if len(convo) > 50:
#             del convo[: len(convo) - 50]

#     # def _handle_travel_query(self,query: str)-> Optional[Dict]:
#     def _handle_travel_query(self, query: str) -> Optional[Dict]:
#         """Check if the query is travel-related and fetch real-time data if so."""
#         q = query.lower()
#         if "flight" in q or "travel" in q or "trip" in q:
#             try:
#                 # Example: hard-coded extraction (replace with NLP parsing if needed)
#                 # Here we just use dummy values for demo
#                 origin, destination, date = "DEL", "BOM", "2025-10-15"
#                 flights = self.travel_client.search_flights(origin, destination, date)
#                 return {
#                     "response": f"Here are flight options from {origin} to {destination} on {date}: {flights}",
#                     "citations": []
#                 }
#             except Exception as e:
#                 logger.exception("Travel API failed: %s", e)
#                 return {"response": "Sorry — Travel API request failed.", "citations": []}
#         return None

#     def answer_query(self, query: str, conversation_id: Optional[str] = None) -> Dict:
#         logger.info("RAG: answering query (truncated): %s", (query or "")[:120])

#         travel_resp = self._handle_travel_query(query)
#         if travel_resp:
#             self._update_history(conversation_id, user=query, assistant=travel_resp["response"])
#             return travel_resp
#         try:
#             # Search service will encode the query text and run robust search
#             search_results = self.search.search(query_text=query, top_k=5, file_id=None)
#         except Exception as e:
#             logger.exception("DB.search raised: %s", e)
#             return {"response": "Sorry — internal error while searching the knowledge base."}

#         if not search_results:
#             self._update_history(conversation_id, user=query, assistant="Sorry, no relevant information found.")
#             return {"response": "Sorry, no relevant information found.", "citations": []}

#         # Build context, dedupe, and cap length to keep prompt reasonable
#         chunks = []
#         citations = []
#         seen_chunks = set()
#         total_chars = 0
#         for r in search_results:
#             c = r.get("text_chunk")
#             if not c or c in seen_chunks:
#                 continue
#             seen_chunks.add(c)
#             chunks.append(c)
#             # capture citation info if available
#             src = r.get("source")
#             page = r.get("page")
#             if src or page:
#                 citations.append({"source": src, "page": page})
#             total_chars += len(c)
#             if total_chars > 4000:
#                 break

#         context = "\n\n".join(chunks)
#         history = self._conversations.get(conversation_id or "", [])
#         prompt = self._build_prompt(query=query, context=context, history=history)

#         try:
#             response_text = self.gemini_client.generate_response(prompt)
#         except Exception as e:
#             logger.exception("LLM generation failed: %s", e)
#             response_text = "Sorry — failed to generate an answer from the LLM."

#         self._update_history(conversation_id, user=query, assistant=response_text)
#         # Deduplicate citations and keep order
#         deduped = []
#         seen = set()
#         for c in citations:
#             key = (c.get("source"), c.get("page"))
#             if key in seen:
#                 continue
#             seen.add(key)
#             deduped.append(c)

#         return {"response": response_text, "citations": deduped}

#     def stream_answer(self, query: str, conversation_id: Optional[str] = None) -> Generator[str, None, None]:
#         """Stream only the LLM response text. Caller can fetch citations separately by calling a non-stream method first.
#         """
#         logger.info("RAG: streaming query (truncated): %s", (query or "")[:120])

#         if "flight" in query.lower():
#             yield "Streaming not supported for travel queries. Please use non-stream mode."
#             return
        
#         try:
#             search_results = self.search.search(query_text=query, top_k=5, file_id=None)
#         except Exception as e:
#             logger.exception("DB.search raised: %s", e)
#             yield "Sorry — internal error while searching the knowledge base."
#             return

#         if not search_results:
#             self._update_history(conversation_id, user=query, assistant="Sorry, no relevant information found.")
#             yield "Sorry, no relevant information found."
#             return

#         chunks = []
#         seen_chunks = set()
#         total_chars = 0
#         for r in search_results:
#             c = r.get("text_chunk")
#             if not c or c in seen_chunks:
#                 continue
#             seen_chunks.add(c)
#             chunks.append(c)
#             total_chars += len(c)
#             if total_chars > 4000:
#                 break

#         context = "\n\n".join(chunks)
#         history = self._conversations.get(conversation_id or "", [])
#         prompt = self._build_prompt(query=query, context=context, history=history)

#         final_text = ""
#         try:
#             for piece in self.gemini_client.stream_response(prompt):
#                 final_text += piece
#                 yield piece
#         except Exception as e:
#             logger.exception("LLM streaming failed: %s", e)
#             yield "\n[Stream error: failed to generate response]"
#         finally:
#             if final_text:
#                 self._update_history(conversation_id, user=query, assistant=final_text)


# from services.utils.qdrant_storage import QdrantStorage
# from services.gemini_api import GeminiClient
# from api.travel_api import TravelAPIClient
# from services.utils.redis_client import redis_client
# import logging
# from typing import Dict, List, Optional, Generator

# logger = logging.getLogger("RAGPipeline")
# if not logger.handlers:
#     h = logging.StreamHandler()
#     h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
#     logger.addHandler(h)
# logger.setLevel(logging.INFO)


# class RAGPipeline:
#     def __init__(self):
#         self.search = QdrantStorage()
#         self.gemini_client = GeminiClient()
#         self.travel_client = TravelAPIClient()
#         self._conversations: Dict[str, List[Dict[str, str]]] = {}
#         self._history_max_messages = 6

#     # ---------------------------
#     # 🧠 Build Prompt
#     # ---------------------------
#     def _build_prompt(self, query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
#         history_text = ""
#         if history:
#             trimmed = history[-self._history_max_messages:]
#             lines = [f"{m['role']}: {m['content']}" for m in trimmed]
#             history_text = "\n".join(lines)

#         prompt = (
#             "You are an assistant that answers questions using the provided context. "
#             "Be concise and return useful, accurate information based on the context. "
#             "If the answer is not contained in the context, say you don't know.\n\n"
#         )
#         if history_text:
#             prompt += f"Conversation so far:\n{history_text}\n\n"
#         prompt += f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
#         return prompt

#     # ---------------------------
#     # 🧠 Update Conversation History
#     # ---------------------------
#     def _update_history(self, conversation_id: Optional[str], user: str, assistant: Optional[str] = None):
#         if not conversation_id:
#             return
#         convo = self._conversations.setdefault(conversation_id, [])
#         if user:
#             convo.append({"role": "user", "content": user})
#         if assistant:
#             convo.append({"role": "assistant", "content": assistant})
#         if len(convo) > 50:
#             del convo[: len(convo) - 50]

#     # ---------------------------
#     # ✈️ Handle Travel Queries
#     # ---------------------------
#     def _handle_travel_query(self, query: str) -> Optional[Dict]:
#         q = query.lower()
#         if "flight" in q or "travel" in q or "trip" in q:
#             try:
#                 origin, destination, date = "DEL", "BOM", "2025-10-15"
#                 flights = self.travel_client.search_flights(origin, destination, date)
#                 return {
#                     "response": f"Here are flight options from {origin} to {destination} on {date}: {flights}",
#                     "citations": []
#                 }
#             except Exception as e:
#                 logger.exception("Travel API failed: %s", e)
#                 return {"response": "Sorry — Travel API request failed.", "citations": []}
#         return None

#     # ---------------------------
#     # 🧩 Main Query Handler with Redis Caching
#     # ---------------------------
#     def answer_query(self, query: str, conversation_id: Optional[str] = None) -> Dict:
#         logger.info("RAG: answering query (truncated): %s", (query or "")[:120])

#         # 1️⃣ Travel-related queries
#         travel_resp = self._handle_travel_query(query)
#         if travel_resp:
#             self._update_history(conversation_id, user=query, assistant=travel_resp["response"])
#             return travel_resp

#         # 2️⃣ Create cache key
#         cache_key = f"query_cache:{conversation_id or 'global'}:{query.strip().lower()}"

#         # 3️⃣ Check if response is already cached
#         cached_response = redis_client.get(cache_key)
#         if cached_response:
#             logger.info("⚡ Using cached response from Redis")
#             return {"response": cached_response, "citations": []}

#         # 4️⃣ Search Qdrant
#         try:
#             search_results = self.search.search(query_text=query, top_k=5, file_id=None)
#         except Exception as e:
#             logger.exception("DB.search raised: %s", e)
#             return {"response": "Sorry — internal error while searching the knowledge base."}

#         if not search_results:
#             self._update_history(conversation_id, user=query, assistant="Sorry, no relevant information found.")
#             return {"response": "Sorry, no relevant information found.", "citations": []}

#         # 5️⃣ Build context for LLM
#         chunks = []
#         citations = []
#         seen_chunks = set()
#         total_chars = 0
#         for r in search_results:
#             c = r.get("text_chunk")
#             if not c or c in seen_chunks:
#                 continue
#             seen_chunks.add(c)
#             chunks.append(c)
#             src = r.get("source")
#             page = r.get("page")
#             if src or page:
#                 citations.append({"source": src, "page": page})
#             total_chars += len(c)
#             if total_chars > 4000:
#                 break

#         context = "\n\n".join(chunks)
#         history = self._conversations.get(conversation_id or "", [])
#         prompt = self._build_prompt(query=query, context=context, history=history)

#         # 6️⃣ Generate LLM response
#         try:
#             response_text = self.gemini_client.generate_response(prompt)
#         except Exception as e:
#             logger.exception("LLM generation failed: %s", e)
#             response_text = "Sorry — failed to generate an answer from the LLM."

#         # 7️⃣ Cache the new response for 1 hour (3600 seconds)
#         try:
#             redis_client.setex(cache_key, 3600, response_text)
#             logger.info(f"✅ Cached response in Redis for key: {cache_key}")
#         except Exception as e:
#             logger.warning(f"Redis caching failed: {e}")

#         self._update_history(conversation_id, user=query, assistant=response_text)

#         # 8️⃣ Deduplicate citations
#         deduped = []
#         seen = set()
#         for c in citations:
#             key = (c.get("source"), c.get("page"))
#             if key not in seen:
#                 seen.add(key)
#                 deduped.append(c)

#         return {"response": response_text, "citations": deduped}

#     # ---------------------------
#     # 🔄 Streaming Responses
#     # ---------------------------
#     def stream_answer(self, query: str, conversation_id: Optional[str] = None) -> Generator[str, None, None]:
#         logger.info("RAG: streaming query (truncated): %s", (query or "")[:120])

#         if "flight" in query.lower():
#             yield "Streaming not supported for travel queries. Please use non-stream mode."
#             return

#         try:
#             search_results = self.search.search(query_text=query, top_k=5, file_id=None)
#         except Exception as e:
#             logger.exception("DB.search raised: %s", e)
#             yield "Sorry — internal error while searching the knowledge base."
#             return

#         if not search_results:
#             self._update_history(conversation_id, user=query, assistant="Sorry, no relevant information found.")
#             yield "Sorry, no relevant information found."
#             return

#         chunks = []
#         seen_chunks = set()
#         total_chars = 0
#         for r in search_results:
#             c = r.get("text_chunk")
#             if not c or c in seen_chunks:
#                 continue
#             seen_chunks.add(c)
#             chunks.append(c)
#             total_chars += len(c)
#             if total_chars > 4000:
#                 break

#         context = "\n\n".join(chunks)
#         history = self._conversations.get(conversation_id or "", [])
#         prompt = self._build_prompt(query=query, context=context, history=history)

#         final_text = ""
#         try:
#             for piece in self.gemini_client.stream_response(prompt):
#                 final_text += piece
#                 yield piece
#         except Exception as e:
#             logger.exception("LLM streaming failed: %s", e)
#             yield "\n[Stream error: failed to generate response]"
#         finally:
#             if final_text:
#                 self._update_history(conversation_id, user=query, assistant=final_text)


from services.utils.qdrant_storage import QdrantStorage
from services.gemini_api import GeminiClient
from api.travel_api import TravelAPIClient
from services.utils.redis_client import redis_client
import logging
from typing import Dict, List, Optional, Generator, Any
import json

logger = logging.getLogger("RAGPipeline")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

class RAGPipeline:
    def __init__(self):
        self.search = QdrantStorage()
        self.gemini_client = GeminiClient()
        self.travel_client = TravelAPIClient()
        self._conversations: Dict[str, List[Dict[str, str]]] = {}
        self._history_max_messages = 6

    def _build_prompt(self, query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build the prompt for the LLM with context and conversation history."""
        history_text = ""
        if history:
            trimmed = history[-self._history_max_messages:]
            lines = [f"{m['role']}: {m['content']}" for m in trimmed if m.get('content')]
            history_text = "\n".join(lines)

        prompt = (
            "You are an assistant that answers questions using the provided context. "
            "Be concise and return useful, accurate information based on the context. "
            "If the answer is not contained in the context, say you don't know.\n\n"
        )
        
        if history_text:
            prompt += f"Conversation so far:\n{history_text}\n\n"
            
        prompt += f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        logger.debug(f"Built prompt: {prompt[:500]}...")  # Log first 500 chars
        return prompt

    def _update_history(self, conversation_id: Optional[str], user: str, assistant: Optional[str] = None) -> None:
        """Update conversation history if conversation_id is provided."""
        if not conversation_id or not user:
            return
            
        convo = self._conversations.setdefault(conversation_id, [])
        if user:
            convo.append({"role": "user", "content": user})
        if assistant:
            convo.append({"role": "assistant", "content": assistant})
        
        # Keep only the most recent messages
        if len(convo) > 50:
            del convo[:len(convo) - 50]

    def _handle_travel_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle travel-related queries using the travel API."""
        q = query.lower()
        if any(keyword in q for keyword in ["flight", "travel", "trip", "book"]):
            try:
                origin, destination, date = "DEL", "BOM", "2025-10-15"
                flights = self.travel_client.search_flights(origin, destination, date)
                return {
                    "response": f"Here are flight options from {origin} to {destination} on {date}: {flights}",
                    "citations": []
                }
            except Exception as e:
                logger.exception("Travel API failed: %s", e)
                return {"response": "Sorry — Travel API request failed.", "citations": []}
        return None

    def _process_search_results(self, search_results: List[Dict]) -> tuple[str, List[Dict]]:
        """Process search results into context and citations."""
        chunks = []
        citations = []
        seen_chunks = set()
        total_chars = 0
        
        for r in search_results:
            c = r.get("text_chunk")
            if not c or not c.strip() or c in seen_chunks:
                continue
                
            seen_chunks.add(c)
            chunks.append(c)
            
            src = r.get("source")
            page = r.get("page")
            if src or page:
                citations.append({"source": src, "page": page})
                
            total_chars += len(c)
            if total_chars > 4000:
                break

        return "\n\n".join(chunks) if chunks else "", citations

    def answer_query(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query and return a response with citations."""
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Invalid or empty query received")
            return {"response": "Please provide a valid query.", "citations": []}

        logger.info("Processing query: %s", query[:100] + ("..." if len(query) > 100 else ""))

        # 1. Handle travel queries
        travel_resp = self._handle_travel_query(query)
        if travel_resp:
            self._update_history(conversation_id, user=query, assistant=travel_resp["response"])
            return travel_resp

        # 2. Check cache
        cache_key = f"query_cache:{conversation_id or 'global'}:{query.strip().lower()}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info("Using cached response")
                return {"response": cached, "citations": []}
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")

        # 3. Search Qdrant
        try:
            search_results = self.search.search(query_text=query, top_k=5, file_id=None)
            logger.info(f"Found {len(search_results)} search results")
        except Exception as e:
            logger.exception("Search failed")
            return {"response": "Error searching knowledge base.", "citations": []}

        if not search_results:
            return {"response": "No relevant information found.", "citations": []}

        # 4. Process results and generate response
        context, citations = self._process_search_results(search_results)
        if not context:
            return {"response": "No valid content found in documents.", "citations": []}

        history = self._conversations.get(conversation_id or "", [])
        prompt = self._build_prompt(query=query, context=context, history=history)

        # 5. Generate LLM response
        try:
            response_text = self.gemini_client.generate_response(prompt)
            if not response_text or not response_text.strip():
                raise ValueError("Empty response from LLM")
        except Exception as e:
            logger.exception("LLM generation failed")
            response_text = "Sorry, I couldn't generate a response. Please try again."

        # 6. Cache the response
        try:
            redis_client.setex(cache_key, 3600, response_text)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

        # 7. Update history
        self._update_history(conversation_id, user=query, assistant=response_text)

        # 8. Deduplicate citations
        seen = set()
        deduped = []
        for c in citations:
            key = (c.get("source"), c.get("page"))
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        return {"response": response_text, "citations": deduped}

    def stream_answer(self, query: str, conversation_id: Optional[str] = None) -> Generator[str, None, None]:
        """Stream the response for a query."""
        logger.info("Streaming response for query: %s", query[:100] + ("..." if len(query) > 100 else ""))

        if any(keyword in query.lower() for keyword in ["flight", "travel", "trip"]):
            yield "Streaming not available for travel queries. Please use non-stream mode."
            return

        try:
            # Get search results
            search_results = self.search.search(query_text=query, top_k=5, file_id=None)
            if not search_results:
                yield "No relevant information found."
                return

            # Process search results
            context, _ = self._process_search_results(search_results)
            if not context:
                yield "No valid content found in documents."
                return

            # Build prompt and stream response
            history = self._conversations.get(conversation_id or "", [])
            prompt = self._build_prompt(query=query, context=context, history=history)
            
            final_text = ""
            try:
                for piece in self.gemini_client.stream_response(prompt):
                    final_text += piece
                    yield piece
            except Exception as e:
                logger.exception("Streaming failed")
                yield "\n[Error generating response]"
            finally:
                if final_text:
                    self._update_history(conversation_id, user=query, assistant=final_text)

        except Exception as e:
            logger.exception("Error in stream_answer")
            yield "Sorry, an error occurred while processing your request."
