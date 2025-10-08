from backend.database import QdrantDB
from gemini_api import GeminiClient
import logging
from typing import Dict, List, Optional, Generator

logger = logging.getLogger("RAGPipeline")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class RAGPipeline:
    def __init__(self):
        self.db = QdrantDB()
        self.gemini_client = GeminiClient()
        # Simple in-memory conversation history per conversation_id
        # {conversation_id: [{"role":"user"|"assistant","content": str}, ...]}
        self._conversations: Dict[str, List[Dict[str, str]]] = {}
        # How many prior messages to include
        self._history_max_messages = 6

    def _build_prompt(self, query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        history_text = ""
        if history:
            # Include only recent messages for brevity
            trimmed = history[-self._history_max_messages :]
            lines = []
            for turn in trimmed:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                lines.append(f"{role}: {content}")
            history_text = "\n".join(lines)

        prompt = (
            "You are an assistant that answers questions using the provided context. "
            "Be concise and return useful, accurate information based on the context. "
            "If the answer is not contained in the context, say you don't know.\n\n"
        )
        if history_text:
            prompt += f"Conversation so far:\n{history_text}\n\n"
        prompt += (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return prompt

    def _update_history(self, conversation_id: Optional[str], user: str, assistant: Optional[str] = None):
        if not conversation_id:
            return
        convo = self._conversations.setdefault(conversation_id, [])
        if user:
            convo.append({"role": "user", "content": user})
        if assistant:
            convo.append({"role": "assistant", "content": assistant})
        # Trim
        if len(convo) > 50:
            del convo[: len(convo) - 50]

    def answer_query(self, query: str, conversation_id: Optional[str] = None) -> Dict:
        logger.info("RAG: answering query (truncated): %s", (query or "")[:120])
        try:
            # DB will encode the query text and run robust search
            search_results = self.db.search(query_text=query, top_k=5, file_id=None)
        except Exception as e:
            logger.exception("DB.search raised: %s", e)
            return {"response": "Sorry — internal error while searching the knowledge base."}

        if not search_results:
            self._update_history(conversation_id, user=query, assistant="Sorry, no relevant information found.")
            return {"response": "Sorry, no relevant information found.", "citations": []}

        # Build context, dedupe, and cap length to keep prompt reasonable
        chunks = []
        citations = []
        seen_chunks = set()
        total_chars = 0
        for r in search_results:
            c = r.get("text_chunk")
            if not c or c in seen_chunks:
                continue
            seen_chunks.add(c)
            chunks.append(c)
            # capture citation info if available
            src = r.get("source")
            page = r.get("page")
            if src or page:
                citations.append({"source": src, "page": page})
            total_chars += len(c)
            if total_chars > 4000:
                break

        context = "\n\n".join(chunks)
        history = self._conversations.get(conversation_id or "", [])
        prompt = self._build_prompt(query=query, context=context, history=history)

        try:
            response_text = self.gemini_client.generate_response(prompt)
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            response_text = "Sorry — failed to generate an answer from the LLM."

        self._update_history(conversation_id, user=query, assistant=response_text)
        # Deduplicate citations and keep order
        deduped = []
        seen = set()
        for c in citations:
            key = (c.get("source"), c.get("page"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        return {"response": response_text, "citations": deduped}

    def stream_answer(self, query: str, conversation_id: Optional[str] = None) -> Generator[str, None, None]:
        """Stream only the LLM response text. Caller can fetch citations separately by calling a non-stream method first.
        """
        logger.info("RAG: streaming query (truncated): %s", (query or "")[:120])
        try:
            search_results = self.db.search(query_text=query, top_k=5, file_id=None)
        except Exception as e:
            logger.exception("DB.search raised: %s", e)
            yield "Sorry — internal error while searching the knowledge base."
            return

        if not search_results:
            self._update_history(conversation_id, user=query, assistant="Sorry, no relevant information found.")
            yield "Sorry, no relevant information found."
            return

        chunks = []
        seen_chunks = set()
        total_chars = 0
        for r in search_results:
            c = r.get("text_chunk")
            if not c or c in seen_chunks:
                continue
            seen_chunks.add(c)
            chunks.append(c)
            total_chars += len(c)
            if total_chars > 4000:
                break

        context = "\n\n".join(chunks)
        history = self._conversations.get(conversation_id or "", [])
        prompt = self._build_prompt(query=query, context=context, history=history)

        final_text = ""
        try:
            for piece in self.gemini_client.stream_response(prompt):
                final_text += piece
                yield piece
        except Exception as e:
            logger.exception("LLM streaming failed: %s", e)
            yield "\n[Stream error: failed to generate response]"
        finally:
            if final_text:
                self._update_history(conversation_id, user=query, assistant=final_text)

