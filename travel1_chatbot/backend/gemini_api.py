import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
from prompts import SYSTEM_PROMPT
import logging
from typing import Optional, Generator

logger = logging.getLogger("GeminiClient")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Load environment (optional; useful in dev)
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not _GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is empty. Gemini calls will fail without a valid key.")

genai.configure(api_key=_GEMINI_API_KEY)

# create a configured model object once (system prompt injected)
try:
    _MODEL = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT
    )
except Exception as e:
    _MODEL = None
    logger.warning("Could not initialize Gemini model object at import time: %s", e)


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        # If user passed an api_key to constructor, reconfigure (optional)
        if api_key:
            genai.configure(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        """
        Synchronously call Gemini generate_content. Returns the generated text or an error message.
        Replace generation_config properties as you like.
        """
        if _MODEL is None:
            logger.error("Gemini model object is not initialized.")
            return "LLM service is not configured."

        try:
            # Note: genai .generate_content accepts either a structured prompt or a plain string depending on SDK version.
            # Here we pass the prompt string; adjust if your SDK version needs a different shape.
            result = _MODEL.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048
                )
            )
            # result may be model-specific; take `.text` which is common in genai SDK responses
            text = getattr(result, "text", None)
            if text is None and isinstance(result, dict):
                # some SDK versions return dict-like
                text = result.get("candidates", [{}])[0].get("content", "")
            return text or "Gemini returned an empty response."
        except Exception as e:
            logger.exception("GeminiClient.generate_response error: %s", e)
            return "Sorry — the LLM service failed to generate a response."

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """Yield the response text in chunks (streaming)."""
        if _MODEL is None:
            yield "LLM service is not configured."
            return
        try:
            for chunk in _MODEL.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048
                ),
                stream=True,
            ):
                # Some SDK versions expose .text per chunk
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as e:
            logger.exception("GeminiClient.stream_response error: %s", e)
            yield "\n[Stream error: LLM failed]"
