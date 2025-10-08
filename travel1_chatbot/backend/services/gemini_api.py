# import os
# import google.generativeai as genai
# from dotenv import load_dotenv
# from pathlib import Path
# import logging
# from typing import Optional, Generator
# import sys

# # Configure logging
# logger = logging.getLogger("GeminiClient")
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
#     logger.addHandler(handler)
# logger.setLevel(logging.INFO)

# # Load environment variables
# ENV_PATH = Path(__file__).parent.parent / '.env'
# if ENV_PATH.exists():
#     load_dotenv(ENV_PATH, override=True)
#     logger.info(f"✅ Loaded .env from {ENV_PATH}")
# else:
#     logger.warning(f"⚠️ .env file not found at {ENV_PATH}")

# # Get API key
# _GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip('"\'')
# if not _GEMINI_API_KEY:
#     logger.error("❌ GEMINI_API_KEY not found in environment variables")
# else:
#     logger.info(f"✅ GEMINI_API_KEY loaded (length: {len(_GEMINI_API_KEY)})")
#     try:
#         genai.configure(api_key=_GEMINI_API_KEY)
#         logger.info("✅ Successfully configured Google Generative AI")
#     except Exception as e:
#         logger.error(f"❌ Failed to configure Google Generative AI: {e}")

# # Import SYSTEM_PROMPT
# try:
#     from services.utils.prompts import SYSTEM_PROMPT
#     logger.info("✅ Successfully imported SYSTEM_PROMPT")
# except ImportError as e:
#     logger.error(f"❌ Failed to import SYSTEM_PROMPT: {e}")
#     SYSTEM_PROMPT = "You are a helpful AI assistant."

# # Create model instance
# _MODEL = None
# if _GEMINI_API_KEY:
#     try:
#         _MODEL = genai.GenerativeModel(
#             "gemini-2.0-flash",
#             system_instruction=SYSTEM_PROMPT
#         )
#         logger.info("✅ Successfully initialized Gemini model")
#     except Exception as e:
#         logger.error(f"❌ Failed to initialize Gemini model: {e}")



import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import logging
from typing import Optional, Generator
import sys

# Configure logging
logger = logging.getLogger("GeminiClient")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Load environment variables
ENV_PATH = Path(__file__).parent.parent / '.env'
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)
    logger.info(f"✅ Loaded .env from {ENV_PATH}")
else:
    logger.warning(f"⚠️ .env file not found at {ENV_PATH}")

# Get API key
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip('"\'')
if not _GEMINI_API_KEY:
    logger.error("❌ GEMINI_API_KEY not found in environment variables")
else:
    logger.info(f"✅ GEMINI_API_KEY loaded (length: {len(_GEMINI_API_KEY)})")
    try:
        genai.configure(api_key=_GEMINI_API_KEY)
        logger.info("✅ Successfully configured Google Generative AI")
    except Exception as e:
        logger.error(f"❌ Failed to configure Google Generative AI: {e}")

# Import SYSTEM_PROMPT
try:
    from services.utils.prompts import SYSTEM_PROMPT
    logger.info("✅ Successfully imported SYSTEM_PROMPT")
except ImportError as e:
    logger.error(f"❌ Failed to import SYSTEM_PROMPT: {e}")
    SYSTEM_PROMPT = "You are a helpful AI assistant."

# Create model instance
_MODEL = None
if _GEMINI_API_KEY:
    try:
        _MODEL = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT
        )
        logger.info("✅ Successfully initialized Gemini model")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini model: {e}")

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional API key override."""
        self.api_key = api_key or _GEMINI_API_KEY
        self.model = None
        
        if not self.api_key:
            logger.error("❌ No API key available")
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = _MODEL or genai.GenerativeModel(
                "gemini-2.0-flash",
                system_instruction=SYSTEM_PROMPT
            )
            logger.info("✅ Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            raise

    def query_document(self, query: str, document_text: str) -> str:
        """
        Query a document and return a response based SOLELY on the document content.
        
        Args:
            query: The user's question about the document
            document_text: The text content of the uploaded document
            
        Returns:
            str: Response based strictly on the document content
        """
        if not hasattr(self, 'model') or self.model is None:
            return "Error: Model not initialized"
            
        try:
            # Clean and validate document text
            document_text = document_text.strip()
            if not document_text:
                return "Error: The uploaded document appears to be empty."
                
            # Truncate if too long (keep most relevant part)
            max_doc_length = 15000  # Reduced to leave room for prompt
            if len(document_text) > max_doc_length:
                document_text = f"...{document_text[-max_doc_length:]}"
                logger.warning("Document was truncated to fit context window")
            
            # Strict instruction to only use the provided document
            prompt = f"""You are a precise document analysis tool. Your ONLY task is to answer questions using EXACTLY and ONLY the information provided in the document below.

DOCUMENT CONTENT:
{document_text}

INSTRUCTIONS:
1. Read the document content carefully.
2. Answer the question using ONLY the information from the document.
3. If the document doesn't contain the answer, respond with: "The document does not contain information about this."
4. Do not use any external knowledge or make assumptions.
5. If the question is not related to the document, respond with: "This appears unrelated to the document content."
6. Keep your response concise and directly based on the document.

QUESTION: {query}

YOUR RESPONSE (ONLY FROM DOCUMENT):"""
            
            logger.info(f"Processing query with document (length: {len(document_text)} chars)")
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Very low for factual accuracy
                    top_p=0.9,
                    top_k=1,  # More deterministic
                    max_output_tokens=1024
                )
            )
            
            result = response.text.strip() if response.text else "No response from model"
            
            # Post-process to ensure no hallucinations
            if not any(word in result.lower() for word in ["document", "does not contain", "unrelated", "not mentioned"]):
                if not any(word in query.lower() for word in result.lower().split()):
                    result = "The document does not contain specific information about this query."
            
            return result
                
        except Exception as e:
            logger.error(f"Document query error: {str(e)}")
            return f"Error processing your request: {str(e)}"

    def generate_response(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        if not hasattr(self, 'model') or self.model is None:
            logger.error("❌ Gemini model is not initialized")
            return "Error: Model not initialized"
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048
                )
            )
            return response.text or "No response from model"
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return f"Error: {str(e)}"

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """Stream response from the model."""
        if not hasattr(self, 'model') or self.model is None:
            yield "Error: Model not initialized"
            return
            
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048
                ),
                stream=True
            )
            
            for chunk in response:
                if hasattr(chunk, 'text'):
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"❌ Error in stream_response: {e}")
            yield f"Error: {str(e)}"

