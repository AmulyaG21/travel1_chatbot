from pydantic_settings import BaseSettings  # Updated import

class Settings(BaseSettings):
    QDRANT_URL: str
    QDRANT_API_KEY: str
    EMBEDDING_MODEL: str
    GEMINI_API_KEY: str
    QDRANT_FORCE_REST: bool = False

    # Single default collection for the whole app
    QDRANT_COLLECTION: str = "travel1_chatbot"

    class Config:
        env_file = ".env"

settings = Settings()
