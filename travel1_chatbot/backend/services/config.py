from pydantic_settings import BaseSettings  # Updated import

class Settings(BaseSettings):
    QDRANT_URL: str
    QDRANT_API_KEY: str
    EMBEDDING_MODEL: str
    GEMINI_API_KEY: str
    TRAVEL_API_KEY: str
    TRAVEL_API_URL: str
    
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_URL: str
    REDIS_DB: int = 0
    REDIS_USERNAME:str

    QDRANT_FORCE_REST: bool = False

    # Single default collection for the whole app
    QDRANT_COLLECTION: str = "travel1_chatbot"

    class Config:
        env_file = ".env"

settings = Settings()
