from sentence_transformers import SentenceTransformer
from config import settings


embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def generate_embedding(self, text: str):
        return self.model.encode(text).tolist()



