import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
        self.model_embeddings: str = os.getenv("MODEL_EMBEDDINGS", "text-embedding-3-small")
        self.mistral_model: str = os.getenv("MISTRAL_MODEL_NAME", "mistral-large-latest")
        self.vector_db_url: str = os.getenv("VECTOR_DB_URL", "http://localhost:6333")
        self.vector_db_api_key: str = os.getenv("VECTOR_DB_API_KEY", "")
        self.collection_name: str = os.getenv("VECTOR_DB_COLLECTION", "rag_collection")
        self.content_field: str = os.getenv("VECTOR_FIELD_CONTENT_NAME", "text")
        self.dense_field: str = "text-dense"
        self.sparse_field: str = "text-sparse"
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
