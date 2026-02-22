from typing import Any, Dict, List, Optional

from pydantic import SecretStr

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client.http import models

from src.config.settings import Settings


class HybridSearcher:
    def __init__(self, settings: Settings, limit: int = 3):
        self._store = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(model=settings.model_embeddings, api_key=SecretStr(settings.openai_api_key)),
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            collection_name=settings.collection_name,
            url=settings.vector_db_url,
            api_key=settings.vector_db_api_key,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=settings.dense_field,
            sparse_vector_name=settings.sparse_field,
            content_payload_key=settings.content_field,
            validate_collection_config=False,
        )
        self._limit = limit

    def as_retriever(self, metadata: Optional[Dict[str, str]] = None) -> BaseRetriever:
        search_kwargs: Dict[str, Any] = {"k": self._limit}
        if metadata:
            search_kwargs["filter"] = self._build_filter(metadata)
        return self._store.as_retriever(search_kwargs=search_kwargs)

    def search(self, query: str, metadata: Optional[Dict[str, str]] = None) -> List[Document]:
        qdrant_filter = self._build_filter(metadata) if metadata else None
        return self._store.similarity_search(query, k=self._limit, filter=qdrant_filter)

    @staticmethod
    def _build_filter(metadata: Dict[str, str]) -> models.Filter:
        conditions: List[models.Condition] = [
            models.FieldCondition(
                key=f"metadata.{field}",
                match=models.MatchValue(value=value),
            )
            for field, value in metadata.items()
        ]
        return models.Filter(must=conditions)
