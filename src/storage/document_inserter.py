import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config.settings import Settings
from src.document.text_chunker import Chunks, TextChunk
from src.embeddings.dense_embedder import DenseEmbedder
from src.embeddings.sparse_embedder import SparseEmbedder


class DocumentInserter:
    def __init__(
        self,
        client: QdrantClient,
        settings: Settings,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
    ):
        self._client = client
        self._settings = settings
        self._dense_embedder = dense_embedder
        self._sparse_embedder = sparse_embedder

    def insert(self, chunks: Chunks) -> int:
        points = self._build_points(chunks)
        self._upsert(points)
        return len(points)

    def _build_points(self, chunks: Chunks) -> List[models.PointStruct]:
        return [self._build_point(chunk) for chunk in chunks]

    def _build_point(self, chunk: TextChunk) -> models.PointStruct:
        dense_vector = self._dense_embedder.embed(chunk.content)
        sparse_vector = self._sparse_embedder.embed(chunk.content)

        return models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                self._settings.dense_field: dense_vector,
                self._settings.sparse_field: sparse_vector,
            },
            payload={
                self._settings.content_field: chunk.content,
                "chunk_index": chunk.index,
                "metadata": chunk.metadata,
            },
        )

    def _upsert(self, points: List[models.PointStruct]) -> None:
        self._client.upsert(
            collection_name=self._settings.collection_name,
            wait=True,
            points=points,
        )
