from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config.settings import Settings


class CollectionManager:
    def __init__(self, client: QdrantClient, settings: Settings):
        self._client = client
        self._settings = settings

    METADATA_INDEX_FIELDS = [
        "metadata.header_1",
        "metadata.header_2",
        "metadata.header_3",
    ]

    def recreate(self, embedding_dimension: int) -> None:
        self.delete()
        self._create(embedding_dimension)
        self._create_payload_indexes()

    def exists(self) -> bool:
        collections = self._client.get_collections().collections
        names = [collection.name for collection in collections]
        return self._settings.collection_name in names

    def delete(self) -> None:
        if not self.exists():
            return
        self._client.delete_collection(self._settings.collection_name)

    def _create(self, dimension: int) -> None:
        self._client.create_collection(
            collection_name=self._settings.collection_name,
            vectors_config={
                self._settings.dense_field: models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self._settings.sparse_field: models.SparseVectorParams(
                    index=models.SparseIndexParams()
                )
            },
        )

    def _create_payload_indexes(self) -> None:
        for field in self.METADATA_INDEX_FIELDS:
            self._client.create_payload_index(
                collection_name=self._settings.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
