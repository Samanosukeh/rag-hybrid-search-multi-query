from fastembed import SparseTextEmbedding
from qdrant_client.http import models


class SparseEmbedder:
    DEFAULT_MODEL = "Qdrant/bm25"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed(self, text: str) -> models.SparseVector:
        result = list(self._model.query_embed(text))[0]
        return models.SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )
