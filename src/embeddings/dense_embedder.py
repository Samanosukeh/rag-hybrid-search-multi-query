from typing import List

from langchain_openai import OpenAIEmbeddings


class DenseEmbedder:
    def __init__(self, model: str, api_key: str):
        self._embeddings = OpenAIEmbeddings(model=model, api_key=api_key)

    def embed(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    def dimension(self) -> int:
        sample = self._embeddings.embed_query("dimension probe")
        return len(sample)
