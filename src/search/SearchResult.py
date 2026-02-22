from typing import List


class SearchResult:
    def __init__(self, content: str, score: float, chunk_index: int):
        self.content = content
        self.score = score
        self.chunk_index = chunk_index

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"SearchResult(score={self.score:.4f}, chunk={self.chunk_index}, text='{preview}...')"


class SearchResults:
    def __init__(self, dense: List[SearchResult], sparse: List[SearchResult]):
        self.dense = dense
        self.sparse = sparse

    def all_unique_contents(self) -> List[str]:
        seen = set()
        unique = []
        for result in self.dense + self.sparse:
            if result.content not in seen:
                seen.add(result.content)
                unique.append(result.content)
        return unique

    def total_hits(self) -> int:
        return len(self.dense) + len(self.sparse)

    def __repr__(self) -> str:
        return f"SearchResults(dense={len(self.dense)}, sparse={len(self.sparse)})"
