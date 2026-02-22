from typing import Dict, Iterator

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]


class TextChunk:
    def __init__(self, content: str, index: int, metadata: Dict[str, str]):
        self.content = content
        self.index = index
        self.metadata = metadata

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"TextChunk(index={self.index}, metadata={self.metadata}, preview='{preview}...')"


class Chunks:
    def __init__(self, items: list):
        self._items = items

    def __iter__(self) -> Iterator[TextChunk]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0


class TextChunker:
    def __init__(self, chunk_size: int, overlap: int):
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

    def chunk(self, text: str) -> Chunks:
        header_splits = self._header_splitter.split_text(text)
        sized_splits = self._text_splitter.split_documents(header_splits)
        items = [
            TextChunk(
                content=doc.page_content,
                index=index,
                metadata=dict(doc.metadata),
            )
            for index, doc in enumerate(sized_splits)
        ]
        return Chunks(items)
