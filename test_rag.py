import unittest

from qdrant_client import QdrantClient

from src.config.Settings import Settings
from src.document.DocumentLoader import DocumentLoader
from src.document.TextChunker import TextChunker
from src.embeddings.DenseEmbedder import DenseEmbedder
from src.embeddings.SparseEmbedder import SparseEmbedder
from src.agent.RAGAgent import RAGAgent
from src.generation.RAGChain import RAGChain
from src.search.HybridSearcher import HybridSearcher
from src.storage.CollectionManager import CollectionManager
from src.storage.DocumentInserter import DocumentInserter


class TestRAGPipeline(unittest.TestCase):
    """
    End-to-end tests for the Hybrid RAG pipeline.

    Tests are prefixed with numbers to guarantee execution order:
      01 → create_collection
      02 → insert_documents
      03 → hybrid search (LangChain retriever)
      04 → hybrid search with metadata filter
      05 → RAG chain (Mistral)
      06 → RAG agent (Mistral tool-calling)
      07 → delete_collection
    """

    settings = Settings()
    client = QdrantClient(url=settings.vector_db_url, api_key=settings.vector_db_api_key)
    dense_embedder = DenseEmbedder(
        model=settings.model_embeddings,
        api_key=settings.openai_api_key,
    )
    sparse_embedder = SparseEmbedder()
    collection_manager = CollectionManager(client=client, settings=settings)
    document_inserter = DocumentInserter(
        client=client,
        settings=settings,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )
    searcher = HybridSearcher(settings=settings, limit=3)
    rag_chain = RAGChain(searcher=searcher, settings=settings)
    rag_agent = RAGAgent(searcher=searcher, settings=settings)

    # ------------------------------------------------------------------ #
    #  Test 01 – Create the Qdrant collection                             #
    # ------------------------------------------------------------------ #

    def test_01_create_collection(self):
        """Creates (or recreates) the Qdrant collection with the correct schema."""
        dimension = self.dense_embedder.dimension()
        self.collection_manager.recreate(embedding_dimension=dimension)

        self.assertTrue(
            self.collection_manager.exists(),
            msg=f"Collection '{self.settings.collection_name}' was not found after creation.",
        )


    # ------------------------------------------------------------------ #
    #  Test 02 – Insert documents from doc.txt                            #
    # ------------------------------------------------------------------ #

    def test_02_insert_documents(self):
        """Loads doc.txt, splits it into chunks, and upserts them into Qdrant."""
        loader = DocumentLoader(file_path="doc.md")
        chunker = TextChunker(
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )

        text = loader.load()
        chunks = chunker.chunk(text)

        self.assertFalse(chunks.is_empty(), msg="doc.txt produced zero chunks – check the file.")

        inserted = self.document_inserter.insert(chunks)
        point_count = self.client.count(collection_name=self.settings.collection_name).count

        print(f"\n[OK] Inserted {inserted} chunk(s) – collection now holds {point_count} point(s).")

    # ------------------------------------------------------------------ #
    #  Test 03 – Hybrid search via LangChain retriever                   #
    # ------------------------------------------------------------------ #

    def test_03_search_returns_results(self):
        """Runs a hybrid search via LangChain QdrantVectorStore and asserts hits."""
        query = "What is Retrieval-Augmented Generation?"
        documents = self.searcher.search(query)

        self.assertGreater(
            len(documents), 0,
            msg="No search results returned – collection may be empty.",
        )

        print(f"\n[OK] Query: '{query}'")
        print(f"     Documents returned: {len(documents)}")
        print()

        for idx, doc in enumerate(documents, start=1):
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  [{idx}] {preview}...")

    # ------------------------------------------------------------------ #
    #  Test 04 – Hybrid search with metadata filter                       #
    # ------------------------------------------------------------------ #

    def test_04_search_with_metadata_filter(self):
        """Runs a hybrid search filtered by header metadata."""
        query = "How does it work?"
        metadata_filter = {"header_2": "What is Hybrid Search?"}
        documents = self.searcher.search(query, metadata=metadata_filter)

        self.assertGreater(
            len(documents), 0,
            msg="No results returned with metadata filter.",
        )

        print(f"\n[OK] Query: '{query}' | Filter: {metadata_filter}")
        print(f"     Documents returned: {len(documents)}")
        print()

        for idx, doc in enumerate(documents, start=1):
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  [{idx}] {preview}...")

    # ------------------------------------------------------------------ #
    #  Test 05 – Full RAG: retrieve + generate via Mistral                #
    # ------------------------------------------------------------------ #

    def test_05_rag_generates_answer(self):
        """Sends a question through the full RAG chain and asserts a non-empty answer."""
        query = "What is Retrieval-Augmented Generation?"
        answer = self.rag_chain.invoke(query)

        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0, msg="RAG chain returned an empty answer.")

        print(f"\n[OK] Query: '{query}'")
        print(f"     Answer:\n{answer}")


    # ------------------------------------------------------------------ #
    #  Test 06 – Agent decides when to filter by metadata                  #
    # ------------------------------------------------------------------ #

    def test_06_agent_answers_with_tool_calling(self):
        """The agent autonomously chooses search tools and metadata filters."""
        query = "How does Qdrant handle sparse vectors?"
        answer = self.rag_agent.invoke(query)

        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0, msg="Agent returned an empty answer.")

        print(f"\n[OK] Agent query: '{query}'")
        print(f"     Answer:\n{answer}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
