# Retrieval-Augmented Generation (RAG): A Comprehensive Overview

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) is an AI framework that enhances the output of large language models (LLMs) by grounding them in external, up-to-date knowledge. Instead of relying solely on parametric knowledge baked into model weights during training, RAG dynamically retrieves relevant documents from a knowledge base and feeds them as context to the LLM at inference time.

The technique was introduced in the 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Lewis et al. at Facebook AI Research. Since then it has become one of the most widely adopted patterns in production AI systems.

## How Does RAG Work?

A RAG pipeline consists of two main phases: the indexing phase and the retrieval-and-generation phase.

During indexing, source documents are split into smaller chunks. Each chunk is converted into a vector embedding using an embedding model. The embeddings are stored in a vector database alongside the original text.

During inference, a user query is embedded using the same embedding model. The vector database performs a similarity search to find the most relevant chunks. The retrieved chunks are injected into the LLM prompt as context. The LLM generates a response grounded in that retrieved context.

## What is Hybrid Search?

Hybrid search combines two fundamentally different retrieval strategies to improve accuracy and recall.

**Dense retrieval** uses neural embeddings (typically from transformer models) to capture semantic similarity. Even if the query and document share no keywords, dense retrieval can find them if they express the same meaning. OpenAI's `text-embedding-3-small` is a popular choice for dense embeddings.

**Sparse retrieval** uses term-frequency statistics such as BM25 to perform keyword-based matching. It excels at finding exact term matches and works well on queries with rare or technical terms. The `Qdrant/bm25` model from FastEmbed provides efficient sparse embeddings.

By fusing the scores from both approaches, hybrid search delivers higher precision on keyword-specific queries while still capturing semantic nuance.

## What is Qdrant?

Qdrant is a high-performance, open-source vector search engine written in Rust. It supports:

- Dense vector search with HNSW indexing
- Sparse vector search with inverted index support
- Hybrid search via the `search_batch` API
- Payload filtering to narrow results by metadata fields
- Scalar and product quantization for memory efficiency

Qdrant exposes a REST and gRPC API and provides official Python, TypeScript, Go, and Rust clients. It can run locally via Docker or as a managed cloud service at [cloud.qdrant.io](https://cloud.qdrant.io).

## What is Multi-Query Retrieval?

Multi-query retrieval is a strategy where the user's original question is automatically reformulated into several alternative phrasings. Each reformulation is used as an independent query against the vector store. The results from all queries are merged and deduplicated before being passed to the LLM.

This technique addresses the limitation that a single query phrasing may miss relevant documents whose wording is slightly different. By exploring multiple perspectives of the same question, multi-query retrieval increases recall without requiring the user to manually rephrase their question.

LangChain provides a `MultiQueryRetriever` class that integrates seamlessly with any underlying retriever, including Qdrant-based ones.

## Object Calisthenics and Clean Code

This project applies Object Calisthenics, a set of nine rules proposed by Jeff Bay to encourage writing cleaner, more maintainable object-oriented code:

1. One level of indentation per method.
2. Do not use the `else` keyword.
3. Wrap all primitives and strings in domain-specific types.
4. Use first-class collections for groups of objects.
5. Use only one dot per line (avoid method chaining across objects).
6. Do not abbreviate names.
7. Keep all entities small (files, classes, methods).
8. No classes with more than two instance variables.
9. No getters or setters; expose behavior, not data.

Applying these rules leads to highly cohesive, loosely coupled code where each class has a single clear responsibility, making the system easier to test, extend, and reason about.

## Key Design Decisions in This Project

The project separates concerns into five distinct layers:

### Configuration Layer

The configuration layer (`Settings`) reads environment variables and exposes them as typed attributes. It is the single source of truth for all configuration.

### Embedding Layer

The embedding layer provides two interchangeable embedders: `DenseEmbedder` wraps OpenAI embeddings and `SparseEmbedder` wraps BM25. Both expose a consistent `embed(text)` interface.

### Document Layer

The document layer contains `DocumentLoader` for reading raw text files and `TextChunker` for splitting text into overlapping chunks. `TextChunk` and `Chunks` are value objects that wrap primitives and collections respectively.

### Storage Layer

The storage layer contains `CollectionManager` for creating and deleting Qdrant collections, and `DocumentInserter` for building and upserting `PointStruct` objects.

### Search Layer

The search layer contains `HybridSearcher` which orchestrates a hybrid search against both the dense and sparse vector indices via LangChain's `QdrantVectorStore`, and `RAGChain` which feeds the retrieved context to Mistral for answer generation.

## Running the Project

**Prerequisites:** Python 3.10+, a Qdrant Cloud instance (or local Qdrant), and valid API keys in the `.env` file.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run the full test pipeline with:

```bash
python -m pytest test_rag.py -v -s
```

The tests run in order: first the collection is created, then `doc.md` is indexed, then a hybrid search query is executed, and finally the full RAG chain generates an answer via Mistral.
