# RAG Hybrid Search

```
██████╗  █████╗  ██████╗
██╔══██╗██╔══██╗██╔════╝
██████╔╝███████║██║  ███╗
██╔══██╗██╔══██║██║   ██║
██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝

  Hybrid Search · Agentic RAG · Retrieval-Augmented Generation
```

> A clean, **Object Calisthenics**-driven RAG system featuring an **autonomous agent**
> that combines **dense (OpenAI)** and **sparse (BM25)** hybrid retrieval in Qdrant Cloud
> with **Mistral** as the generation LLM — fully orchestrated by LangChain.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![Mistral](https://img.shields.io/badge/Mistral_AI-FF7000?style=flat-square&logo=mistral&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white) ![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat-square&logo=qdrant&logoColor=white) ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white) ![Langfuse](https://img.shields.io/badge/Langfuse-Observability-000000?style=flat-square&logo=langfuse&logoColor=white)

---

## Observability with Langfuse

Every execution — chain or agent — is **fully traced** in [Langfuse](https://langfuse.com), capturing latency, token usage, cost, and model name per call.

```mermaid
flowchart LR
    subgraph TRACE ["Langfuse Trace"]
        direction TB
        OBS["@observe decorator<br/><i>root span</i>"]

        subgraph CHAIN_T ["RAG Chain trace"]
            direction LR
            LC1["LangChain CallbackHandler"] --> LLM1["ChatMistralAI<br/>model · tokens · cost"]
        end

        subgraph AGENT_T ["RAG Agent trace"]
            direction LR
            LC2["LangChain CallbackHandler"] --> LLM2["ChatMistralAI<br/>model · tokens · cost"]
            LLM2 --> TOOLS["tool calls<br/>search_documents /<br/>search_by_section"]
        end

        OBS --> CHAIN_T
        OBS --> AGENT_T
    end

    style TRACE fill:#0d1117,color:#e0e0e0,stroke:#000000,stroke-width:2px
    style OBS fill:#2d2d44,color:#c0c0c0,stroke:none
    style CHAIN_T fill:#1a1a2e,color:#e0e0e0,stroke:#2ea043,stroke-width:1px
    style AGENT_T fill:#1a1a2e,color:#e0e0e0,stroke:#FF7000,stroke-width:1px
    style LC1 fill:#2d2d44,color:#c0c0c0,stroke:none
    style LC2 fill:#2d2d44,color:#c0c0c0,stroke:none
    style LLM1 fill:#2d2d44,color:#c0c0c0,stroke:none
    style LLM2 fill:#2d2d44,color:#c0c0c0,stroke:none
    style TOOLS fill:#2d2d44,color:#c0c0c0,stroke:none
```

What is captured per trace:

| Signal | RAG Chain | RAG Agent |
|---|---|---|
| Root span + tags | ✅ `chain` | ✅ `agent` |
| Model name | ✅ | ✅ |
| Prompt & completion tokens | ✅ | ✅ |
| Cost | ✅ | ✅ |
| Tool calls & results | — | ✅ |
| Latency (end-to-end) | ✅ | ✅ |

> **Live trace →** <a href="https://us.cloud.langfuse.com/project/cmlxvbg68071ead07jrhkqnfp/traces/847d3f4286cb7a58357a65e0afec63d8?timestamp=2026-02-22T15:18:57.880Z" target="_blank">RAG Agent · "How does Qdrant handle sparse vectors?"</a>

---

## Two Modes of RAG

This project ships with **two approaches** to the "G" in RAG:


| Mode          | Description                                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **RAG Chain** | Deterministic LCEL pipeline: retrieve → prompt → generate. Simple, predictable, fast.                                           |
| **RAG Agent** | Autonomous tool-calling agent that **decides on its own** whether to search broadly or filter by document section via metadata. |


---

## The Agent in Action

```mermaid
flowchart TD
    U["<b>User</b><br/>How does Qdrant handle<br/>sparse vectors?"]

    subgraph AGENT ["RAG Agent  ·  Mistral tool-calling"]
        direction TB
        THINK["<b>Reason</b><br/>This is about Qdrant.<br/>I should filter by section."]
        PICK["<b>Pick tool</b><br/><code>search_by_section</code><br/>section = <i>What is Qdrant?</i>"]
        GEN["<b>Generate</b><br/>Synthesize answer<br/>from retrieved chunks"]
    end

    subgraph SEARCH ["Hybrid Search"]
        direction LR
        HS["HybridSearcher<br/>Dense + Sparse"]
        QD[("Qdrant Cloud<br/>filtered by<br/>metadata")]
        HS -->|query + filter| QD
        QD -->|Document chunks| HS
    end

    A["<b>Answer</b><br/><i>Qdrant supports sparse vectors<br/>using an inverted index...</i>"]

    U --> THINK
    THINK --> PICK
    PICK --> HS
    HS --> GEN
    GEN --> A

    style U fill:#4A90D9,color:#fff,stroke:none
    style AGENT fill:#1a1a2e,color:#e0e0e0,stroke:#FF7000,stroke-width:2px
    style THINK fill:#2d2d44,color:#e0e0e0,stroke:none
    style PICK fill:#2d2d44,color:#e0e0e0,stroke:none
    style GEN fill:#2d2d44,color:#e0e0e0,stroke:none
    style SEARCH fill:#0d1117,color:#e0e0e0,stroke:#DC244C,stroke-width:2px
    style HS fill:#2d2d44,color:#e0e0e0,stroke:none
    style QD fill:#DC244C,color:#fff,stroke:none
    style A fill:#2ea043,color:#fff,stroke:none
```



The agent has **two tools** at its disposal and picks the right one per query:

```mermaid
flowchart LR
    subgraph TOOLBELT ["Agent Tool Belt"]
        direction LR
        subgraph T1 ["search_documents"]
            D1["Broad search across<br/>the entire knowledge base"]
            D2["Use when the question<br/>is general or ambiguous"]
        end
        subgraph T2 ["search_by_section"]
            S1["Targeted search filtered<br/>by section heading"]
            S2["Use when the question<br/>clearly maps to one topic"]
        end
    end

    style TOOLBELT fill:#0d1117,color:#e0e0e0,stroke:#FF7000,stroke-width:2px
    style T1 fill:#1a1a2e,color:#e0e0e0,stroke:#4A90D9,stroke-width:1px
    style T2 fill:#1a1a2e,color:#e0e0e0,stroke:#DC244C,stroke-width:1px
    style D1 fill:#2d2d44,color:#c0c0c0,stroke:none
    style D2 fill:#2d2d44,color:#888,stroke:none
    style S1 fill:#2d2d44,color:#c0c0c0,stroke:none
    style S2 fill:#2d2d44,color:#888,stroke:none
```



---

## Full Pipeline

```mermaid
flowchart TB
    subgraph INDEX ["Indexing Pipeline"]
        direction LR
        DOC["doc.md"] --> SPLIT["MarkdownSplitter<br/><i>headers → metadata</i>"]
        SPLIT --> EMB["Embedder<br/>Dense + Sparse"]
        EMB --> QD1[("Qdrant Cloud<br/><i>hybrid index +<br/>payload indexes</i>")]
    end

    subgraph QUERY ["Query Pipeline"]
        direction TB
        Q["User Query"]

        subgraph CHAIN ["RAG Chain  (LCEL)"]
            direction LR
            R1["Retriever"] --> P1["Prompt"] --> M1["Mistral"]
        end

        subgraph AG ["RAG Agent"]
            direction LR
            R2["Mistral picks tools<br/>on its own"] --> R3["broad search /<br/>section filter"]
        end

        Q --> CHAIN
        Q --> AG
        CHAIN --> ANS["Answer"]
        AG --> ANS
    end

    QD1 -.->|serves both| CHAIN
    QD1 -.->|serves both| AG

    style INDEX fill:#0d1117,color:#e0e0e0,stroke:#4A90D9,stroke-width:2px
    style DOC fill:#2d2d44,color:#c0c0c0,stroke:none
    style SPLIT fill:#2d2d44,color:#c0c0c0,stroke:none
    style EMB fill:#2d2d44,color:#c0c0c0,stroke:none
    style QD1 fill:#DC244C,color:#fff,stroke:none
    style QUERY fill:#0d1117,color:#e0e0e0,stroke:#FF7000,stroke-width:2px
    style Q fill:#4A90D9,color:#fff,stroke:none
    style CHAIN fill:#1a1a2e,color:#e0e0e0,stroke:#2ea043,stroke-width:1px
    style AG fill:#1a1a2e,color:#e0e0e0,stroke:#FF7000,stroke-width:1px
    style R1 fill:#2d2d44,color:#c0c0c0,stroke:none
    style P1 fill:#2d2d44,color:#c0c0c0,stroke:none
    style M1 fill:#2d2d44,color:#c0c0c0,stroke:none
    style R2 fill:#2d2d44,color:#c0c0c0,stroke:none
    style R3 fill:#2d2d44,color:#c0c0c0,stroke:none
    style ANS fill:#2ea043,color:#fff,stroke:none
```



---

## Architecture

```mermaid
graph TD
    subgraph Config
        S[Settings]
    end

    subgraph Document
        DL[DocumentLoader]
        TC[TextChunker<br/>MarkdownHeaderSplitter]
        CH[Chunks / TextChunk<br/>+ metadata]
    end

    subgraph Embeddings
        DE[DenseEmbedder<br/>OpenAI]
        SE[SparseEmbedder<br/>BM25 / FastEmbed]
    end

    subgraph Storage
        CM[CollectionManager<br/>+ payload indexes]
        DI[DocumentInserter]
    end

    subgraph Search
        HS[HybridSearcher<br/>QdrantVectorStore<br/>+ metadata filter]
    end

    subgraph Generation
        RC[RAGChain<br/>LCEL]
    end

    subgraph Agent
        RA[RAGAgent<br/>create_agent + tools]
    end

    subgraph Infrastructure
        QD[(Qdrant Cloud)]
    end

    S --> DE & SE & CM & DI & HS & RC & RA

    DL --> TC --> CH --> DI
    DE & SE --> DI
    DI --> QD

    HS --> QD
    HS --> RC & RA
    RC -->|Answer| User((User))
    RA -->|Answer| User
```



---

## Agent Flow (Tool-Calling)

```mermaid
sequenceDiagram
    participant User
    participant Agent as RAGAgent<br/>(Mistral tool-calling)
    participant Tool as search_by_section /<br/>search_documents
    participant Searcher as HybridSearcher
    participant Qdrant as Qdrant Cloud

    User->>Agent: "How does Qdrant handle sparse vectors?"
    Agent->>Agent: Reason about question
    Agent->>Tool: search_by_section(query, "What is Qdrant?")
    Tool->>Searcher: search(query, metadata)
    Searcher->>Qdrant: hybrid search + metadata filter
    Qdrant-->>Searcher: Document[]
    Searcher-->>Tool: Document[]
    Tool-->>Agent: formatted context
    Agent->>Agent: Generate answer from context
    Agent-->>User: Final answer
```



---

## RAG Chain Flow (LCEL)

```mermaid
sequenceDiagram
    participant User
    participant RAGChain
    participant Retriever as HybridSearcher<br/>(LangChain Retriever)
    participant QdrantCloud as Qdrant Cloud
    participant Mistral as Mistral LLM

    User->>RAGChain: invoke("What is RAG?")
    RAGChain->>Retriever: get_relevant_documents(query)
    Retriever->>QdrantCloud: hybrid search (dense + sparse)
    QdrantCloud-->>Retriever: Document[]
    Retriever-->>RAGChain: Document[]
    RAGChain->>RAGChain: format context
    RAGChain->>Mistral: prompt(context + question)
    Mistral-->>RAGChain: generated answer
    RAGChain-->>User: answer string
```



---

## Indexing Flow

```mermaid
sequenceDiagram
    participant Test
    participant DocumentLoader
    participant TextChunker as TextChunker<br/>(Markdown)
    participant DocumentInserter
    participant DenseEmbedder
    participant SparseEmbedder
    participant Qdrant as Qdrant Cloud

    Test->>DocumentLoader: load("doc.md")
    DocumentLoader-->>Test: raw markdown
    Test->>TextChunker: chunk(text)
    Note over TextChunker: MarkdownHeaderSplitter<br/>extracts h1/h2/h3 as metadata
    TextChunker-->>Test: Chunks[TextChunk + metadata]
    Test->>DocumentInserter: insert(chunks)
    loop For each chunk
        DocumentInserter->>DenseEmbedder: embed(chunk.content)
        DenseEmbedder-->>DocumentInserter: float[]
        DocumentInserter->>SparseEmbedder: embed(chunk.content)
        SparseEmbedder-->>DocumentInserter: SparseVector
    end
    DocumentInserter->>Qdrant: upsert(points with metadata)
```



---

## Project Structure

```
rag-hybrid-search-multi-query/
│
├── doc.md                         ← source Markdown document to index
├── test_rag.py                    ← end-to-end test suite (7 tests)
├── requirements.txt
├── .env                           ← API keys & config
│
└── src/
    ├── config/
    │   └── Settings.py            ← single source of truth for env vars
    │
    ├── document/
    │   ├── DocumentLoader.py      ← reads raw text from file
    │   └── TextChunker.py         ← MarkdownHeaderSplitter + metadata
    │
    ├── embeddings/
    │   ├── DenseEmbedder.py       ← OpenAI text-embedding-3-small
    │   └── SparseEmbedder.py      ← BM25 via FastEmbed
    │
    ├── storage/
    │   ├── CollectionManager.py   ← create / delete collections + payload indexes
    │   └── DocumentInserter.py    ← build PointStructs with metadata and upsert
    │
    ├── search/
    │   └── HybridSearcher.py      ← QdrantVectorStore (hybrid + metadata filter)
    │
    ├── generation/
    │   └── RAGChain.py            ← LCEL chain: retriever → Mistral → answer
    │
    └── agent/
        └── RAGAgent.py            ← autonomous agent with search tools
```

---

## Object Calisthenics Applied


| Rule                     | How It Is Applied                                                            |
| ------------------------ | ---------------------------------------------------------------------------- |
| One level of indentation | Every method does one thing; loops delegate to helpers                       |
| No `else`                | Early returns replace every `if/else` branch                                 |
| Wrap primitives          | `TextChunk` wraps `(content, index, metadata)`                               |
| First-class collections  | `Chunks` is a dedicated class, not a plain list                              |
| One dot per line         | No chained calls across object boundaries                                    |
| No abbreviations         | `dense_embedder`, `sparse_vector`, `embedding_dimension` — always full names |
| Small entities           | Every class fits on one screen; every method under ~12 lines                 |


---

## Tech Stack

| Component | Technology |
|---|---|
| Vector DB | ![Qdrant](https://img.shields.io/badge/Qdrant_Cloud-DC244C?style=flat-square&logo=qdrant&logoColor=white) |
| Dense embeddings | ![OpenAI](https://img.shields.io/badge/OpenAI-text--embedding--3--small-412991?style=flat-square&logo=openai&logoColor=white) |
| Sparse embeddings | ![FastEmbed](https://img.shields.io/badge/FastEmbed-Qdrant%2Fbm25-DC244C?style=flat-square&logo=qdrant&logoColor=white) |
| Text splitting | ![LangChain](https://img.shields.io/badge/LangChain-MarkdownHeaderSplitter-1C3C3C?style=flat-square&logo=langchain&logoColor=white) |
| Retriever | ![LangChain](https://img.shields.io/badge/LangChain-QdrantVectorStore_Hybrid-1C3C3C?style=flat-square&logo=langchain&logoColor=white) |
| LLM | ![Mistral](https://img.shields.io/badge/Mistral_AI-mistral--large--latest-FF7000?style=flat-square&logo=mistral&logoColor=white) |
| RAG Chain | ![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=flat-square&logo=langchain&logoColor=white) |
| RAG Agent | ![LangChain](https://img.shields.io/badge/LangChain-create__agent_+_tool--calling-1C3C3C?style=flat-square&logo=langchain&logoColor=white) |
| Observability | ![Langfuse](https://img.shields.io/badge/Langfuse-traces_·_tokens_·_cost-000000?style=flat-square&logo=langfuse&logoColor=white) |
| Language | ![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white) |

