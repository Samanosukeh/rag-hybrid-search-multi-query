import os

from langchain.agents import create_agent
from langchain_core.tools import tool

from src.config.Settings import Settings
from src.search.HybridSearcher import HybridSearcher


class RAGAgent:
    SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions using a knowledge base.\n"
        "Always search the knowledge base before answering — never guess.\n\n"
        "You have two search tools:\n"
        "- search_documents: broad search across the entire knowledge base.\n"
        "- search_by_section: targeted search filtered by a specific section heading.\n\n"
        "Strategy:\n"
        "1. If the question clearly targets one topic, use search_by_section first.\n"
        "2. If results are insufficient or the question is broad, fall back to search_documents.\n"
        "3. Synthesize a clear answer from the retrieved content."
    )

    def __init__(self, searcher: HybridSearcher, settings: Settings):
        os.environ["MISTRAL_API_KEY"] = settings.mistral_api_key
        self._agent = create_agent(
            model=f"mistralai:{settings.mistral_model}",
            tools=self._build_tools(searcher),
            system_prompt=self.SYSTEM_PROMPT,
        )

    def invoke(self, question: str) -> str:
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        return result["messages"][-1].content

    @staticmethod
    def _build_tools(searcher: HybridSearcher):

        @tool
        def search_documents(query: str) -> str:
            """Search the entire knowledge base with a natural language query.
            Use this for broad questions or when unsure which section to look in."""
            docs = searcher.search(query)
            if not docs:
                return "No results found."
            return "\n\n".join(doc.page_content for doc in docs)

        @tool
        def search_by_section(query: str, section: str) -> str:
            """Search within a specific section of the knowledge base.
            The 'section' parameter must be a h2 heading from the document:
            - 'What is Retrieval-Augmented Generation?'
            - 'How Does RAG Work?'
            - 'What is Hybrid Search?'
            - 'What is Qdrant?'
            - 'What is Multi-Query Retrieval?'
            - 'Object Calisthenics and Clean Code'
            - 'Key Design Decisions in This Project'
            - 'Running the Project'"""
            docs = searcher.search(query, metadata={"header_2": section})
            if not docs:
                return "No results found in that section. Try search_documents instead."
            return "\n\n".join(doc.page_content for doc in docs)

        return [search_documents, search_by_section]
