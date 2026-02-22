from typing import Dict, Optional

from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from pydantic import SecretStr

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI

from src.config.settings import Settings
from src.search.hybrid_searcher import HybridSearcher


class RAGChain:
    PROMPT_TEMPLATE = (
        "Answer the question based only on the following context:\n\n"
        "{context}\n\n"
        "Question: {question}"
    )

    def __init__(self, searcher: HybridSearcher, settings: Settings):
        self._searcher = searcher
        self._llm = ChatMistralAI(
            name=settings.mistral_model,
            api_key=SecretStr(settings.mistral_api_key),
        )
        self._prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)

    @observe(name="rag_chain")
    def invoke(self, question: str, metadata: Optional[Dict[str, str]] = None) -> str:
        langfuse = get_client()
        langfuse.update_current_trace(tags=["chain"])
        retriever = self._searcher.as_retriever(metadata=metadata)
        chain = (
            {"context": retriever | self._format_documents, "question": RunnablePassthrough()}
            | self._prompt
            | self._llm
            | StrOutputParser()
        )
        result = chain.invoke(question, config={"callbacks": [CallbackHandler()]})
        langfuse.flush()
        return result

    @staticmethod
    def _format_documents(documents):
        return "\n\n".join(document.page_content for document in documents)
