from typing import Dict, Optional

from pydantic import SecretStr

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI

from src.config.Settings import Settings
from src.search.HybridSearcher import HybridSearcher


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

    def invoke(self, question: str, metadata: Optional[Dict[str, str]] = None) -> str:
        retriever = self._searcher.as_retriever(metadata=metadata)
        chain = (
            {"context": retriever | self._format_documents, "question": RunnablePassthrough()}
            | self._prompt
            | self._llm
            | StrOutputParser()
        )
        return chain.invoke(question)

    @staticmethod
    def _format_documents(documents):
        return "\n\n".join(document.page_content for document in documents)
