"""
RAG Pipeline — 检索 → Prompt → OpenAI 生成
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from src.config import LLM_MODEL

load_dotenv()

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            'If the answer cannot be found in the context, respond with "I don\'t know."',
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class RAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            api_key=os.environ["OPENAI_API_KEY"],
        )

        self.chain = (
            RunnablePassthrough.assign(docs=itemgetter("question") | retriever)
            | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
            | RunnablePassthrough.assign(
                answer=(
                    {
                        "question": itemgetter("question"),
                        "context": itemgetter("context"),
                    }
                    | PROMPT_TEMPLATE
                    | self.llm
                    | StrOutputParser()
                )
            )
        )

    def run(self, query: str) -> dict:
        return self.chain.invoke({"question": query})

    def stream(self, query: str):
        docs = self.retriever.invoke(query)
        context = format_docs(docs)

        yield from (PROMPT_TEMPLATE | self.llm | StrOutputParser()).stream(
            {"question": query, "context": context}
        )
