"""
RAG Pipeline — 检索 → Prompt → OpenAI 生成
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, TOP_K

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


class RAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL, top_k: int = TOP_K):
        self.retriever = retriever
        self.top_k = top_k
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.chain = PROMPT_TEMPLATE | self.llm

    def run(self, query: str) -> dict:
        docs: list[Document] = self.retriever.retrieve(query, top_k=self.top_k)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        response = self.chain.invoke({"context": context, "question": query})
        sources = [
            {k: v for k, v in doc.metadata.items() if not k.startswith("_")}
            for doc in docs
        ]
        return {
            "answer": response.content,
            "sources": sources,
            "retrieved_docs": docs,
        }
