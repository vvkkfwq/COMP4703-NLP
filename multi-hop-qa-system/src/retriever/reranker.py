"""
Cross-encoder 重排器 — BAAI/bge-reranker-base
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, docs: list[Document], top_k: int = 3
    ) -> list[Document]:
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = []
        for score, doc in ranked[:top_k]:
            doc.metadata["_rerank_score"] = float(score)
            result.append(doc)
        return result
