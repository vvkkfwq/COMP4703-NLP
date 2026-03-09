"""
BM25 关键词检索器 — 载入磁盘 BM25Okapi 索引
"""

import pickle

from langchain_core.documents import Document

from src.config import BM25_PATH, TOP_K


class BM25Retriever:
    def __init__(self, index_path=BM25_PATH, top_k: int = TOP_K):
        self.top_k = top_k
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.index = data["index"]
        self.docs: list[Document] = data["docs"]

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        k = top_k or self.top_k
        scores = self.index.get_scores(query.lower().split())
        top_indices = scores.argsort()[::-1][:k]
        docs = []
        for idx in top_indices:
            doc = self.docs[idx]
            doc.metadata["_score"] = float(scores[idx])
            docs.append(doc)
        return docs
