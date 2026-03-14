"""
混合检索器 — RRF 融合语义 + BM25，可选 Cross-encoder 重排
RRF 公式: score(d) = Σ 1 / (k + rank(d))，k=60
"""

from langchain_core.documents import Document

from src.config import RRF_K, TOP_K
from .bm25_retriever import BM25Retriever
from .reranker import CrossEncoderReranker
from .semantic import SemanticRetriever


class HybridRetriever:
    def __init__(
        self,
        semantic: SemanticRetriever,
        bm25: BM25Retriever,
        enable_reranker: bool = False,
        rrf_k: int = RRF_K,
        top_k: int = TOP_K,
    ):
        self.semantic = semantic
        self.bm25 = bm25
        self.enable_reranker = enable_reranker
        self.reranker = CrossEncoderReranker()
        self.rrf_k = rrf_k
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        k = top_k or self.top_k

        semantic_docs = self.semantic.retrieve(query, top_k=k * 2)
        bm25_docs = self.bm25.retrieve(query, top_k=k * 2)

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(semantic_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        fused = []
        for key in sorted_keys[:k]:
            doc = doc_map[key]
            doc.metadata["_rrf_score"] = scores[key]
            fused.append(doc)

        if self.enable_reranker:
            fused = self.reranker.rerank(query, fused, top_k=k)

        return fused
