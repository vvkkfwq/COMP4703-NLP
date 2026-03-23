"""
混合检索器 — RRF 融合语义 + BM25，可选 Cross-encoder 重排
RRF 公式: score(d) = Σ 1 / (k + rank(d))，k=60
"""

from typing import Any

from src.config import RRF_K, TOP_K
from .bm25_retriever import BM25Retriever
from .reranker import CrossEncoderReranker
from .semantic import SemanticRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class HybridRetriever(BaseRetriever):

    semantic: Any = None
    bm25: Any = None
    enable_reranker: bool = False
    reranker: Any = None
    rrf_k: int = RRF_K
    top_k: int = TOP_K

    def model_post_init(self, context: Any) -> None:
        """初始化 semantic、bm25、reranker 实例。"""
        if self.semantic is None:
            self.semantic = SemanticRetriever()
        if self.bm25 is None:
            self.bm25 = BM25Retriever()
        if self.enable_reranker and self.reranker is None:
            self.reranker = CrossEncoderReranker()

        super().model_post_init(context)

    def _get_relevant_documents(
        self, query, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.retrieve(query)

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

        if self.enable_reranker and self.reranker is not None:
            fused = self.reranker.rerank(query, fused, top_k=k)

        return fused
