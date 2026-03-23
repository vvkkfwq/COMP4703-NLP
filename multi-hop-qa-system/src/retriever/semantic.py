"""
语义检索器 — BGE-Large + ChromaDB
"""

from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from src.config import (
    EMBED_MODELS,
    EMBED_MODEL_DEFAULT,
    TOP_K,
    EMBED_DEVICE,
)
from .reranker import CrossEncoderReranker


class SemanticRetriever(BaseRetriever):

    chroma_dir: str = EMBED_MODELS[EMBED_MODEL_DEFAULT]["chroma_dir"]
    embedding_model: str = EMBED_MODELS[EMBED_MODEL_DEFAULT]["model_name"]
    top_k: int = TOP_K
    enable_reranker: bool = False
    vectorstore: Any = None
    reranker: Any = None

    def model_post_init(self, context: Any) -> None:

        # 初始化 Embeddings
        embed = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": EMBED_DEVICE},
        )

        # 初始化 Chroma vectorstore
        self.vectorstore = Chroma(
            persist_directory=str(self.chroma_dir), embedding_function=embed
        )

        # 固定加载 reranker
        self.reranker = CrossEncoderReranker()

        # 必须调用父类的 model_post_init，确保 LangChain 内部逻辑执行
        super().model_post_init(context)

    def _get_relevant_documents(
        self, query, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """LangChain BaseRetriever 的标准接口"""
        return self.retrieve(query)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        k = top_k or self.top_k
        fetch_k = k * 3 if self.enable_reranker else k  # reranker 开启时多取候选
        results = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
        docs = []
        for doc, score in results:
            doc.metadata["_score"] = float(score)
            docs.append(doc)

        if self.enable_reranker:
            docs = self.reranker.rerank(query, docs, top_k=k)
        return docs
