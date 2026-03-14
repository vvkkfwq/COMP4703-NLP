"""
语义检索器 — BGE-Large + ChromaDB
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.config import (
    CHROMA_DIR,
    EMBED_MODELS,
    EMBED_MODEL_DEFAULT,
    TOP_K,
    EMBED_DEVICE,
)
from .reranker import CrossEncoderReranker


class SemanticRetriever:
    def __init__(
        self,
        chroma_dir=EMBED_MODELS[EMBED_MODEL_DEFAULT]["chroma_dir"],
        embedding_model: str = EMBED_MODELS[EMBED_MODEL_DEFAULT]["model_name"],
        top_k: int = TOP_K,
        enable_reranker: bool = False,
    ):
        self.top_k = top_k
        self.enable_reranker = enable_reranker
        self.reranker = CrossEncoderReranker()
        embed = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": EMBED_DEVICE},
        )
        self.vectorstore = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embed,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        k = top_k or self.top_k
        fetch_k = k * 3 if self.reranker else k  # reranker 开启时多取候选
        results = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
        docs = []
        for doc, score in results:
            doc.metadata["_score"] = float(score)
            docs.append(doc)

        if self.enable_reranker:
            docs = self.reranker.rerank(query, docs, top_k=k)
        return docs
