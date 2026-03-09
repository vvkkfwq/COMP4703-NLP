"""
语义检索器 — BGE-Large + ChromaDB
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.config import CHROMA_DIR, EMBED_MODEL, TOP_K, EMBED_DEVICE


class SemanticRetriever:
    def __init__(
        self,
        chroma_dir=CHROMA_DIR,
        embedding_model: str = EMBED_MODEL,
        top_k: int = TOP_K,
    ):
        self.top_k = top_k
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
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        docs = []
        for doc, score in results:
            doc.metadata["_score"] = float(score)
            docs.append(doc)
        return docs
