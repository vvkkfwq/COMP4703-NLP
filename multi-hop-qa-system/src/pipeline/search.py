"""
快速搜索测试 - 验证 ChromaDB 和 BM25 索引
用法: python -m src.pipeline.search "your query here"
"""

import sys
import pickle
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = Path("chroma_db")
BM25_PATH = Path("bm25_index.pkl")
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
TOP_K = 3


def search(query: str) -> None:
    print(f"\n查询: {query}\n")

    # ChromaDB 语义检索
    print("=" * 50)
    print("ChromaDB 语义检索")
    print("=" * 50)
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "mps"},
    )
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR), embedding_function=embed_model
    )
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)
    for i, (doc, score) in enumerate(results, 1):
        print(f"[{i}] score={score:.4f} | {doc.metadata['title']}")
        print(f"    {doc.page_content[:150]}...")
        print()

    # BM25 关键词检索
    print("=" * 50)
    print("BM25 关键词检索")
    print("=" * 50)
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    index, docs = data["index"], data["docs"]
    scores = index.get_scores(query.lower().split())
    top_indices = scores.argsort()[::-1][:TOP_K]
    for i, idx in enumerate(top_indices, 1):
        print(f"[{i}] score={scores[idx]:.4f} | {docs[idx].metadata['title']}")
        print(f"    {docs[idx].page_content[:150]}...")
        print()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What happened to FTX?"
    search(query)
