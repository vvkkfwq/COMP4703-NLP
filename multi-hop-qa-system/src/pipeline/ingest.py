"""
- 读取 corpus.json
- 按 body 分块，保留 url / title / source metadata
- BGE-Large embedding 写入 ChromaDB（持久化）
- 同步构建 BM25 索引并序列化到磁盘
"""

import json
import pickle
from pathlib import Path

from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# Configure
CORPUS_PATH = Path("data/sample-corpus.json")
CHROMA_DIR = Path("chroma_db")
BM25_PATH = Path("bm25_index.pkl")

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
BATCH_SIZE = 64


def load_corpus(path: Path) -> list[Document]:
    """读取 corpus.json，切块，附加 metadata。"""
    with open(path) as f:
        corpus = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = []
    for item in corpus:
        chunks = splitter.split_text(item["body"])
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": item.get("title", ""),
                        "author": item.get("author", ""),
                        "source": item.get("source", ""),
                        "published_at": item.get("published_at", ""),
                        "category": item.get("category", ""),
                        "url": item.get("url", ""),
                    },
                )
            )

    return docs


def build_chroma(docs: list[Document]) -> None:
    """向量化并写入 ChromaDB。"""
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "mps"},
    )

    batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    vectorstore = None
    for batch in tqdm(batches, desc="Embedding", unit="batch"):
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embed_model,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            vectorstore.add_documents(batch)

    print(f"ChromaDB 写入完成 → {CHROMA_DIR}/")


def build_bm25(docs: list[Document]) -> None:
    """构建 BM25 索引并序列化到磁盘。"""
    tokenized = [doc.page_content.lower().split() for doc in docs]
    index = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index": index, "docs": docs}, f)
    print(f"BM25 索引写入完成 → {BM25_PATH}")


if __name__ == "__main__":
    print("加载语料库 ...")
    docs = load_corpus(CORPUS_PATH)
    print(f"共 {len(docs)} 个 chunks")

    print("构建 ChromaDB ...")
    build_chroma(docs)

    print("构建 BM25 ...")
    build_bm25(docs)

    print("Embedding 完成 ✓")
