"""
为所有 embedding 模型构建独立的 ChromaDB 集合。
每个模型写入 chroma_db/<model-key>/ 子目录，互不干扰。

运行：
    python -m src.pipeline.build_all_indexes
"""

from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.pipeline.ingest import load_corpus

from src.config import (
    CORPUS_PATH,
    CHROMA_DIR,
    EMBED_MODELS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
    EMBED_DEVICE,
)


def build_chroma_for_model(docs: list[Document], model_key: str) -> None:
    cfg = EMBED_MODELS[model_key]
    chroma_dir = cfg["chroma_dir"]
    chroma_dir.mkdir(parents=True, exist_ok=True)

    embed_model = HuggingFaceEmbeddings(
        model_name=cfg["model_name"],
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": EMBED_DEVICE},
    )

    batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    vectorstore = None
    for batch in tqdm(batches, desc="Embedding", unit="batch"):
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embed_model,
                persist_directory=str(chroma_dir),
            )
        else:
            vectorstore.add_documents(batch)

    print(f"  写入完成 → {chroma_dir}/")


if __name__ == "__main__":
    print("加载语料库...")
    docs = load_corpus(CORPUS_PATH)
    print(f"  共 {len(docs)} 个 chunks\n")

    for key in EMBED_MODELS:
        model_name = EMBED_MODELS[key]["model_name"]
        print(f"构建 ChromaDB [{key}]  ({model_name})")
        build_chroma_for_model(docs, key)
        print(f"{model_name} 构建完成")
        print()

    print("全部 ChromaDB 构建完成 ✓  ")
