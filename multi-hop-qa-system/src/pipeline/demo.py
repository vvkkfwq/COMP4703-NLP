"""
端到端演示 — 用 sample-rag.json 第一条问题验证完整 pipeline
用法: python -m src.pipeline.demo
"""

import json

from src.config import RAG_PATH
from src.pipeline.rag import RAGPipeline
from src.retriever import (
    BM25Retriever,
    HybridRetriever,
    SemanticRetriever,
)


def main():
    with open(RAG_PATH) as f:
        qa_pairs = json.load(f)

    first = qa_pairs[0]
    query = first["query"]
    ground_truth = first["answer"]

    print(f"问题: {query}")
    print(f"Ground truth: {ground_truth}\n")

    print("初始化检索器 ...")
    semantic = SemanticRetriever()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(semantic=semantic, bm25=bm25, enable_reranker=True)

    pipeline = RAGPipeline(retriever=hybrid)

    print("运行 RAG pipeline ...\n")
    result = pipeline.run(query)

    print("=" * 60)
    print(f"回答: {result['answer']}")
    print("=" * 60)
    print(f"\n检索来源 ({len(result['sources'])} 条):")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src.get('title', 'N/A')} — {src.get('source', 'N/A')}")
        print(f"       {src.get('url', '')}")


if __name__ == "__main__":
    main()
