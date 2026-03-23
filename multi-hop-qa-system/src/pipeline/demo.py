"""
端到端演示 — 用 sample-rag.json 第一条问题验证完整 pipeline
用法:
    python -m src.pipeline.demo run
    python -m src.pipeline.demo st
"""

import argparse
import json

from src.config import RAG_PATH
from src.pipeline.rag import RAGPipeline
from src.retriever import (
    HybridRetriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG pipeline demo")
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run", "st"],
        help="run: 一次性返回结果, st: 流式输出答案",
    )
    return parser.parse_args()


def print_retrieved_docs(docs):
    print(f"\n检索文档 ({len(docs)} 条):")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"  [{i}] {meta.get('title', 'N/A')} — {meta.get('source', 'N/A')}")
        print(f"       {meta.get('url', '')}")


def main():
    args = parse_args()

    with open(RAG_PATH) as f:
        qa_pairs = json.load(f)

    first = qa_pairs[0]
    query = first["query"]
    ground_truth = first["answer"]

    print(f"问题: {query}")
    print(f"Ground truth: {ground_truth}\n")

    print("初始化检索器 ...")
    hybrid = HybridRetriever(enable_reranker=True)

    pipeline = RAGPipeline(retriever=hybrid)

    if args.mode == "run":
        print("运行 RAG pipeline (run) ...\n")
        result = pipeline.run(query)

        print("=" * 60)
        print(f"回答: {result['answer']}")
        print("=" * 60)
        print_retrieved_docs(result["docs"])
        return

    print("运行 RAG pipeline (st) ...\n")
    print("=" * 60)
    print("流式回答:", end=" ", flush=True)
    for chunk in pipeline.stream(query):
        print(chunk, end="", flush=True)
    print()
    print("=" * 60)

    # stream 模式仅流出文本，这里单独再取一遍文档用于展示。
    docs = hybrid.invoke(query)
    print_retrieved_docs(docs)


if __name__ == "__main__":
    main()
