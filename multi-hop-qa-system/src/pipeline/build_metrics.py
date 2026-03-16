"""
预计算所有模型 × 策略组合的检索指标，保存到 data/retrieval_metrics.json。
部署前运行一次：
    python -m src.pipeline.build_metrics
"""

import json

from src.config import METRICS_PATH, EMBED_MODELS, TOP_K, RAG_PATH, STRATEGIES
from src.retriever import (
    BM25Retriever,
    HybridRetriever,
    SemanticRetriever,
)
from src.evaluation.metrics import retrieval_metrics


def build_retriever(model_key: str, use_bm25: bool, enable_reranker: bool):
    semantic = SemanticRetriever(
        chroma_dir=EMBED_MODELS[model_key]["chroma_dir"],
        embedding_model=EMBED_MODELS[model_key]["model_name"],
        enable_reranker=enable_reranker if not use_bm25 else False,
    )
    if use_bm25:
        return HybridRetriever(
            semantic=semantic,
            bm25=BM25Retriever(),
            enable_reranker=enable_reranker,
        )

    return semantic


def compute_metrics(retriever, qa_data: list[dict]) -> dict[str, float | None]:
    buckets: dict[str, list[float]] = {
        "NDCG@10": [],
        "MAP@10": [],
        "MRR@10": [],
        "Hits@10": [],
    }
    for qa in qa_data:
        docs = retriever.retrieve(qa["query"], top_k=TOP_K)
        ndcg, map_s, mrr, hits = retrieval_metrics(
            docs, qa.get("evidence_list", []), k=TOP_K
        )
        if ndcg is not None:
            buckets["NDCG@10"].append(ndcg)
            buckets["MAP@10"].append(map_s)
            buckets["MRR@10"].append(mrr)
            buckets["Hits@10"].append(hits)
    return {k: round(sum(v) / len(v), 4) if v else None for k, v in buckets.items()}


if __name__ == "__main__":
    with open(RAG_PATH) as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} QA pairs\n")

    results = {}
    for model_key, cfg in EMBED_MODELS.items():
        print(f"=== {cfg['label']} ===")
        results[model_key] = {}
        for label, use_bm25, enable_reranker in STRATEGIES:
            print(f"[{label}]...  ", end="", flush=True)
            retriever = build_retriever(model_key, use_bm25, enable_reranker)
            results[model_key][label] = compute_metrics(retriever, qa_data)
            print(results[model_key][label])
        print()

    METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved to {METRICS_PATH}")
