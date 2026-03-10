"""
评估指标 — token_f1, retrieval_metrics (NDCG@k, MAP@k, MRR@k)
"""

from __future__ import annotations

import math
from collections import Counter
from string import punctuation


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, whitespace-tokenize (SQuAD style)."""
    text = text.lower()
    text = "".join(ch if ch not in punctuation else " " for ch in text)
    return text.split()


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token-level F1 between predicted and gold answers."""
    pred_tokens = _normalize(pred)
    gold_tokens = _normalize(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_metrics(
    retrieved_docs, evidence_list: list[dict], k: int = 10
) -> tuple[float, float, float, float] | tuple[None, None, None, None]:
    """Compute NDCG@k, MAP@k, MRR@k, Hits@k with binary title-match relevance.

    Args:
        retrieved_docs: list of LangChain Document objects (in ranked order).
        evidence_list: ground-truth evidence dicts with a 'title' field.
        k: cutoff rank (default 10).

    Returns:
        (ndcg, map_score, mrr, hits) or (None, None, None, None) if evidence_list is empty.
    """
    relevant = {ev.get("title", "").strip().lower() for ev in evidence_list}
    if not relevant:
        return None, None, None, None

    hits = [
        1 if doc.metadata.get("title", "").strip().lower() in relevant else 0
        for doc in retrieved_docs[:k]
    ]

    # MRR@k
    mrr = next((1.0 / rank for rank, rel in enumerate(hits, 1) if rel), 0.0)

    # MAP@k
    num_rel = len(relevant)
    num_hits, ap = 0, 0.0
    for rank, rel in enumerate(hits, 1):
        if rel:
            num_hits += 1
            ap += num_hits / rank
    map_score = ap / min(num_rel, k)

    # NDCG@k (binary relevance)
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(hits, 1))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(num_rel, k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Hits@k — 1 if at least one relevant doc is in top-k, else 0
    hits_at_k = 1.0 if any(hits) else 0.0

    return ndcg, map_score, mrr, hits_at_k
