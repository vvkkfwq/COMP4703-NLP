from langchain_core.documents import Document
from dataclasses import dataclass


# 数据类
@dataclass
class HopResult:
    sub_question: str
    docs: list[Document]


@dataclass
class MultiHopResult:
    question: str
    sub_questions: list[str]
    hop_results: list[HopResult]
    merged_docs: list[Document]
    answer: str


def _doc_score_value(doc: Document) -> float:
    """返回文档的数值分数，用于 merge_docs 去重比较。内部使用，不导出。"""
    meta = doc.metadata
    if "_rerank_score" in meta:
        return float(meta["_rerank_score"])
    if "_rrf_score" in meta:
        return float(meta["_rrf_score"])
    if "_score" in meta:
        return float(meta["_score"])
    return 0.0
