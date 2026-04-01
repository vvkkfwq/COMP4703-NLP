"""
RAGAS 评估包装器 — 单条按需评分
用法: from src.evaluation.ragas_eval import run_ragas
"""
from __future__ import annotations

import logging

_NULL_RESULT: dict[str, float | None] = {
    "faithfulness": None,
    "answer_relevancy": None,
    "context_precision": None,
    "context_recall": None,
}


def run_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float | None]:
    """Run RAGAS evaluation on a single QA sample.

    Args:
        question:     The user question.
        answer:       The pipeline-generated answer.
        contexts:     List of retrieved document texts (page_content).
        ground_truth: The reference answer string from the QA dataset.

    Returns:
        Dict with keys faithfulness, answer_relevancy, context_precision,
        context_recall. Values are floats in [0, 1] or None on failure.
    """
    try:
        from ragas import EvaluationDataset, evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception:  # noqa: BLE001
        logging.warning(
            "ragas import failed (not installed or incompatible). "
            "Run: pip install 'ragas>=0.2,<0.3'"
        )
        return dict(_NULL_RESULT)

    try:
        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": question,
                    "retrieved_contexts": contexts,
                    "response": answer,
                    "reference": ground_truth,
                }
            ]
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores: dict = result.scores[0]
        return {
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_precision": scores.get("context_precision"),
            "context_recall": scores.get("context_recall"),
        }
    except Exception as exc:  # noqa: BLE001
        logging.warning("RAGAS evaluation failed: %s", exc)
        return dict(_NULL_RESULT)
