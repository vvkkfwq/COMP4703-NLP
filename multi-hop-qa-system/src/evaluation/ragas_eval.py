"""
RAGAS 评估包装器 — 单条按需评分
用法: from src.evaluation.ragas_eval import run_ragas
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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
    except Exception as _import_exc:  # noqa: BLE001
        logging.warning(
            "ragas import failed (%s: %s). " "Run: pip install 'ragas>=0.2,<0.3'",
            type(_import_exc).__name__,
            _import_exc,
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
        # Run in a worker thread with a fresh standard asyncio event loop so
        # that nest_asyncio (used internally by ragas) can patch it — uvloop
        # (used by Streamlit) cannot be patched and raises ValueError otherwise.
        def _run_evaluate():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return evaluate(
                    dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                )
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            result = _pool.submit(_run_evaluate).result(timeout=180)
        scores: dict = result.scores[0]
        return {
            k: float(v) if v is not None else None
            for k, v in {
                "faithfulness": scores.get("faithfulness"),
                "answer_relevancy": scores.get("answer_relevancy"),
                "context_precision": scores.get("context_precision"),
                "context_recall": scores.get("context_recall"),
            }.items()
        }
    except Exception as exc:  # noqa: BLE001
        logging.warning("RAGAS evaluation failed: %s", exc)
        return dict(_NULL_RESULT)
