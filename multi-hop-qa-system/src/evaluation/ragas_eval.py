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

    Note:
        All imports and evaluate() run inside a worker thread with a fresh
        standard asyncio event loop. This bypasses the uvloop incompatibility
        with nest_asyncio (which RAGAS uses internally). If the imports were
        done in the main Streamlit thread, nest_asyncio.apply() would attempt
        to patch the uvloop event loop and raise ValueError.
    """
    def _run_in_thread() -> dict[str, float | None]:
        # Create a standard SelectorEventLoop directly — bypasses the global
        # event loop policy (which Streamlit sets to uvloop). uvloop cannot be
        # patched by nest_asyncio (used internally by ragas), so we must avoid
        # it. SelectorEventLoop works on all platforms and is nest_asyncio-safe.
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            from ragas import EvaluationDataset, evaluate  # noqa: PLC0415
            from ragas.metrics import (  # noqa: PLC0415
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

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
        finally:
            loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            return _pool.submit(_run_in_thread).result(timeout=180)
    except concurrent.futures.TimeoutError:
        logging.warning("RAGAS evaluation timed out after 180 s")
        return dict(_NULL_RESULT)
    except Exception as exc:  # noqa: BLE001
        logging.warning("RAGAS thread execution failed: %s", exc)
        return dict(_NULL_RESULT)
