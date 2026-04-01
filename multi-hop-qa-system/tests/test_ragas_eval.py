"""Unit tests for src/evaluation/ragas_eval.py"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.ragas_eval import run_ragas


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_mock_ragas(scores: dict):
    """Return a mock ragas module that returns the given scores dict."""
    mock_ragas = MagicMock()
    mock_metrics = MagicMock()

    mock_result = MagicMock()
    mock_result.scores = [scores]
    mock_ragas.evaluate.return_value = mock_result

    # EvaluationDataset.from_list returns a sentinel dataset object
    mock_ragas.EvaluationDataset.from_list.return_value = MagicMock(name="dataset")

    return mock_ragas, mock_metrics


# ── tests ─────────────────────────────────────────────────────────────────────

class TestRunRagasReturnStructure:
    def test_returns_dict_with_four_keys(self):
        mock_ragas, mock_metrics = _make_mock_ragas(
            {"faithfulness": 0.9, "answer_relevancy": 0.8,
             "context_precision": 0.7, "context_recall": 0.85}
        )
        with patch.dict(sys.modules, {"ragas": mock_ragas, "ragas.metrics": mock_metrics}):
            result = run_ragas("q", "a", ["ctx"], "gt")

        assert set(result.keys()) == {
            "faithfulness", "answer_relevancy", "context_precision", "context_recall"
        }

    def test_returns_correct_float_values(self):
        mock_ragas, mock_metrics = _make_mock_ragas(
            {"faithfulness": 0.9, "answer_relevancy": 0.8,
             "context_precision": 0.7, "context_recall": 0.85}
        )
        with patch.dict(sys.modules, {"ragas": mock_ragas, "ragas.metrics": mock_metrics}):
            result = run_ragas("q", "a", ["ctx"], "gt")

        assert result["faithfulness"] == pytest.approx(0.9)
        assert result["answer_relevancy"] == pytest.approx(0.8)
        assert result["context_precision"] == pytest.approx(0.7)
        assert result["context_recall"] == pytest.approx(0.85)


class TestRunRagasInputPassing:
    def test_passes_question_to_dataset(self):
        mock_ragas, mock_metrics = _make_mock_ragas(
            {"faithfulness": 1.0, "answer_relevancy": 1.0,
             "context_precision": 1.0, "context_recall": 1.0}
        )
        with patch.dict(sys.modules, {"ragas": mock_ragas, "ragas.metrics": mock_metrics}):
            run_ragas("my question", "my answer", ["ctx1", "ctx2"], "ground truth")

        call_args = mock_ragas.EvaluationDataset.from_list.call_args[0][0][0]
        assert call_args["user_input"] == "my question"
        assert call_args["response"] == "my answer"
        assert call_args["retrieved_contexts"] == ["ctx1", "ctx2"]
        assert call_args["reference"] == "ground truth"


class TestRunRagasErrorHandling:
    def test_returns_none_values_when_ragas_not_installed(self):
        # Remove ragas from sys.modules to simulate ImportError
        with patch.dict(sys.modules, {"ragas": None, "ragas.metrics": None}):
            result = run_ragas("q", "a", ["ctx"], "gt")

        assert result == {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

    def test_returns_none_values_on_evaluate_exception(self):
        mock_ragas, mock_metrics = _make_mock_ragas({})
        mock_ragas.evaluate.side_effect = RuntimeError("API error")

        with patch.dict(sys.modules, {"ragas": mock_ragas, "ragas.metrics": mock_metrics}):
            result = run_ragas("q", "a", ["ctx"], "gt")

        assert result == {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

    def test_does_not_raise_on_any_exception(self):
        mock_ragas, mock_metrics = _make_mock_ragas({})
        mock_ragas.EvaluationDataset.from_list.side_effect = ValueError("bad input")

        with patch.dict(sys.modules, {"ragas": mock_ragas, "ragas.metrics": mock_metrics}):
            # Must not raise
            result = run_ragas("q", "a", ["ctx"], "gt")

        assert all(v is None for v in result.values())
