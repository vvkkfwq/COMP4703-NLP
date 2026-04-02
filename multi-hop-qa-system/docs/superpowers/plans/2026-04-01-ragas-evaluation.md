# Milestone 8 · RAGAS Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add on-demand RAGAS evaluation (faithfulness, answer_relevancy, context_precision, context_recall) to the Streamlit UI for preset-question mode across all three pipeline types.

**Architecture:** A thin wrapper `src/evaluation/ragas_eval.py` exposes `run_ragas()`, which builds a single-sample `EvaluationDataset` and calls RAGAS `evaluate()`. The UI adds a "Run RAGAS Evaluation" button in each pipeline's preset-eval section; clicking triggers the wrapper with a spinner, stores scores in `st.session_state["ragas_result"]`, and renders four `st.metric` widgets. A shared helper `_render_ragas_eval()` in `app.py` eliminates code duplication across the three pipeline branches.

**Tech Stack:** `ragas>=0.2,<0.3`, existing `openai` key (RAGAS uses it as evaluator LLM), `streamlit`, `pytest` + `unittest.mock`

---

## File Map

| Action | Path                           | Responsibility                                                                    |
| ------ | ------------------------------ | --------------------------------------------------------------------------------- |
| Create | `src/evaluation/ragas_eval.py` | `run_ragas()` wrapper; isolates RAGAS API details                                 |
| Create | `tests/test_ragas_eval.py`     | Unit tests for `run_ragas()` (all mocked)                                         |
| Modify | `requirements.txt`             | Add `ragas>=0.2,<0.3`                                                             |
| Modify | `src/ui/app.py`                | Helper `_render_ragas_eval()`, session-state cleanup, wiring into 3 eval sections |

---

## Task 1: Add ragas to requirements and install

**Files:**

- Modify: `requirements.txt`

- [ ] **Step 1: Add dependency**

Open `requirements.txt` and append one line so it reads:

```
langchain-chroma
langchain-core
langchain-huggingface
langchain-openai
langchain-text-splitters
langgraph
python-dotenv
pytest
ragas>=0.2,<0.3
rank-bm25
sentence-transformers
streamlit
tqdm
```

- [ ] **Step 2: Install**

```bash
conda run -n rag pip install "ragas>=0.2,<0.3"
```

Expected: installs without error, ends with `Successfully installed ragas-0.2.x …`

- [ ] **Step 3: Verify import**

```bash
conda run -n rag python -c "import ragas; print(ragas.__version__)"
```

Expected: prints a version string starting with `0.2`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add ragas>=0.2,<0.3 dependency"
```

---

## Task 2: Write failing tests for run_ragas()

**Files:**

- Create: `tests/test_ragas_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ragas_eval.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n rag pytest tests/test_ragas_eval.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'src.evaluation.ragas_eval'` — confirms the file doesn't exist yet.

---

## Task 3: Implement src/evaluation/ragas_eval.py

**Files:**

- Create: `src/evaluation/ragas_eval.py`

- [ ] **Step 1: Create the file**

```python
"""
RAGAS 评估包装器 — 单条按需评分
用法: from src.evaluation.ragas_eval import run_ragas
"""
from __future__ import annotations

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
    except (ImportError, TypeError):
        print(
            "WARNING: ragas not installed or incompatible version. "
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
        print(f"WARNING: RAGAS evaluation failed: {exc}")
        return dict(_NULL_RESULT)
```

- [ ] **Step 2: Run tests to confirm they pass**

```bash
conda run -n rag pytest tests/test_ragas_eval.py -v
```

Expected: all 7 tests `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/ragas_eval.py tests/test_ragas_eval.py
git commit -m "feat: add ragas_eval.run_ragas() wrapper with unit tests"
```

---

## Task 4: Update app.py — session cleanup + helper function

**Files:**

- Modify: `src/ui/app.py`

This task adds the `_render_ragas_eval()` helper and clears `ragas_result` when the question changes or a new query is submitted. No wiring into pipeline branches yet.

- [ ] **Step 1: Add `ragas_result` to the question-change clearing block**

Find this block (around line 216–228):

```python
if (
    previous_query_key is not None
    and previous_query_key != current_query_key
    and not ask_clicked
):
    st.session_state["result"] = None
    st.session_state["compare_results"] = None
    st.session_state["pending_stream"] = False
    st.session_state["multi_hop_result"] = None
    st.session_state["pending_mh_stream"] = False
    st.session_state["agent_result"] = None
    st.session_state["pending_agent_stream"] = False
```

Add one line at the end of the `if` block:

```python
    st.session_state["ragas_result"] = None
```

- [ ] **Step 2: Also clear `ragas_result` when a new query is submitted**

Find the `if ask_clicked and query and query.strip():` block (around line 232). After the opening line add:

```python
    st.session_state["ragas_result"] = None
```

(So it sits as the very first statement inside that `if` block, before the `if pipeline_mode == "Agent":` checks.)

- [ ] **Step 3: Add the `_render_ragas_eval` helper function**

Place it immediately after `build_agent_pipeline` (around line 124), before `load_qa_data`:

```python
def _render_ragas_eval(answer: str, docs: list, last_qa: dict) -> None:
    """Render RAGAS evaluation button and metric results (preset mode only)."""
    from src.evaluation.ragas_eval import run_ragas  # lazy — avoids import cost at startup

    st.divider()
    st.subheader("RAGAS Evaluation")

    if st.button("Run RAGAS Evaluation", key="run_ragas_btn"):
        contexts = [doc.page_content for doc in docs]
        with st.spinner("Running RAGAS evaluation… (15–30 s, uses OpenAI)"):
            scores = run_ragas(
                question=last_qa["query"],
                answer=answer,
                contexts=contexts,
                ground_truth=last_qa.get("answer", ""),
            )
        st.session_state["ragas_result"] = scores

    scores = st.session_state.get("ragas_result")
    if scores is not None:
        cols = st.columns(4)
        labels = [
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Ans. Relevancy"),
            ("context_precision", "Ctx. Precision"),
            ("context_recall", "Ctx. Recall"),
        ]
        for col, (key, label) in zip(cols, labels):
            val = scores.get(key)
            with col:
                st.metric(label, f"{val:.2f}" if val is not None else "—")
```

- [ ] **Step 4: Commit**

```bash
git add src/ui/app.py
git commit -m "feat: add _render_ragas_eval helper and session state cleanup"
```

---

## Task 5: Wire RAGAS eval into all three pipeline eval sections

**Files:**

- Modify: `src/ui/app.py`

Each pipeline's eval block ends after the "Expected Evidence Sources" column. Wire `_render_ragas_eval()` as the last call inside each `if last_mode == "Preset questions"` block.

- [ ] **Step 1: Wire into the Agent eval section**

Find the Agent eval block (the one inside `if last_pipeline_mode == "Agent" and agent_result is not None:`). It ends after this code:

```python
            else:
                st.caption("No evidence list available.")
```

Immediately after that `else` clause (still inside `if last_mode == "Preset questions" and last_qa is not None:`), append:

```python
        _render_ragas_eval(
            agent_result.get("answer", ""),
            agent_result.get("retrieved_docs", []),
            last_qa,
        )
```

- [ ] **Step 2: Wire into the Multi-hop eval section**

Find the Multi-hop eval block (the one inside `elif last_pipeline_mode == "Multi-hop" and multi_hop_result is not None:`). It ends after the same evidence `else` clause. Append:

```python
        _render_ragas_eval(
            multi_hop_result.answer or "",
            multi_hop_result.merged_docs,
            last_qa,
        )
```

- [ ] **Step 3: Wire into the Single-hop eval section**

Find the Single-hop eval block (the one inside `elif result:`). It ends after the same evidence `else` clause. Append:

```python
        _render_ragas_eval(
            result.get("answer", ""),
            _result_docs(result),
            last_qa,
        )
```

- [ ] **Step 4: Run existing tests to confirm nothing is broken**

```bash
conda run -n rag pytest tests/ -v --ignore=tests/integration -x 2>&1 | tail -20
```

Expected: all existing tests still `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/ui/app.py
git commit -m "feat: wire RAGAS evaluation into Single-hop, Multi-hop, and Agent eval sections"
```

---

## Task 6: Smoke test

- [ ] **Step 1: Start the app**

```bash
conda run -n rag streamlit run src/ui/app.py
```

- [ ] **Step 2: Verify RAGAS button appears**

1. Open the app in the browser
2. Select **Preset questions** mode, pick any question
3. Choose **Single-hop** pipeline, click **Ask**
4. Wait for the answer to stream
5. Scroll to the **Evaluation** section — confirm a "RAGAS Evaluation" sub-section with a "Run RAGAS Evaluation" button appears **below** the Token-F1 / Evidence block

- [ ] **Step 3: Verify RAGAS scores render**

1. Click **Run RAGAS Evaluation**
2. Wait for the spinner (~15–30 s)
3. Confirm four `st.metric` boxes appear: Faithfulness, Ans. Relevancy, Ctx. Precision, Ctx. Recall — each showing a value between 0.00 and 1.00

- [ ] **Step 4: Verify score clears on question switch**

1. Switch to a different preset question (without clicking Ask)
2. Click **Ask**, wait for answer
3. Confirm the RAGAS metrics area is empty (no stale scores from previous question)

- [ ] **Step 5: Repeat for Multi-hop and Agent modes**

Switch `Pipeline` to **Multi-hop**, ask a question, verify the same RAGAS button and metrics appear. Repeat for **Agent**.

---

## Self-review checklist (pre-commit)

- Spec requirement: `src/evaluation/ragas_eval.py` → Task 3 ✓
- Spec requirement: faithfulness / answer_relevancy / context_precision / context_recall → Task 3 ✓
- Spec requirement: "Run RAGAS" button (async/spinner) → Tasks 4 & 5 ✓
- Spec requirement: results alongside token-F1 → Tasks 4 & 5 ✓
- Spec requirement: Preset questions only → `_render_ragas_eval` only called inside `if last_mode == "Preset questions"` blocks ✓
- Spec requirement: all 3 pipeline modes → Task 5 (agent, multi-hop, single-hop) ✓
- Spec requirement: RAGAS not installed → graceful None return → Task 3 (`ImportError` catch) ✓
- Spec requirement: ragas>=0.2 API (EvaluationDataset.from_list + SingleTurnSample schema) → Task 3 ✓
- No placeholder steps ✓
- Type consistency: `run_ragas()` signature same in tests (Task 2) and implementation (Task 3) ✓
