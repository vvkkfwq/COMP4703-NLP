# Milestone 6 · Multi-Hop Query Decomposition — Design Spec

**Date:** 2026-03-23
**Scope:** Implement Query Decomposition multi-hop pipeline and adapt the Streamlit UI.
**Out of scope:** Single-hop vs multi-hop quantitative comparison, iterative retrieval (M7).

---

## Goals

- Implement a `MultiHopRAGPipeline` that decomposes a complex question into 2–3 sub-questions, retrieves documents for each sub-question independently, merges and deduplicates context, and generates a final answer.
- Add a **Pipeline mode** toggle (Single-hop / Multi-hop) to the Streamlit sidebar.
- In multi-hop mode, display the sub-question breakdown and per-hop sources before the streamed answer.

---

## Architecture

### New file: `src/pipeline/multi_hop_rag.py`

Independent of `RAGPipeline`. Does not inherit from it.

**Data classes:**

```python
@dataclass
class HopResult:
    sub_question: str
    docs: list[Document]

@dataclass
class MultiHopResult:
    question: str
    sub_questions: list[str]
    hop_results: list[HopResult]   # per-hop sub-question + retrieved docs
    merged_docs: list[Document]    # deduplicated union of all hop docs
    answer: str                    # populated by run(); empty string when using stream()
```

**Public interface:**

```python
class MultiHopRAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL): ...

    def run(self, query: str) -> MultiHopResult:
        """Decompose → retrieve each sub-question → merge → generate. Fully synchronous."""

    def stream(self, query: str, merged_docs: list[Document], sub_questions: list[str]) -> Iterator[str]:
        """Stream only the final answer given pre-retrieved merged_docs."""
```

**Internal functions (module-level, pure or near-pure):**

| Function | Signature | Notes |
|---|---|---|
| `decompose_query` | `(query, llm) -> list[str]` | LLM call returning JSON array; fallback to `[query]` on parse error |
| `merge_docs` | `(hop_docs: list[list[Document]]) -> list[Document]` | Deduplicate by `metadata["url"]` then `metadata["title"]`; keep highest-scored copy; preserve original order |
| `_build_answer_prompt` | internal | Includes sub-questions in the system context for transparency |

**Decompose prompt contract:**
Returns a JSON array of strings, e.g. `["sub_q1", "sub_q2"]`. If LLM output cannot be parsed as a list, `decompose_query` logs a warning and returns `[query]` (graceful single-hop fallback).

**Merge deduplication key priority:**
1. `metadata["url"]` (preferred — globally unique)
2. `metadata["title"]` (fallback when URL is absent)

When two docs share the same key, keep the one with the higher score (`_rerank_score` → `_rrf_score` → `_score` → 0).

---

### Modified file: `src/ui/app.py`

#### Sidebar addition

Add a `Pipeline` radio above the existing Configuration section:

```
Settings
  Mode:        ○ Preset questions  ○ Free-form
  Pipeline:    ○ Single-hop  ● Multi-hop       ← NEW
─────────────────────────────────────────────
Configuration
  Embedding Model  [dropdown]
  BM25 + RRF       [toggle]
  Reranker         [toggle]
  Compare all 4    [checkbox]
```

`Compare all 4 strategies` is disabled (greyed out) when Pipeline = Multi-hop, since comparison mode is single-hop only.

#### Session state keys (new)

| Key | Type | Purpose |
|---|---|---|
| `"multi_hop_result"` | `MultiHopResult \| None` | Stores decomposition + hop docs between reruns |
| `"pending_mh_stream"` | `bool` | Triggers streamed answer render on next rerun |

#### Ask button execution (multi-hop branch)

```
spinner("Decomposing question and retrieving…")
  → build MultiHopRAGPipeline(retriever)
  → result = pipeline.run_without_generation(query)
      # run() minus final LLM call: returns MultiHopResult with answer=""
store result → session_state["multi_hop_result"]
set session_state["pending_mh_stream"] = True
```

`run_without_generation` is a thin wrapper (or a flag param) that skips the generation step, keeping retrieval and stream steps separate — consistent with how single-hop works.

#### Multi-hop output rendering

```
── Sub-question Breakdown ──────────────────
  [1] <sub_question_1>
      expander → sources list (title + score)
  [2] <sub_question_2>
      expander → sources list (title + score)

── Answer ──────────────────────────────────
  st.write_stream(pipeline.stream(query, merged_docs, sub_questions))

── Retrieved Sources (N) ───────────────────
  [existing _render_sources() logic, reused]

── Evaluation ──────────────────────────────
  [unchanged — F1 + evidence hit check]
```

The Sources section shows `merged_docs` (deduplicated union), not per-hop docs. Per-hop docs are only visible inside the Sub-question Breakdown expanders.

---

## Error handling

| Failure point | Behaviour |
|---|---|
| `decompose_query` JSON parse error | Warn + fallback to `[query]` (effective single-hop) |
| Sub-question retrieves 0 docs | `HopResult.docs = []`; excluded from merge; no crash |
| OpenAI API error during generation | Propagate exception to Streamlit; display error message via `st.error()` |

---

## What is NOT changing

- `src/pipeline/rag.py` — untouched
- `src/retriever/` — untouched
- `src/config.py` — no new constants needed (reuses `LLM_MODEL`)
- Evaluation metrics logic — untouched; multi-hop result feeds the same `token_f1()` call

---

## Future seam for M7 (LangGraph)

`decompose_query`, `merge_docs`, and the per-hop retrieve loop are all stateless functions. When M7 arrives, they map directly onto LangGraph nodes (`decompose_node`, `retrieve_node`, `judge_node`, `generate_node`) with no structural changes to the retriever or UI layers.
