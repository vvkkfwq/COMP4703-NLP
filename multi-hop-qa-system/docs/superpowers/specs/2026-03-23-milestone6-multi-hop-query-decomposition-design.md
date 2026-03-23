# Milestone 6 · Multi-Hop Query Decomposition — Design Spec

**Date:** 2026-03-23
**Scope:** Implement Query Decomposition multi-hop pipeline and adapt the Streamlit UI.
**Out of scope:** Single-hop vs multi-hop quantitative comparison, iterative retrieval (M7).

---

## Goals

- Implement a `MultiHopRAGPipeline` that decomposes a complex question into 2–3 sub-questions, retrieves documents for each sub-question independently, merges and deduplicates context, and streams the final answer.
- Add a **Pipeline mode** toggle (Single-hop / Multi-hop) to the Streamlit sidebar.
- In multi-hop mode, display the sub-question breakdown and per-hop sources before the streamed answer.

---

## Architecture

### New file: `src/pipeline/multi_hop_rag.py`

Independent of `RAGPipeline`. Does not inherit from it.

**Data classes:**

```python
@dataclass          # NOT frozen=True — the UI patches answer after streaming
class HopResult:
    sub_question: str
    docs: list[Document]

@dataclass          # NOT frozen=True — the UI patches answer after streaming
class MultiHopResult:
    question: str
    sub_questions: list[str]
    hop_results: list[HopResult]   # per-hop sub-question + retrieved docs
    merged_docs: list[Document]    # deduplicated union of all hop docs
    answer: str                    # empty string after retrieve(); patched by UI after stream()
```

**Public interface:**

```python
class MultiHopRAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL): ...

    def retrieve(self, query: str) -> MultiHopResult:
        """Decompose query → retrieve each sub-question → merge docs.
        Does NOT call the LLM for generation. Returns MultiHopResult with answer="".
        This is the method called inside the spinner block in the UI."""

    def stream(self, result: MultiHopResult) -> Iterator[str]:
        """Stream the final answer given a pre-populated MultiHopResult.
        Constructs prompt from result.question, result.sub_questions, and result.merged_docs.
        Called after retrieve() to produce the streamed answer."""
```

**Why this split:** mirrors the existing single-hop pattern in `app.py` exactly —
`p.retriever.invoke(query)` runs inside a spinner, then `p.stream(query, docs=docs)` is called
for `st.write_stream()`. Here, `pipeline.retrieve(query)` runs inside the spinner, then
`pipeline.stream(result)` drives `st.write_stream()`.

**Internal functions (module-level, pure or near-pure):**

| Function | Signature | Notes |
|---|---|---|
| `decompose_query` | `(query: str, llm) -> list[str]` | LLM call returning JSON array; fallback to `[query]` on parse error |
| `merge_docs` | `(hop_docs: list[list[Document]]) -> list[Document]` | Deduplicate by `metadata["url"]` then `metadata["title"]`; keep highest-scored copy |
| `_doc_score_value` | `(doc: Document) -> float` | Returns numeric score: `_rerank_score` → `_rrf_score` → `_score` → 0.0. Internal only — not exported; `app.py` has a separate `_doc_score()` that returns a formatted string. |

**Decompose LLM temperature:** `decompose_query` uses a separate `ChatOpenAI` instance with `temperature=0.0` (structured extraction task) rather than reusing the generation `llm` (which uses `temperature=0.1`). Both share the same `model` name passed to `__init__`.

**`stream()` prompt contract:**
The generation prompt includes sub-questions to give the LLM explicit reasoning context:

```
System: You are a helpful assistant. Answer using ONLY the provided context.
Human:
  Sub-questions used for retrieval:
  1. <sub_q1>
  2. <sub_q2>

  Context:
  <merged_docs formatted>

  Question: <original query>
  Answer:
```

**Decompose prompt contract:**
Returns a JSON array of strings, e.g. `["sub_q1", "sub_q2"]`. If the LLM output cannot be
parsed as a list, `decompose_query` logs a warning and returns `[query]` (graceful single-hop
fallback). Maximum 3 sub-questions enforced in the prompt instruction.

**Merge deduplication key priority:**
1. `metadata["url"]` (preferred — globally unique)
2. `metadata["title"]` (fallback when URL is absent)

When two docs share the same key, keep the one with the higher score (`_rerank_score` →
`_rrf_score` → `_score` → 0). Result is ordered by descending score.

---

### Modified file: `src/ui/app.py`

#### Sidebar addition

Add a `Pipeline` radio to the **Settings** section, below the Mode radio:

```
Settings
  Mode:        ○ Preset questions  ○ Free-form
  Pipeline:    ○ Single-hop  ● Multi-hop       ← NEW
─────────────────────────────────────────────
Configuration
  Embedding Model  [dropdown]
  BM25 + RRF       [toggle]
  Reranker         [toggle]
  Compare all 4    [checkbox — disabled when Pipeline = Multi-hop]
```

`Compare all 4 strategies` is disabled (`st.checkbox(..., disabled=pipeline_mode=="Multi-hop")`)
since comparison mode is single-hop only.

#### Session state keys (new)

| Key | Type | Purpose |
|---|---|---|
| `"multi_hop_result"` | `MultiHopResult \| None` | Stores decomposition + hop docs between reruns |
| `"pending_mh_stream"` | `bool` | Triggers streamed answer render on next rerun |
| `"last_pipeline_mode"` | `str` | `"Single-hop"` or `"Multi-hop"` — captured at click time, used during rendering rerun |

The following read-back lines must be added to the existing read-back block (after the `if ask_clicked` tree, analogous to `last_model_key`, `last_use_bm25`, etc.):

```python
last_pipeline_mode = st.session_state.get("last_pipeline_mode", "Single-hop")  # NEW
multi_hop_result = st.session_state.get("multi_hop_result")                     # NEW
pending_mh_stream = st.session_state.get("pending_mh_stream", False)            # NEW
```

#### Stale-state invalidation (updated)

The existing clearing block (triggered when query changes without clicking Ask) must be extended
to include the new multi-hop keys:

```python
if previous_query_key != current_query_key and not ask_clicked:
    st.session_state["result"] = None
    st.session_state["compare_results"] = None
    st.session_state["pending_stream"] = False
    st.session_state["multi_hop_result"] = None      # NEW
    st.session_state["pending_mh_stream"] = False    # NEW
```

#### Ask button execution — branch ordering

```python
if ask_clicked and query and query.strip():
    if pipeline_mode == "Multi-hop":
        # Multi-hop branch (checked first; compare_mode is disabled in this mode)
        with st.spinner("Decomposing question and retrieving…"):
            p = build_multi_hop_pipeline(model_key, use_bm25, enable_reranker)
            mh_result = p.retrieve(query)
        st.session_state["multi_hop_result"] = mh_result
        st.session_state["pending_mh_stream"] = True
        st.session_state["result"] = None
        st.session_state["compare_results"] = None
        st.session_state["pending_stream"] = False
    elif compare_mode:
        # Existing compare branch — unchanged
        ...
    else:
        # Existing single-hop branch — unchanged
        ...
    # Shared: store last_* keys
    st.session_state["last_pipeline_mode"] = pipeline_mode
    st.session_state["last_query"] = query
    st.session_state["last_mode"] = mode
    st.session_state["last_qa"] = selected_qa
    st.session_state["last_model_key"] = model_key
    st.session_state["last_use_bm25"] = use_bm25
    st.session_state["last_enable_reranker"] = enable_reranker
```

`build_multi_hop_pipeline(model_key, use_bm25, enable_reranker)` is a new helper analogous to
`build_pipeline()` — constructs a `MultiHopRAGPipeline` using the same `_build_retriever()` logic.
It is **not** decorated with `@st.cache_resource`; like `build_pipeline`, it is lightweight because
`_build_retriever()` internally calls `@st.cache_resource`-decorated loaders that cache the
retriever and vectorstore. The pipeline wrapper itself is always re-instantiated.

#### Multi-hop output rendering

Triggered when `last_pipeline_mode == "Multi-hop"` and `multi_hop_result is not None`.

```
── Sub-question Breakdown ──────────────────
  [1] <sub_question_1>
      st.expander → sources list (title + _doc_score())
  [2] <sub_question_2>
      st.expander → sources list (title + _doc_score())

── Answer ──────────────────────────────────
  if pending_mh_stream:
      p = build_multi_hop_pipeline(last_model_key, last_use_bm25, last_enable_reranker)
      try:
          answer = st.write_stream(p.stream(mh_result))
          mh_result.answer = answer          # patch into stored result
          session_state["multi_hop_result"] = mh_result
      finally:
          session_state["pending_mh_stream"] = False   # always clear to prevent re-stream on error
  else:
      st.markdown(mh_result.answer)

── Retrieved Sources (N) ───────────────────
  docs = mh_result.merged_docs
  [existing per-doc expander rendering — reused unchanged]

── Evaluation ──────────────────────────────
  answer_str = mh_result.answer
  docs = mh_result.merged_docs
  f1 = token_f1(answer_str, ground_truth)    # same call as single-hop
  [existing evidence hit-check — reused, reading from docs above]
```

The Sources section shows `merged_docs` (deduplicated union). Per-hop docs are only visible
inside the Sub-question Breakdown expanders. The Evaluation block is structurally identical
to single-hop — only the variable names differ (`mh_result.answer` vs `result["answer"]`).

---

## Error handling

| Failure point | Behaviour |
|---|---|
| `decompose_query` JSON parse error | Log warning; fallback to `[query]` (effective single-hop) |
| Sub-question retrieves 0 docs | `HopResult.docs = []`; excluded from merge; no crash |
| All sub-questions retrieve 0 docs | `merged_docs = []`; generation proceeds with empty context; LLM returns "I don't know." |
| OpenAI API error during `stream()` | Exception propagates to Streamlit; display via `st.error()` |
| `compare_mode=True` + `pipeline_mode="Multi-hop"` (stale state) | Multi-hop branch is checked first in the `if ask_clicked` tree; compare branch is unreachable |

---

## What is NOT changing

- `src/pipeline/rag.py` — untouched
- `src/retriever/` — untouched
- `src/config.py` — no new constants needed (reuses `LLM_MODEL`)
- Evaluation metrics logic — untouched; uses `token_f1(mh_result.answer, ground_truth)`

---

## Future seam for M7 (LangGraph)

`decompose_query`, `merge_docs`, and the per-hop retrieve loop inside `retrieve()` are
stateless functions. They map naturally onto LangGraph nodes (`decompose_node`,
`retrieve_node`, `generate_node`). Note that `MultiHopResult` is a dataclass while M7's
`RAGState` will be a `TypedDict` — migration will require mapping fields between the two
containers, but no changes to the retriever or UI rendering layers are expected.
