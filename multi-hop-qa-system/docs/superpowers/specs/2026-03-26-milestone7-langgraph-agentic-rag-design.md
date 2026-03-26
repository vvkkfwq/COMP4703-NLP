# Milestone 7 · LangGraph Agentic RAG — Design Spec

**Date:** 2026-03-26
**Status:** Approved

---

## Overview

Upgrade the multi-hop retrieval logic to a true Agentic RAG using LangGraph `StateGraph`. The agent starts from the original question, retrieves, then uses an LLM to judge whether the accumulated context is sufficient. If not, it generates a targeted follow-up query and retrieves again — up to 3 hops. This is qualitatively different from M6 (which does static query decomposition) because each hop is driven by an actual information gap.

---

## Data Structures

### `HopTrace` (TypedDict)

```python
class HopTrace(TypedDict):
    query: str            # the retrieval query used this hop
    docs: list[Document]  # documents retrieved this hop
    reasoning: str        # judge_node's explanation
```

### `RAGState` (TypedDict)

```python
class RAGState(TypedDict):
    question: str
    current_query: str          # query for the next retrieve_node call; initialised to question
    retrieved_docs: list[Document]  # all accumulated, deduped docs across hops
    hop_count: int
    sufficient: bool            # judge_node's verdict
    follow_up_query: str        # judge_node's suggested next query (empty string if sufficient)
    trace: list[HopTrace]       # one entry per completed hop
    answer: str                 # filled by generate_node
```

---

## Graph Topology

```
START → retrieve_node → judge_node ──(sufficient or hop_count ≥ 3)──→ generate_node → END
                            │
                    (not sufficient and hop_count < 3)
                            ↓
                       retrieve_node  (current_query = follow_up_query)
```

Conditional edge function:

```python
def should_continue(state: RAGState) -> str:
    if state["sufficient"] or state["hop_count"] >= 3:
        return "generate"
    return "retrieve"
```

---

## Node Specifications

### `retrieve_node`

- Calls `retriever.invoke(state["current_query"])`
- Merges results into `state["retrieved_docs"]` using the existing `merge_docs()` from `multi_hop_rag.py`
- Appends a `HopTrace` entry (query, docs for this hop, reasoning placeholder `""`)
- Increments `hop_count`
- Updates `current_query` to `state["follow_up_query"]` only if `follow_up_query` is non-empty (i.e. after the first hop)

### `judge_node`

Single LLM call with a structured output prompt. Input: original `question` + concatenated `page_content` of all `retrieved_docs` (truncated to fit context). Output JSON:

```json
{
  "sufficient": true,
  "follow_up_query": "",
  "reasoning": "Context contains X, Y, and Z which fully answers the question."
}
```

- Parses JSON; on parse failure, falls back to `sufficient=True` (prevents infinite loops)
- Updates `state["sufficient"]`, `state["follow_up_query"]`
- Writes `reasoning` back into the last `HopTrace` entry in `state["trace"]`

### `generate_node`

- Builds context string from `state["retrieved_docs"]` using existing `_format_docs()`
- Calls `chain.invoke()` (same prompt as M6's `_ANSWER_PROMPT`) and stores result in `state["answer"]`
- Generation is synchronous here; streaming is handled separately by `AgentRAGPipeline.stream_answer()`

---

## `AgentRAGPipeline` Interface (`src/pipeline/agent_rag.py`)

```python
class AgentRAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL) -> None: ...

    def run(self, question: str) -> RAGState:
        """Execute the full graph. Returns the final state with trace and answer."""

    def stream_answer(self, state: RAGState) -> Iterator[str]:
        """Stream the answer from retrieved_docs + question. Called by UI after run()."""
```

Internally constructs a `StateGraph[RAGState]`, adds nodes and the conditional edge, compiles to `app`, and invokes via `app.invoke(initial_state)`.

---

## Streamlit UI Changes (`src/ui/app.py`)

1. Pipeline radio gains a third option: `"Agent"` (Single-hop / Multi-hop unchanged)
2. `compare_mode` checkbox is disabled when `pipeline_mode == "Agent"` (same as Multi-hop)
3. On Ask click in Agent mode:
   - Spinner: `"Running agentic retrieval…"`
   - Calls `AgentRAGPipeline.run(query)`, stores result in `st.session_state["agent_result"]`
   - Sets `st.session_state["pending_agent_stream"] = True`
4. Rendering (always shown, not gated on preset mode):
   - Section header: `"Agent Reasoning Trace"`
   - One `st.expander` per hop, default expanded, title: `Hop N — "{query}"`
     - Judge reasoning (italic text)
     - Retrieved sources (title + score, same style as existing source display)
   - Below trace: stream final answer via `st.write_stream(pipeline.stream_answer(state))`

---

## Reuse & Non-Changes

| Component | Action |
|---|---|
| `merge_docs()` in `multi_hop_rag.py` | Reuse directly |
| `_format_docs()` in `multi_hop_rag.py` | Reuse directly |
| `_ANSWER_PROMPT` in `multi_hop_rag.py` | Reuse directly |
| `rag.py` | No changes |
| `multi_hop_rag.py` | No changes |
| Retriever layer | No changes |
| Evaluation / metrics | Not in scope |

---

## Error Handling

- **Judge parse failure**: fall back to `sufficient=True` to exit the loop gracefully
- **Empty retrieved_docs after a hop**: treat as sufficient (nothing new to retrieve)
- **Max hops reached**: `hop_count >= 3` forces exit to `generate_node` regardless of judge verdict

---

## Testing

- Unit test `retrieve_node`, `judge_node`, `generate_node` independently by passing mock `RAGState` dicts
- Integration test: `AgentRAGPipeline.run()` with a real retriever on one sample question, assert `hop_count >= 1`, `answer != ""`
- No changes to existing tests required
