"""
Streamlit 问答界面
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from string import punctuation

# Ensure the project root (multi-hop-qa-system/) is on sys.path so that
# `src.*` imports resolve when Streamlit runs this file as a plain script.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.config import RAG_PATH
from src.pipeline.rag import RAGPipeline
from src.retriever import (
    BM25Retriever,
    CrossEncoderReranker,
    HybridRetriever,
    SemanticRetriever,
)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Hop QA",
    page_icon="🔍",
    layout="wide",
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, whitespace-tokenize (SQuAD style)."""
    text = text.lower()
    text = "".join(ch if ch not in punctuation else " " for ch in text)
    return text.split()


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred)
    gold_tokens = _normalize(gold)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _doc_score(doc) -> str:
    """Return the best available score from a Document's metadata."""
    meta = doc.metadata
    if "_rerank_score" in meta:
        return f"{meta['_rerank_score']:.4f} (rerank)"
    if "_score" in meta:
        return f"{meta['_score']:.4f} (semantic)"
    return "N/A"


# ─── Cached resource loaders ─────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading models (first run only)…")
def load_pipeline() -> RAGPipeline:
    semantic = SemanticRetriever()
    bm25 = BM25Retriever()
    reranker = CrossEncoderReranker()
    hybrid = HybridRetriever(semantic=semantic, bm25=bm25, reranker=reranker)
    return RAGPipeline(retriever=hybrid)


@st.cache_data(show_spinner=False)
def load_qa_data() -> list[dict]:
    with open(RAG_PATH) as f:
        return json.load(f)


# ─── App ─────────────────────────────────────────────────────────────────────

st.title("🔍 Multi-Hop QA System")

pipeline = load_pipeline()
qa_data = load_qa_data()

# ── Sidebar / mode selector ──────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Preset questions", "Free-form"], index=0)
    st.divider()
    st.caption(
        "**Retriever**: Hybrid (BGE-Large + BM25, RRF fusion) + Cross-encoder Reranker  \n"
        "**Generator**: " + pipeline.llm.model_name
    )

# ── Input area ───────────────────────────────────────────────────────────────

selected_qa: dict | None = None

if mode == "Preset questions":
    options = [f"[{i+1}] {qa['query']}" for i, qa in enumerate(qa_data)]
    idx = st.selectbox(
        "Select a question", range(len(options)), format_func=lambda i: options[i]
    )
    selected_qa = qa_data[idx]
    query = selected_qa["query"]
    st.markdown(f"**Question:** {query}")
else:
    query = st.text_area(
        "Enter your question",
        placeholder="Ask anything about the corpus…",
        height=80,
    )

ask_clicked = st.button("Ask", type="primary", disabled=not (query and query.strip()))

# ── Session state: run pipeline on button click ───────────────────────────────

if ask_clicked and query and query.strip():
    with st.spinner("Retrieving and generating…"):
        st.session_state["result"] = pipeline.run(query)
        st.session_state["last_query"] = query
        st.session_state["last_mode"] = mode
        st.session_state["last_qa"] = selected_qa

# Retrieve stored result (survives widget re-renders)
result: dict | None = st.session_state.get("result")
last_mode: str | None = st.session_state.get("last_mode")
last_qa: dict | None = st.session_state.get("last_qa")

# ── Output ────────────────────────────────────────────────────────────────────

if result:
    st.divider()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.info(result["answer"])

    # ── Sources ───────────────────────────────────────────────────────────────
    docs = result["retrieved_docs"]
    st.subheader(f"Retrieved Sources ({len(docs)})")

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        title = meta.get("title", "Untitled")
        source_pub = meta.get("source", "")
        pub_date = meta.get("published_at", "")[:10] if meta.get("published_at") else ""
        url = meta.get("url", "")
        score_str = _doc_score(doc)
        label = f"[{i}] {title}"
        if source_pub:
            label += f" — {source_pub}"

        with st.expander(label, expanded=(i == 1)):
            col_a, col_b, col_c = st.columns([3, 2, 2])
            with col_a:
                st.markdown(f"**Source:** {source_pub or '—'}")
            with col_b:
                st.markdown(f"**Published:** {pub_date or '—'}")
            with col_c:
                st.markdown(f"**Score:** {score_str}")
            if url:
                st.link_button("Open article ↗", url)
            st.markdown("**Excerpt:**")
            excerpt = doc.page_content
            st.markdown(
                f'<div style="background:#f8f9fa;padding:10px;border-radius:6px;'
                f'font-size:0.9em;line-height:1.5">{excerpt[:700]}{"…" if len(excerpt) > 700 else ""}</div>',
                unsafe_allow_html=True,
            )

    # ── Preset-only evaluation ────────────────────────────────────────────────
    if last_mode == "Preset questions" and last_qa is not None:
        st.divider()
        st.subheader("Evaluation")

        ground_truth: str = last_qa.get("answer", "")
        evidence_list: list[dict] = last_qa.get("evidence_list", [])

        f1 = token_f1(result["answer"], ground_truth)

        col_left, col_right = st.columns(2)

        with col_left:
            st.metric("Token-F1", f"{f1:.1%}")
            st.markdown(f"**Ground Truth Answer:** {ground_truth}")
            q_type = last_qa.get("question_type", "")
            if q_type:
                st.caption(f"Question type: `{q_type}`")

        with col_right:
            st.markdown("**Expected Evidence Sources**")
            if evidence_list:
                retrieved_titles = {
                    doc.metadata.get("title", "").strip().lower() for doc in docs
                }
                rows = []
                for ev in evidence_list:
                    ev_title = ev.get("title", "").strip()
                    hit = ev_title.lower() in retrieved_titles
                    icon = "✅" if hit else "❌"
                    ev_source = ev.get("source", "")
                    rows.append(
                        f"{icon} **{ev_title}**"
                        + (f" — {ev_source}" if ev_source else "")
                    )
                st.markdown("\n\n".join(rows))
            else:
                st.caption("No evidence list available.")
