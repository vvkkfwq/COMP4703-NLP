"""
Streamlit 问答界面
启动: streamlit run src/ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root (multi-hop-qa-system/) is on sys.path.
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.config import (
    RAG_PATH,
    EMBED_MODELS,
    EMBED_MODEL_DEFAULT,
    METRICS_PATH,
    STRATEGIES,
)
from src.evaluation.metrics import token_f1
from src.pipeline.rag import RAGPipeline
from src.retriever import (
    BM25Retriever,
    HybridRetriever,
    SemanticRetriever,
)


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Hop QA",
    page_icon="🔍",
    layout="wide",
)

# ─── UI helpers ──────────────────────────────────────────────────────────────


def _doc_score(doc) -> str:
    """Format the best available retrieval score for display."""
    meta = doc.metadata
    if "_rerank_score" in meta:
        return f"{meta['_rerank_score']:.4f} (rerank)"
    if "_rrf_score" in meta:
        return f"{meta['_rrf_score']:.4f} (rrf)"
    if "_score" in meta:
        return f"{meta['_score']:.4f} (semantic)"
    return "N/A"


def _result_docs(payload: dict | None) -> list:
    if not payload:
        return []
    return payload.get("retrieved_docs") or payload.get("docs") or []


# ─── Cached resource loaders ─────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading embedding model…")
def _get_semantic_retriever(model_key: str, enable_reranker: bool) -> SemanticRetriever:
    cfg = EMBED_MODELS[model_key]
    return SemanticRetriever(
        chroma_dir=str(cfg["chroma_dir"]),
        embedding_model=cfg["model_name"],
        enable_reranker=enable_reranker,
    )


@st.cache_resource(show_spinner="Loading BM25 index…")
def _get_bm25_retriever() -> BM25Retriever:
    return BM25Retriever()


def _build_retriever(model_key: str, use_bm25: bool, enable_reranker: bool):
    semantic = _get_semantic_retriever(
        model_key, enable_reranker if not use_bm25 else False
    )
    if use_bm25:
        return HybridRetriever(
            semantic=semantic,
            bm25=_get_bm25_retriever(),
            enable_reranker=enable_reranker,
        )
    return semantic


@st.cache_resource(show_spinner=False)
def build_pipeline(
    model_key: str, use_bm25: bool, enable_reranker: bool
) -> RAGPipeline:

    retriever = _build_retriever(model_key, use_bm25, enable_reranker)
    return RAGPipeline(retriever)


@st.cache_data(show_spinner=False)
def load_qa_data() -> list[dict]:
    with open(RAG_PATH) as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS_PATH) as f:
        return json.load(f)


# ─── App ─────────────────────────────────────────────────────────────────────

st.title("🔍 Multi-Hop QA System")

# strategy_name, use_bm25, enable_reranker = STRATEGIES[0]
# pipeline = build_pipeline(EMBED_MODEL_DEFAULT, use_bm25, enable_reranker)  # 默认
qa_data = load_qa_data()

# ── Sidebar / mode selector ──────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Preset questions", "Free-form"], index=0)
    st.divider()

with st.sidebar:
    st.header("Configuration")
    model_key = st.selectbox(
        "Embedding Model",
        options=list(EMBED_MODELS),
        format_func=lambda k: EMBED_MODELS[k]["label"],
        index=list(EMBED_MODELS).index(EMBED_MODEL_DEFAULT),
    )
    use_bm25 = st.toggle("BM25 + RRF", value=True)
    enable_reranker = st.toggle("Reranker (bge-reranker-base)", value=True)
    compare_mode = st.checkbox("Compare all 4 strategies", value=False)

    st.divider()

    with st.expander("Retrieval Metrics", expanded=False):
        try:
            all_metrics = load_metrics().get(model_key, {})
            rows = []
            for label, _, _ in STRATEGIES:
                m = all_metrics.get(label, {})
                rows.append(
                    {
                        "Strategy": label,
                        "NDCG@10": f"{m['NDCG@10']:.4f}" if m.get("NDCG@10") else "—",
                        "MAP@10": f"{m['MAP@10']:.4f}" if m.get("MAP@10") else "—",
                        "MRR@10": f"{m['MRR@10']:.4f}" if m.get("MRR@10") else "—",
                        "Hits@10": f"{m['Hits@10']:.4f}" if m.get("Hits@10") else "—",
                    }
                )
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption(f"Model: {EMBED_MODELS[model_key]['label']} · k=10")
        except FileNotFoundError:
            st.caption("Run `build_metrics.py` first.")

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
    if compare_mode:
        with st.spinner("Running 4 strategy configurations…"):
            compare_results = {}
            for label, bm25_flag, rerank_flag in STRATEGIES:
                p = build_pipeline(model_key, bm25_flag, rerank_flag)
                compare_results[label] = p.run(query)
        st.session_state["compare_results"] = compare_results
        st.session_state["result"] = None
        st.session_state["pending_stream"] = False
    else:
        with st.spinner("Retrieving documents…"):
            p = build_pipeline(model_key, use_bm25, enable_reranker)
            docs = p.retriever.invoke(query)
        st.session_state["result"] = {"question": query, "retrieved_docs": docs}
        st.session_state["pending_stream"] = True
        st.session_state["compare_results"] = None
    st.session_state["last_query"] = query
    st.session_state["last_mode"] = mode
    st.session_state["last_qa"] = selected_qa
    st.session_state["last_model_key"] = model_key
    st.session_state["last_use_bm25"] = use_bm25
    st.session_state["last_enable_reranker"] = enable_reranker

# Retrieve stored result (survives widget re-renders)
compare_results = st.session_state.get("compare_results")
result = st.session_state.get("result")
last_mode = st.session_state.get("last_mode")
last_qa = st.session_state.get("last_qa")
last_model_key = st.session_state.get("last_model_key")
last_use_bm25 = st.session_state.get("last_use_bm25", True)
last_enable_reranker = st.session_state.get("last_enable_reranker", True)
pending_stream = st.session_state.get("pending_stream", False)

# ── Output ────────────────────────────────────────────────────────────────────

if compare_results:
    st.divider()
    model_label = EMBED_MODELS.get(last_model_key or "", {}).get("label", "")
    st.subheader(f"Strategy Comparison — {model_label}")
    cols = st.columns(4)
    for col, (label, _, _) in zip(cols, STRATEGIES):
        r = compare_results[label]
        docs = _result_docs(r)
        with col:
            st.markdown(f"#### {label}")
            if last_mode == "Preset questions" and last_qa:
                f1 = token_f1(r["answer"], last_qa.get("answer", ""))
                st.metric("Token-F1", f"{f1:.1%}")
            st.info(r["answer"])
            st.caption(f"Sources ({len(docs)})")
            for i, doc in enumerate(docs[:5], 1):
                title = doc.metadata.get("title", "Untitled")[:50]
                st.markdown(f"**{i}.** {title}  \n`{_doc_score(doc)}`")

elif result:
    st.divider()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Answer")
    if pending_stream and st.session_state.get("last_query"):
        p = build_pipeline(last_model_key, last_use_bm25, last_enable_reranker)
        docs = _result_docs(result)
        answer = st.write_stream(p.stream(st.session_state["last_query"], docs=docs))
        result = {
            "question": st.session_state["last_query"],
            "answer": answer,
            "retrieved_docs": docs,
        }
        st.session_state["result"] = result
        st.session_state["pending_stream"] = False
        pending_stream = False
    else:
        st.markdown(result["answer"])

    # ── Sources ───────────────────────────────────────────────────────────────
    st.divider()
    docs = _result_docs(result)
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
