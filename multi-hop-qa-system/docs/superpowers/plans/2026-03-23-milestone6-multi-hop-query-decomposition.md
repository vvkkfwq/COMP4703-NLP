# Milestone 6 · 多跳查询分解 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Query Decomposition 多跳 RAG pipeline，将复杂问题分解为子问题独立检索后合并，并在 Streamlit UI 中新增 Pipeline 模式切换与多跳结果展示。

**Architecture:** 新增独立的 `MultiHopRAGPipeline`（不继承 `RAGPipeline`），持有 `retrieve()` + `stream()` 两个方法；`retrieve()` 完成分解→检索→合并，`stream()` 流式输出最终答案。UI 层新增 Pipeline radio，多跳模式下在 Answer 前渲染子问题分解块。

**Tech Stack:** Python dataclasses · LangChain `ChatOpenAI` · LCEL chain · Streamlit session state · pytest + unittest.mock

---

## 文件结构

| 操作 | 路径 | 职责 |
|---|---|---|
| 新建 | `src/pipeline/multi_hop_rag.py` | 数据类 + `MultiHopRAGPipeline` + 内部纯函数 |
| 新建 | `tests/test_multi_hop_rag.py` | `multi_hop_rag.py` 的单元测试 |
| 修改 | `src/ui/app.py` | Pipeline radio · session state · 多跳渲染分支 |

---

## Task 1: 数据类与 `_doc_score_value`

**Files:**
- Create: `src/pipeline/multi_hop_rag.py`（完整初始内容，含所有模块级 import）
- Create: `tests/test_multi_hop_rag.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_multi_hop_rag.py  ← 完整初始文件内容
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.pipeline.multi_hop_rag import (
    HopResult,
    MultiHopResult,
    _doc_score_value,
)
from tests.conftest import MockRetriever


class TestDataClasses:
    def test_hop_result_fields(self):
        doc = Document(page_content="test")
        hr = HopResult(sub_question="q", docs=[doc])
        assert hr.sub_question == "q"
        assert hr.docs == [doc]

    def test_multi_hop_result_fields(self):
        hr = HopResult(sub_question="q", docs=[])
        mhr = MultiHopResult(
            question="orig",
            sub_questions=["q"],
            hop_results=[hr],
            merged_docs=[],
            answer="",
        )
        assert mhr.question == "orig"
        assert mhr.answer == ""

    def test_multi_hop_result_is_mutable(self):
        mhr = MultiHopResult(
            question="q", sub_questions=[], hop_results=[], merged_docs=[], answer=""
        )
        mhr.answer = "patched"
        assert mhr.answer == "patched"


class TestDocScoreValue:
    def test_prefers_rerank_score(self):
        doc = Document(page_content="x", metadata={"_rerank_score": 0.9, "_score": 0.5})
        assert _doc_score_value(doc) == 0.9

    def test_falls_back_to_rrf_score(self):
        doc = Document(page_content="x", metadata={"_rrf_score": 0.7, "_score": 0.5})
        assert _doc_score_value(doc) == 0.7

    def test_falls_back_to_score(self):
        doc = Document(page_content="x", metadata={"_score": 0.6})
        assert _doc_score_value(doc) == 0.6

    def test_returns_zero_when_no_score(self):
        doc = Document(page_content="x", metadata={})
        assert _doc_score_value(doc) == 0.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n rag pytest tests/test_multi_hop_rag.py -v
```
预期：`ModuleNotFoundError: No module named 'src.pipeline.multi_hop_rag'`

- [ ] **Step 3: 创建 `src/pipeline/multi_hop_rag.py`（完整初始内容）**

> 注意：所有模块级 import（含 `json`、`logging`、`load_dotenv` 等）一次性写入顶部，避免后续 Task 追加时出现重复 import。

```python
# src/pipeline/multi_hop_rag.py
"""
Multi-Hop RAG Pipeline — Query Decomposition 多跳检索
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL

load_dotenv()

logger = logging.getLogger(__name__)


# ── 数据类 ─────────────────────────────────────────────────────────────────────


@dataclass          # 不使用 frozen=True — UI 在 stream() 后回写 answer 字段
class HopResult:
    sub_question: str
    docs: list[Document]


@dataclass          # 不使用 frozen=True — UI 在 stream() 后回写 answer 字段
class MultiHopResult:
    question: str
    sub_questions: list[str]
    hop_results: list[HopResult]
    merged_docs: list[Document]
    answer: str  # retrieve() 后为 ""; UI 完成 stream() 后回写


# ── 内部辅助函数 ────────────────────────────────────────────────────────────────


def _doc_score_value(doc: Document) -> float:
    """返回文档的数值分数，用于 merge_docs 去重比较。内部使用，不导出。
    注意：app.py 有独立的 _doc_score() 返回格式化字符串，两者不共用。"""
    meta = doc.metadata
    if "_rerank_score" in meta:
        return float(meta["_rerank_score"])
    if "_rrf_score" in meta:
        return float(meta["_rrf_score"])
    if "_score" in meta:
        return float(meta["_score"])
    return 0.0
```

- [ ] **Step 4: 运行测试确认通过**

```bash
conda run -n rag pytest tests/test_multi_hop_rag.py::TestDataClasses tests/test_multi_hop_rag.py::TestDocScoreValue -v
```
预期：8 passed

- [ ] **Step 5: 提交**

```bash
git add src/pipeline/multi_hop_rag.py tests/test_multi_hop_rag.py
git commit -m "feat: add MultiHopResult dataclasses and _doc_score_value"
```

---

## Task 2: `merge_docs` 去重合并

**Files:**
- Modify: `src/pipeline/multi_hop_rag.py`（在 `_doc_score_value` 之后追加）
- Modify: `tests/test_multi_hop_rag.py`（在文件末尾追加）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_multi_hop_rag.py` 末尾）**

```python
# 追加到 tests/test_multi_hop_rag.py 末尾
# 注意：Document 和其他 import 已在文件顶部，无需重复
from src.pipeline.multi_hop_rag import merge_docs


class TestMergeDocs:
    def test_deduplicates_by_url(self):
        doc_a = Document(page_content="v1", metadata={"url": "http://x.com", "_score": 0.9})
        doc_b = Document(page_content="v2", metadata={"url": "http://x.com", "_score": 0.5})
        result = merge_docs([[doc_a], [doc_b]])
        assert len(result) == 1
        assert result[0].page_content == "v1"  # 保留分数高的

    def test_deduplicates_by_title_when_no_url(self):
        doc_a = Document(page_content="v1", metadata={"title": "Doc A", "_score": 0.4})
        doc_b = Document(page_content="v2", metadata={"title": "Doc A", "_score": 0.8})
        result = merge_docs([[doc_a], [doc_b]])
        assert len(result) == 1
        assert result[0].page_content == "v2"  # 保留分数高的

    def test_keeps_unique_docs_from_all_hops(self):
        doc1 = Document(page_content="d1", metadata={"url": "http://a.com"})
        doc2 = Document(page_content="d2", metadata={"url": "http://b.com"})
        doc3 = Document(page_content="d3", metadata={"url": "http://c.com"})
        result = merge_docs([[doc1, doc2], [doc2, doc3]])
        assert len(result) == 3

    def test_empty_hops_excluded(self):
        doc1 = Document(page_content="d1", metadata={"url": "http://a.com"})
        result = merge_docs([[doc1], [], []])
        assert len(result) == 1

    def test_all_empty_returns_empty(self):
        assert merge_docs([[], []]) == []

    def test_result_ordered_by_descending_score(self):
        doc_low = Document(page_content="low", metadata={"url": "http://a.com", "_score": 0.3})
        doc_high = Document(page_content="high", metadata={"url": "http://b.com", "_score": 0.9})
        result = merge_docs([[doc_low, doc_high]])
        assert result[0].page_content == "high"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n rag pytest tests/test_multi_hop_rag.py::TestMergeDocs -v
```
预期：`ImportError: cannot import name 'merge_docs'`

- [ ] **Step 3: 实现 `merge_docs`（追加到 `src/pipeline/multi_hop_rag.py` 末尾）**

```python
# 追加到 src/pipeline/multi_hop_rag.py 末尾


def merge_docs(hop_docs: list[list[Document]]) -> list[Document]:
    """合并多跳检索结果，按 url → title 去重，保留分数最高的副本，按分数降序返回。"""
    seen: dict[str, Document] = {}
    for hop in hop_docs:
        for doc in hop:
            key = doc.metadata.get("url") or doc.metadata.get("title") or doc.page_content[:60]
            if key not in seen or _doc_score_value(doc) > _doc_score_value(seen[key]):
                seen[key] = doc
    return sorted(seen.values(), key=_doc_score_value, reverse=True)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
conda run -n rag pytest tests/test_multi_hop_rag.py::TestMergeDocs -v
```
预期：6 passed

- [ ] **Step 5: 提交**

```bash
git add src/pipeline/multi_hop_rag.py tests/test_multi_hop_rag.py
git commit -m "feat: implement merge_docs with url/title dedup"
```

---

## Task 3: `decompose_query` 子问题分解

**Files:**
- Modify: `src/pipeline/multi_hop_rag.py`（在 `merge_docs` 之后追加）
- Modify: `tests/test_multi_hop_rag.py`（在文件末尾追加）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_multi_hop_rag.py` 末尾）**

```python
# 追加到 tests/test_multi_hop_rag.py 末尾
# 注意：MagicMock 已在文件顶部 import，无需重复
from src.pipeline.multi_hop_rag import decompose_query


class TestDecomposeQuery:
    def _make_llm(self, response_text: str):
        llm = MagicMock()
        llm.invoke.return_value.content = response_text
        return llm

    def test_returns_list_of_strings(self):
        llm = self._make_llm('["What is A?", "What is B?"]')
        result = decompose_query("Complex question about A and B", llm)
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    def test_parses_two_sub_questions(self):
        llm = self._make_llm('["Sub Q1", "Sub Q2"]')
        result = decompose_query("question", llm)
        assert result == ["Sub Q1", "Sub Q2"]

    def test_fallback_on_invalid_json(self):
        llm = self._make_llm("I cannot decompose this")
        result = decompose_query("original question", llm)
        assert result == ["original question"]

    def test_fallback_on_non_list_json(self):
        llm = self._make_llm('{"key": "value"}')
        result = decompose_query("original question", llm)
        assert result == ["original question"]

    def test_llm_is_called_once(self):
        llm = self._make_llm('["Q1", "Q2"]')
        decompose_query("test", llm)
        assert llm.invoke.call_count == 1
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n rag pytest tests/test_multi_hop_rag.py::TestDecomposeQuery -v
```
预期：`ImportError: cannot import name 'decompose_query'`

- [ ] **Step 3: 实现 `decompose_query`（追加到 `src/pipeline/multi_hop_rag.py` 末尾）**

```python
# 追加到 src/pipeline/multi_hop_rag.py 末尾
# 注意：json、logging、logger 已在文件顶部定义，无需重复

_DECOMPOSE_PROMPT = (
    "Break the following complex question into 2-3 independent sub-questions "
    "that together cover all the information needed to answer it. "
    "Return ONLY a JSON array of strings, e.g. [\"sub_q1\", \"sub_q2\"]. "
    "No extra text.\n\nQuestion: {question}"
)


def decompose_query(query: str, llm) -> list[str]:
    """LLM 将复杂问题分解为子问题列表。解析失败时降级返回 [query]。"""
    prompt = _DECOMPOSE_PROMPT.format(question=query)
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        parsed = json.loads(text.strip())
        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
            return parsed
        logger.warning("decompose_query: LLM returned non-list JSON, falling back")
    except Exception as exc:
        logger.warning("decompose_query: parse error (%s), falling back to original query", exc)
    return [query]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
conda run -n rag pytest tests/test_multi_hop_rag.py::TestDecomposeQuery -v
```
预期：5 passed

- [ ] **Step 5: 提交**

```bash
git add src/pipeline/multi_hop_rag.py tests/test_multi_hop_rag.py
git commit -m "feat: implement decompose_query with fallback"
```

---

## Task 4: `MultiHopRAGPipeline` 类

**Files:**
- Modify: `src/pipeline/multi_hop_rag.py`（在 `decompose_query` 之后追加）
- Modify: `tests/test_multi_hop_rag.py`（在文件末尾追加）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_multi_hop_rag.py` 末尾）**

```python
# 追加到 tests/test_multi_hop_rag.py 末尾
# 注意：HopResult、MultiHopResult、Document、MagicMock、patch、ChatOpenAI、MockRetriever
#       均已在文件顶部 import，无需重复
from src.pipeline.multi_hop_rag import MultiHopRAGPipeline


class TestMultiHopRAGPipelineRetrieve:
    def test_retrieve_returns_multi_hop_result(self):
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch("src.pipeline.multi_hop_rag.decompose_query", return_value=["Q1", "Q2"]):
            result = pipeline.retrieve("complex question")
        assert isinstance(result, MultiHopResult)

    def test_retrieve_answer_is_empty_string(self):
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch("src.pipeline.multi_hop_rag.decompose_query", return_value=["Q1"]):
            result = pipeline.retrieve("question")
        assert result.answer == ""

    def test_retrieve_fills_sub_questions(self):
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch("src.pipeline.multi_hop_rag.decompose_query", return_value=["Q1", "Q2"]):
            result = pipeline.retrieve("question")
        assert result.sub_questions == ["Q1", "Q2"]

    def test_retrieve_hop_results_count_matches_sub_questions(self):
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch("src.pipeline.multi_hop_rag.decompose_query", return_value=["Q1", "Q2", "Q3"]):
            result = pipeline.retrieve("question")
        assert len(result.hop_results) == 3

    def test_retrieve_merged_docs_not_empty(self):
        # MockRetriever 每次返回 3 个不同 page_content 的文档（无 url/title），
        # merge_docs 用 page_content[:60] 作为 key，两个子问题各返回相同 3 个内容 → 去重后仍为 3
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch("src.pipeline.multi_hop_rag.decompose_query", return_value=["Q1", "Q2"]):
            result = pipeline.retrieve("question")
        assert len(result.merged_docs) > 0


class TestMultiHopRAGPipelineStream:
    def _make_mhr(self, docs=None):
        return MultiHopResult(
            question="q",
            sub_questions=["Q1"],
            hop_results=[HopResult("Q1", docs or [])],
            merged_docs=docs or [Document(page_content="ctx", metadata={})],
            answer="",
        )

    def test_stream_yields_strings(self):
        """stream() 通过 LCEL chain 调用 _gen_llm.stream()，这里 patch chain 的 stream 方法。"""
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        mhr = self._make_mhr()
        tokens = ["Hello", " world"]

        # patch LCEL chain 最终调用的 _gen_llm.stream，而非 ChatOpenAI 类方法
        with patch.object(pipeline._gen_llm, "stream", return_value=iter(tokens)):
            result = list(pipeline.stream(mhr))
        # StrOutputParser 对字符串直接透传，结果与 tokens 相同
        assert result == tokens

    def test_stream_is_generator(self):
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        mhr = self._make_mhr()
        with patch.object(pipeline._gen_llm, "stream", return_value=iter([])):
            gen = pipeline.stream(mhr)
        # yield from 产生 generator
        import types
        assert isinstance(gen, types.GeneratorType)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n rag pytest tests/test_multi_hop_rag.py::TestMultiHopRAGPipelineRetrieve tests/test_multi_hop_rag.py::TestMultiHopRAGPipelineStream -v
```
预期：`ImportError: cannot import name 'MultiHopRAGPipeline'`

- [ ] **Step 3: 实现 `MultiHopRAGPipeline`（追加到 `src/pipeline/multi_hop_rag.py` 末尾）**

> 注意：所有 import（`ChatOpenAI`、`ChatPromptTemplate`、`StrOutputParser`、`load_dotenv`）已在 Task 1 的文件顶部写入，无需重复。

```python
# 追加到 src/pipeline/multi_hop_rag.py 末尾

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            'If the answer cannot be found in the context, respond with "I don\'t know."',
        ),
        (
            "human",
            "Sub-questions used for retrieval:\n{sub_questions}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\nAnswer:",
        ),
    ]
)


def _format_sub_questions(sub_questions: list[str]) -> str:
    return "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class MultiHopRAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL):
        self.retriever = retriever
        self._decompose_llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self._gen_llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def retrieve(self, query: str) -> MultiHopResult:
        """分解问题 → 各子问题独立检索 → 合并去重。不调用生成 LLM。"""
        sub_questions = decompose_query(query, self._decompose_llm)
        hop_results = []
        for sub_q in sub_questions:
            docs = self.retriever.invoke(sub_q)
            hop_results.append(HopResult(sub_question=sub_q, docs=docs))
        merged = merge_docs([hr.docs for hr in hop_results])
        return MultiHopResult(
            question=query,
            sub_questions=sub_questions,
            hop_results=hop_results,
            merged_docs=merged,
            answer="",
        )

    def stream(self, result: MultiHopResult) -> Iterator[str]:
        """流式输出最终答案。prompt 中包含子问题以提供推理上下文。"""
        yield from (
            _ANSWER_PROMPT | self._gen_llm | StrOutputParser()
        ).stream(
            {
                "question": result.question,
                "sub_questions": _format_sub_questions(result.sub_questions),
                "context": _format_docs(result.merged_docs),
            }
        )
```

- [ ] **Step 4: 运行全部 multi_hop 测试**

```bash
conda run -n rag pytest tests/test_multi_hop_rag.py -v
```
预期：全部 passed

- [ ] **Step 5: 确认现有测试不受影响**

```bash
conda run -n rag pytest tests/ -v --ignore=tests/integration
```
预期：全部 passed

- [ ] **Step 6: 提交**

```bash
git add src/pipeline/multi_hop_rag.py tests/test_multi_hop_rag.py
git commit -m "feat: implement MultiHopRAGPipeline with retrieve() and stream()"
```

---

## Task 5: UI 侧边栏 Pipeline 模式切换

**Files:**
- Modify: `src/ui/app.py`

> Streamlit UI 无法直接单元测试，通过 smoke test 手动验证。`build_multi_hop_pipeline` 是纯 Python 函数，但其正确性由 Task 4 对 `MultiHopRAGPipeline` 的测试覆盖。

- [ ] **Step 1: 新增 import**

在 `src/ui/app.py` 顶部 import 块中追加：

```python
from src.pipeline.multi_hop_rag import MultiHopRAGPipeline, MultiHopResult
```

- [ ] **Step 2: 新增 `build_multi_hop_pipeline` 辅助函数**

在 `build_pipeline` 函数（约第 102 行）之后添加：

```python
def build_multi_hop_pipeline(
    model_key: str, use_bm25: bool, enable_reranker: bool
) -> MultiHopRAGPipeline:
    retriever = _build_retriever(model_key, use_bm25, enable_reranker)
    return MultiHopRAGPipeline(retriever)
```

- [ ] **Step 3: 侧边栏新增 Pipeline radio**

在 `src/ui/app.py` 的**第一个** `with st.sidebar:` 块（Settings 区块，约第 132 行）中，在 `mode = st.radio(...)` 之后追加：

```python
    pipeline_mode = st.radio("Pipeline", ["Single-hop", "Multi-hop"], index=0)
```

- [ ] **Step 4: Compare 复选框加 disabled 条件**

找到 `compare_mode = st.checkbox("Compare all 4 strategies", value=False)` 这一行，修改为：

```python
    compare_mode = st.checkbox(
        "Compare all 4 strategies",
        value=False,
        disabled=(pipeline_mode == "Multi-hop"),
    )
```

- [ ] **Step 5: 提交**

```bash
git add src/ui/app.py
git commit -m "feat: add Pipeline mode radio and disable compare in multi-hop"
```

---

## Task 6: UI Session State 与 Ask 按钮多跳分支

**Files:**
- Modify: `src/ui/app.py`

- [ ] **Step 1: 扩展过期状态清除块**

找到过期状态清除块（完整条件如下，约第 195 行）：

```python
if (
    previous_query_key is not None
    and previous_query_key != current_query_key
    and not ask_clicked
):
```

在其内部末尾追加两行：

```python
    st.session_state["multi_hop_result"] = None
    st.session_state["pending_mh_stream"] = False
```

- [ ] **Step 2: 在 `if ask_clicked` 块中新增多跳分支**

找到（约第 207 行）：

```python
if ask_clicked and query and query.strip():
    if compare_mode:
```

修改为：

```python
if ask_clicked and query and query.strip():
    if pipeline_mode == "Multi-hop":
        with st.spinner("Decomposing question and retrieving…"):
            p = build_multi_hop_pipeline(model_key, use_bm25, enable_reranker)
            mh_result = p.retrieve(query)
        st.session_state["multi_hop_result"] = mh_result
        st.session_state["pending_mh_stream"] = True
        st.session_state["result"] = None
        st.session_state["compare_results"] = None
        st.session_state["pending_stream"] = False
    elif compare_mode:
```

- [ ] **Step 3: 在 `if ask_clicked` 块末尾新增 `last_pipeline_mode` 存储**

找到 `if ask_clicked` 块末尾存储 `last_*` 键的部分（`st.session_state["last_query"] = query` 等），追加：

```python
    st.session_state["last_pipeline_mode"] = pipeline_mode
```

- [ ] **Step 4: 新增 session state 读取**

在现有读取块（约第 232 行，`compare_results = st.session_state.get(...)` 等）之后追加：

```python
last_pipeline_mode = st.session_state.get("last_pipeline_mode", "Single-hop")
multi_hop_result = st.session_state.get("multi_hop_result")
pending_mh_stream = st.session_state.get("pending_mh_stream", False)
```

- [ ] **Step 5: 提交**

```bash
git add src/ui/app.py
git commit -m "feat: add multi-hop session state and ask button branch"
```

---

## Task 7: UI 多跳输出渲染

**Files:**
- Modify: `src/ui/app.py`

- [ ] **Step 1: 新增多跳渲染分支**

在 `src/ui/app.py` 输出区块，找到 `if compare_results:` 块（约第 243 行）。在 `if compare_results:` 之后、`elif result:` 之前，插入多跳分支：

```python
elif last_pipeline_mode == "Multi-hop" and multi_hop_result is not None:
    st.divider()

    # ── Sub-question Breakdown ────────────────────────────────────────────
    st.subheader("Sub-question Breakdown")
    for i, hr in enumerate(multi_hop_result.hop_results, 1):
        label = f"[{i}] {hr.sub_question}"
        with st.expander(label, expanded=False):
            if hr.docs:
                for j, doc in enumerate(hr.docs, 1):
                    title = doc.metadata.get("title", "Untitled")[:60]
                    st.markdown(f"**{j}.** {title}  \n`{_doc_score(doc)}`")
            else:
                st.caption("No documents retrieved for this sub-question.")

    # ── Answer ────────────────────────────────────────────────────────────
    st.subheader("Answer")
    if pending_mh_stream:
        p = build_multi_hop_pipeline(
            last_model_key, last_use_bm25, last_enable_reranker
        )
        try:
            answer = st.write_stream(p.stream(multi_hop_result))
            multi_hop_result.answer = answer
            st.session_state["multi_hop_result"] = multi_hop_result
        finally:
            st.session_state["pending_mh_stream"] = False  # 无论成功与否都清除
    else:
        st.markdown(multi_hop_result.answer)

    # ── Sources ───────────────────────────────────────────────────────────
    st.divider()
    docs = multi_hop_result.merged_docs
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

    # ── Preset-only evaluation ────────────────────────────────────────────
    if last_mode == "Preset questions" and last_qa is not None:
        st.divider()
        st.subheader("Evaluation")
        ground_truth: str = last_qa.get("answer", "")
        evidence_list: list[dict] = last_qa.get("evidence_list", [])
        f1 = token_f1(multi_hop_result.answer, ground_truth)
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
                    doc.metadata.get("title", "").strip().lower()
                    for doc in docs  # docs = multi_hop_result.merged_docs（上方已赋值）
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
```

- [ ] **Step 2: 运行全部单元测试**

```bash
conda run -n rag pytest tests/ -v --ignore=tests/integration
```
预期：全部 passed

- [ ] **Step 3: 手动 smoke test**

```bash
conda run -n rag streamlit run src/ui/app.py
```

验证清单：
- [ ] 侧边栏出现 Pipeline radio（Single-hop / Multi-hop）
- [ ] 切换到 Multi-hop 时 Compare all 4 复选框变灰
- [ ] 选择预设问题，点击 Ask，出现 spinner "Decomposing question and retrieving…"
- [ ] spinner 结束后出现 "Sub-question Breakdown" 区块，展示 2-3 个子问题
- [ ] 每个子问题可展开查看来源列表
- [ ] Answer 区块流式输出答案
- [ ] Sources 区块展示 merged_docs
- [ ] 切换问题后旧输出被清除（stale state 清除逻辑生效）
- [ ] 切回 Single-hop 模式后行为与之前完全相同

- [ ] **Step 4: 提交**

```bash
git add src/ui/app.py
git commit -m "feat: add multi-hop output rendering in Streamlit UI"
```

---

## Task 8: 最终整合验证

- [ ] **Step 1: 运行全部单元测试**

```bash
conda run -n rag pytest tests/ -v --ignore=tests/integration
```
预期：全部 passed，无新增失败

- [ ] **Step 2: 确认单跳 smoke test 不受影响**

```bash
conda run -n rag python -m src.pipeline.demo
```
预期：正常运行

- [ ] **Step 3: 最终提交**

```bash
git add -A
git commit -m "feat: complete Milestone 6 multi-hop query decomposition"
```
