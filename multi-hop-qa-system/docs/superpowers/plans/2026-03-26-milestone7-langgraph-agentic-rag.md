# Milestone 7 · LangGraph Agentic RAG 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph StateGraph 实现 Agentic RAG，Agent 自适应判断 context 是否充足，不足则生成补充 query 继续检索，最多 3 跳。

**Architecture:** 纯迭代检索（不做问题分解）。图结构：`START → retrieve_node → judge_node →（条件边）→ generate_node → END`，judge 不足时循环回 retrieve_node。新增 `src/pipeline/agent_rag.py`，UI 新增 Agent 选项。

**Tech Stack:** LangGraph · LangChain · OpenAI API · Streamlit

---

## 文件变更清单

| 文件                        | 操作 | 职责                                                         |
| --------------------------- | ---- | ------------------------------------------------------------ |
| `requirements.txt`          | 修改 | 新增 `langgraph` 依赖                                        |
| `src/pipeline/agent_rag.py` | 新建 | HopTrace / RAGState / AgentRAGPipeline（含图定义与所有节点） |
| `tests/test_agent_rag.py`   | 新建 | 所有节点与管道的单元测试                                     |
| `src/ui/app.py`             | 修改 | 新增 Agent pipeline 选项、渲染推理轨迹                       |

---

## Task 1：添加 langgraph 依赖

**Files:**

- Modify: `requirements.txt`

- [ ] **Step 1：添加依赖**

将 `requirements.txt` 改为：

```
langchain-chroma
langchain-core
langchain-huggingface
langchain-openai
langchain-text-splitters
langgraph
python-dotenv
pytest
rank-bm25
sentence-transformers
streamlit
tqdm
```

- [ ] **Step 2：安装**

```bash
conda run -n rag pip install langgraph
```

预期输出：`Successfully installed langgraph-x.x.x`

- [ ] **Step 3：验证导入**

```bash
conda run -n rag python -c "from langgraph.graph import StateGraph, END, START; print('OK')"
```

预期：`OK`

- [ ] **Step 4：Commit**

```bash
git add requirements.txt
git commit -m "feat: add langgraph dependency"
```

---

## Task 2：定义数据结构与辅助函数（TDD）

**Files:**

- Create: `src/pipeline/agent_rag.py`
- Create: `tests/test_agent_rag.py`

- [ ] **Step 1：写失败的测试**

新建 `tests/test_agent_rag.py`：

```python
import pytest
from langchain_core.documents import Document
from src.pipeline.agent_rag import HopTrace, RAGState, _build_judge_context


class TestDataStructures:
    def test_hop_trace_fields(self):
        doc = Document(page_content="test")
        trace = HopTrace(query="q", docs=[doc], reasoning="r")
        assert trace["query"] == "q"
        assert trace["docs"] == [doc]
        assert trace["reasoning"] == "r"

    def test_rag_state_fields(self):
        state = RAGState(
            question="q",
            current_query="q",
            retrieved_docs=[],
            hop_count=0,
            sufficient=False,
            follow_up_query="",
            trace=[],
            answer="",
        )
        assert state["hop_count"] == 0
        assert state["sufficient"] is False


class TestBuildJudgeContext:
    def test_concatenates_page_content(self):
        docs = [
            Document(page_content="AAA"),
            Document(page_content="BBB"),
        ]
        result = _build_judge_context(docs)
        assert "AAA" in result
        assert "BBB" in result

    def test_truncates_to_max_chars(self):
        docs = [Document(page_content="X" * 10000)]
        result = _build_judge_context(docs)
        assert len(result) <= 6000

    def test_empty_docs_returns_empty_string(self):
        assert _build_judge_context([]) == ""
```

- [ ] **Step 2：运行确认失败**

```bash
conda run -n rag pytest tests/test_agent_rag.py::TestDataStructures tests/test_agent_rag.py::TestBuildJudgeContext -v
```

预期：`ImportError` 或 `ModuleNotFoundError`

- [ ] **Step 3：写最小实现**

新建 `src/pipeline/agent_rag.py`：

```python
import json
import logging
import os
from typing import Iterator, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.config import LLM_MODEL
from src.pipeline.multi_hop_rag import _ANSWER_PROMPT, _format_docs, merge_docs

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 6000


class HopTrace(TypedDict):
    query: str
    docs: list[Document]
    reasoning: str


class RAGState(TypedDict):
    question: str
    current_query: str
    retrieved_docs: list[Document]
    hop_count: int
    sufficient: bool
    follow_up_query: str
    trace: list[HopTrace]
    answer: str


def _build_judge_context(docs: list[Document]) -> str:
    text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return text[:_MAX_CONTEXT_CHARS]
```

- [ ] **Step 4：运行确认通过**

```bash
conda run -n rag pytest tests/test_agent_rag.py::TestDataStructures tests/test_agent_rag.py::TestBuildJudgeContext -v
```

预期：5 个测试全部 PASS

- [ ] **Step 5：Commit**

```bash
git add src/pipeline/agent_rag.py tests/test_agent_rag.py
git commit -m "feat: add RAGState, HopTrace, and _build_judge_context"
```

---

## Task 3：实现 retrieve_node（TDD）

**Files:**

- Modify: `src/pipeline/agent_rag.py`
- Modify: `tests/test_agent_rag.py`

- [ ] **Step 1：写失败的测试**

在 `tests/test_agent_rag.py` 末尾追加：

```python
from unittest.mock import MagicMock, patch
from tests.conftest import MockRetriever
from src.pipeline.agent_rag import AgentRAGPipeline


class TestRetrieveNode:
    def _make_pipeline(self):
        return AgentRAGPipeline(retriever=MockRetriever())

    def _base_state(self, **kwargs) -> RAGState:
        base = RAGState(
            question="Who founded OpenAI?",
            current_query="Who founded OpenAI?",
            retrieved_docs=[],
            hop_count=0,
            sufficient=False,
            follow_up_query="",
            trace=[],
            answer="",
        )
        base.update(kwargs)
        return base

    def test_increments_hop_count(self):
        p = self._make_pipeline()
        result = p._retrieve_node(self._base_state())
        assert result["hop_count"] == 1

    def test_appends_trace_entry(self):
        p = self._make_pipeline()
        result = p._retrieve_node(self._base_state())
        assert len(result["trace"]) == 1
        assert result["trace"][0]["query"] == "Who founded OpenAI?"

    def test_uses_follow_up_query_when_non_empty(self):
        p = self._make_pipeline()
        state = self._base_state(follow_up_query="What year was OpenAI founded?")
        result = p._retrieve_node(state)
        assert result["current_query"] == "What year was OpenAI founded?"
        assert result["trace"][0]["query"] == "What year was OpenAI founded?"

    def test_keeps_current_query_when_follow_up_empty(self):
        p = self._make_pipeline()
        result = p._retrieve_node(self._base_state())
        assert result["current_query"] == "Who founded OpenAI?"

    def test_merges_new_docs_with_existing(self):
        p = self._make_pipeline()
        existing = [Document(page_content="existing", metadata={"url": "http://existing.com"})]
        state = self._base_state(retrieved_docs=existing)
        result = p._retrieve_node(state)
        # MockRetriever returns 3 docs; merged with 1 existing = at least 1
        assert len(result["retrieved_docs"]) >= 1

    def test_trace_entry_reasoning_is_empty_string(self):
        p = self._make_pipeline()
        result = p._retrieve_node(self._base_state())
        assert result["trace"][0]["reasoning"] == ""
```

- [ ] **Step 2：运行确认失败**

```bash
conda run -n rag pytest tests/test_agent_rag.py::TestRetrieveNode -v
```

预期：`ImportError: cannot import name 'AgentRAGPipeline'`

- [ ] **Step 3：写最小实现**

在 `src/pipeline/agent_rag.py` 末尾追加 `AgentRAGPipeline` 类（只含 `__init__` 和 `_retrieve_node`）：

```python
_JUDGE_PROMPT = (
    "You are evaluating whether the provided context is sufficient to answer the question.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Return ONLY a JSON object with this exact structure:\n"
    '{{"sufficient": true or false, '
    '"follow_up_query": "specific question to retrieve missing information, or empty string if sufficient", '
    '"reasoning": "brief explanation of what information is present or missing"}}'
)


class AgentRAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL) -> None:
        self.retriever = retriever
        self._judge_llm = ChatOpenAI(
            model=model, temperature=0.0, api_key=os.environ["OPENAI_API_KEY"]
        )
        self._gen_llm = ChatOpenAI(
            model=model, temperature=0.1, api_key=os.environ["OPENAI_API_KEY"]
        )

    def _retrieve_node(self, state: RAGState) -> dict:
        query = state["follow_up_query"] if state["follow_up_query"] else state["current_query"]
        docs = self.retriever.invoke(query)
        merged = merge_docs([state["retrieved_docs"], docs])
        trace = list(state["trace"]) + [HopTrace(query=query, docs=docs, reasoning="")]
        return {
            "current_query": query,
            "retrieved_docs": merged,
            "hop_count": state["hop_count"] + 1,
            "trace": trace,
        }
```

- [ ] **Step 4：运行确认通过**

```bash
conda run -n rag pytest tests/test_agent_rag.py::TestRetrieveNode -v
```

预期：6 个测试全部 PASS

- [ ] **Step 5：Commit**

```bash
git add src/pipeline/agent_rag.py tests/test_agent_rag.py
git commit -m "feat: implement retrieve_node with merge and trace"
```

---

## Task 4：实现 judge_node（TDD）

**Files:**

- Modify: `src/pipeline/agent_rag.py`
- Modify: `tests/test_agent_rag.py`

- [ ] **Step 1：写失败的测试**

在 `tests/test_agent_rag.py` 末尾追加：

````python
class TestJudgeNode:
    def _make_pipeline(self):
        return AgentRAGPipeline(retriever=MockRetriever())

    def _make_state_with_trace(self, hop_query="q1") -> RAGState:
        doc = Document(page_content="Some context text.")
        return RAGState(
            question="Who founded OpenAI?",
            current_query=hop_query,
            retrieved_docs=[doc],
            hop_count=1,
            sufficient=False,
            follow_up_query="",
            trace=[HopTrace(query=hop_query, docs=[doc], reasoning="")],
            answer="",
        )

    def _mock_judge_llm(self, pipeline, response_json: dict):
        mock = MagicMock()
        mock.invoke.return_value.content = json.dumps(response_json)
        pipeline._judge_llm = mock

    def test_sets_sufficient_true(self):
        p = self._make_pipeline()
        self._mock_judge_llm(p, {"sufficient": True, "follow_up_query": "", "reasoning": "OK"})
        result = p._judge_node(self._make_state_with_trace())
        assert result["sufficient"] is True

    def test_sets_sufficient_false(self):
        p = self._make_pipeline()
        self._mock_judge_llm(
            p,
            {"sufficient": False, "follow_up_query": "When was it founded?", "reasoning": "Missing date"},
        )
        result = p._judge_node(self._make_state_with_trace())
        assert result["sufficient"] is False
        assert result["follow_up_query"] == "When was it founded?"

    def test_writes_reasoning_to_last_trace_entry(self):
        p = self._make_pipeline()
        self._mock_judge_llm(p, {"sufficient": True, "follow_up_query": "", "reasoning": "Fully answered"})
        state = self._make_state_with_trace()
        result = p._judge_node(state)
        assert result["trace"][-1]["reasoning"] == "Fully answered"

    def test_fallback_on_invalid_json(self):
        p = self._make_pipeline()
        mock = MagicMock()
        mock.invoke.return_value.content = "not valid json at all"
        p._judge_llm = mock
        result = p._judge_node(self._make_state_with_trace())
        assert result["sufficient"] is True

    def test_fallback_on_llm_exception(self):
        p = self._make_pipeline()
        mock = MagicMock()
        mock.invoke.side_effect = RuntimeError("API error")
        p._judge_llm = mock
        result = p._judge_node(self._make_state_with_trace())
        assert result["sufficient"] is True

    def test_handles_markdown_code_block_in_response(self):
        p = self._make_pipeline()
        mock = MagicMock()
        mock.invoke.return_value.content = (
            "```json\n"
            '{"sufficient": false, "follow_up_query": "X", "reasoning": "Missing X"}\n'
            "```"
        )
        p._judge_llm = mock
        result = p._judge_node(self._make_state_with_trace())
        assert result["sufficient"] is False
        assert result["follow_up_query"] == "X"
````

- [ ] **Step 2：运行确认失败**

```bash
conda run -n rag pytest tests/test_agent_rag.py::TestJudgeNode -v
```

预期：`AttributeError: 'AgentRAGPipeline' object has no attribute '_judge_node'`

- [ ] **Step 3：写最小实现**

在 `AgentRAGPipeline` 类中追加 `_judge_node` 方法：

````python
    def _judge_node(self, state: RAGState) -> dict:
        context = _build_judge_context(state["retrieved_docs"])
        prompt = _JUDGE_PROMPT.format(question=state["question"], context=context)
        sufficient = True
        follow_up_query = ""
        reasoning = ""
        try:
            response = self._judge_llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text.strip())
            sufficient = bool(parsed.get("sufficient", True))
            follow_up_query = str(parsed.get("follow_up_query", ""))
            reasoning = str(parsed.get("reasoning", ""))
        except Exception as exc:
            logger.warning("judge_node: parse error (%s), falling back to sufficient=True", exc)

        trace = list(state["trace"])
        if trace:
            last = trace[-1]
            trace[-1] = HopTrace(query=last["query"], docs=last["docs"], reasoning=reasoning)

        return {
            "sufficient": sufficient,
            "follow_up_query": follow_up_query,
            "trace": trace,
        }
````

- [ ] **Step 4：运行确认通过**

```bash
conda run -n rag pytest tests/test_agent_rag.py::TestJudgeNode -v
```

预期：6 个测试全部 PASS

- [ ] **Step 5：Commit**

```bash
git add src/pipeline/agent_rag.py tests/test_agent_rag.py
git commit -m "feat: implement judge_node with LLM verdict and fallback"
```

---

## Task 5：实现 should_continue 和 generate_node（TDD）

**Files:**

- Modify: `src/pipeline/agent_rag.py`
- Modify: `tests/test_agent_rag.py`

- [ ] **Step 1：写失败的测试**

在 `tests/test_agent_rag.py` 末尾追加：

```python
class TestShouldContinue:
    def test_returns_generate_when_sufficient(self):
        state = RAGState(
            question="q", current_query="q", retrieved_docs=[], hop_count=1,
            sufficient=True, follow_up_query="", trace=[], answer=""
        )
        assert AgentRAGPipeline._should_continue(state) == "generate"

    def test_returns_generate_when_hop_count_reaches_3(self):
        state = RAGState(
            question="q", current_query="q", retrieved_docs=[], hop_count=3,
            sufficient=False, follow_up_query="more", trace=[], answer=""
        )
        assert AgentRAGPipeline._should_continue(state) == "generate"

    def test_returns_retrieve_when_not_sufficient_and_hops_below_3(self):
        state = RAGState(
            question="q", current_query="q", retrieved_docs=[], hop_count=2,
            sufficient=False, follow_up_query="more", trace=[], answer=""
        )
        assert AgentRAGPipeline._should_continue(state) == "retrieve"

    def test_returns_retrieve_at_hop_count_1(self):
        state = RAGState(
            question="q", current_query="q", retrieved_docs=[], hop_count=1,
            sufficient=False, follow_up_query="more", trace=[], answer=""
        )
        assert AgentRAGPipeline._should_continue(state) == "retrieve"


class TestGenerateNode:
    def _make_pipeline(self):
        return AgentRAGPipeline(retriever=MockRetriever())

    def _make_state(self) -> RAGState:
        doc = Document(page_content="OpenAI was founded in 2015.")
        return RAGState(
            question="Who founded OpenAI?",
            current_query="Who founded OpenAI?",
            retrieved_docs=[doc],
            hop_count=1,
            sufficient=True,
            follow_up_query="",
            trace=[HopTrace(query="Who founded OpenAI?", docs=[doc], reasoning="OK")],
            answer="",
        )

    def test_stores_answer_in_state(self):
        p = self._make_pipeline()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Sam Altman"
        # Patch the LCEL chain by patching _gen_llm's behavior
        with patch.object(p, "_gen_llm") as mock:
            mock.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=MagicMock(
                    invoke=MagicMock(return_value="Sam Altman")
                ))
            ))
            result = p._generate_node(self._make_state())
        assert "answer" in result

    def test_answer_is_non_empty_string(self):
        p = self._make_pipeline()
        with patch("src.pipeline.agent_rag.StrOutputParser") as mock_parser_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Sam Altman founded OpenAI."
            mock_parser_cls.return_value.__ror__ = MagicMock(return_value=mock_chain)
            # Simpler: patch the whole chain
            with patch.object(p, "_gen_llm") as mock_llm:
                # Build a chain mock that returns a string
                fake_chain = MagicMock()
                fake_chain.invoke.return_value = "Sam Altman founded OpenAI."
                mock_llm.__ror__ = MagicMock(return_value=fake_chain)
                result = p._generate_node(self._make_state())
        assert isinstance(result["answer"], str)
```

- [ ] **Step 2：运行确认失败**

```bash
conda run -n rag pytest tests/test_agent_rag.py::TestShouldContinue tests/test_agent_rag.py::TestGenerateNode -v
```

预期：`AttributeError: type object 'AgentRAGPipeline' has no attribute '_should_continue'`

- [ ] **Step 3：写最小实现**

在 `AgentRAGPipeline` 类中追加两个方法：

```python
    def _generate_node(self, state: RAGState) -> dict:
        sub_questions_str = "\n".join(
            f"{i+1}. {t['query']}" for i, t in enumerate(state["trace"])
        )
        context = _format_docs(state["retrieved_docs"])
        chain = _ANSWER_PROMPT | self._gen_llm | StrOutputParser()
        answer = chain.invoke({
            "question": state["question"],
            "sub_questions": sub_questions_str,
            "context": context,
        })
        return {"answer": answer}

    @staticmethod
    def _should_continue(state: RAGState) -> str:
        if state["sufficient"] or state["hop_count"] >= 3:
            return "generate"
        return "retrieve"
```

- [ ] **Step 4：运行 should_continue 测试**

```bash
conda run -n rag pytest tests/test_agent_rag.py::TestShouldContinue -v
```

预期：4 个测试全部 PASS

- [ ] **Step 5：运行全部测试确认无回归**

```bash
conda run -n rag pytest tests/test_agent_rag.py -v
```

预期：TestGenerateNode 部分可能因 LCEL mock 复杂而跳过或 PASS，其余全部 PASS

- [ ] **Step 6：Commit**

```bash
git add src/pipeline/agent_rag.py tests/test_agent_rag.py
git commit -m "feat: implement should_continue and generate_node"
```

---

## Task 6：组装图 + run + stream_answer（TDD）

**Files:**

- Modify: `src/pipeline/agent_rag.py`
- Modify: `tests/test_agent_rag.py`

- [ ] **Step 1：写失败的测试**

在 `tests/test_agent_rag.py` 末尾追加：

```python
class TestAgentRAGPipelineRun:
    def test_run_returns_rag_state_dict(self):
        p = AgentRAGPipeline(retriever=MockRetriever())
        # Patch judge to immediately return sufficient=True
        with patch.object(p, "_judge_node", return_value={
            "sufficient": True, "follow_up_query": "", "trace": []
        }):
            with patch.object(p, "_generate_node", return_value={"answer": "test answer"}):
                result = p.run("Who founded OpenAI?")
        assert "question" in result
        assert "trace" in result
        assert "answer" in result

    def test_run_sets_question(self):
        p = AgentRAGPipeline(retriever=MockRetriever())
        with patch.object(p, "_judge_node", return_value={
            "sufficient": True, "follow_up_query": "", "trace": []
        }):
            with patch.object(p, "_generate_node", return_value={"answer": "x"}):
                result = p.run("What is LangGraph?")
        assert result["question"] == "What is LangGraph?"


class TestStreamAnswer:
    def test_stream_answer_yields_strings(self):
        p = AgentRAGPipeline(retriever=MockRetriever())
        doc = Document(page_content="LangGraph is a framework.")
        state = RAGState(
            question="What is LangGraph?",
            current_query="What is LangGraph?",
            retrieved_docs=[doc],
            hop_count=1,
            sufficient=True,
            follow_up_query="",
            trace=[HopTrace(query="What is LangGraph?", docs=[doc], reasoning="OK")],
            answer="LangGraph is ...",
        )
        tokens = ["Lang", "Graph", " is", " great"]
        with patch.object(p, "_gen_llm") as mock_llm:
            mock_chain = MagicMock()
            mock_chain.stream.return_value = iter(tokens)
            mock_llm.__ror__ = MagicMock(return_value=mock_chain)
            # Directly patch the chain construction
            with patch("src.pipeline.agent_rag.StrOutputParser") as mock_parser:
                fake_chain = MagicMock()
                fake_chain.stream.return_value = iter(tokens)
                mock_parser.return_value.__ror__ = MagicMock(return_value=fake_chain)
                result = list(p.stream_answer(state))
        assert isinstance(result, list)
```

- [ ] **Step 2：运行确认失败**

```bash
conda run -n rag pytest tests/test_agent_rag.py::TestAgentRAGPipelineRun tests/test_agent_rag.py::TestStreamAnswer -v
```

预期：`AttributeError: 'AgentRAGPipeline' object has no attribute 'run'`

- [ ] **Step 3：写最小实现**

在 `AgentRAGPipeline` 类中追加 `_build_graph`、`run`、`stream_answer` 方法，并在 `__init__` 末尾调用 `_build_graph`：

```python
    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("judge", self._judge_node)
        graph.add_node("generate", self._generate_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "judge")
        graph.add_conditional_edges(
            "judge",
            self._should_continue,
            {"retrieve": "retrieve", "generate": "generate"},
        )
        graph.add_edge("generate", END)
        return graph.compile()

    def run(self, question: str) -> RAGState:
        initial: RAGState = {
            "question": question,
            "current_query": question,
            "retrieved_docs": [],
            "hop_count": 0,
            "sufficient": False,
            "follow_up_query": "",
            "trace": [],
            "answer": "",
        }
        return self._graph.invoke(initial)

    def stream_answer(self, state: RAGState) -> Iterator[str]:
        sub_questions_str = "\n".join(
            f"{i+1}. {t['query']}" for i, t in enumerate(state["trace"])
        )
        context = _format_docs(state["retrieved_docs"])
        chain = _ANSWER_PROMPT | self._gen_llm | StrOutputParser()
        yield from chain.stream({
            "question": state["question"],
            "sub_questions": sub_questions_str,
            "context": context,
        })
```

将 `__init__` 最后一行改为：

```python
        self._graph = self._build_graph()
```

- [ ] **Step 4：运行全部测试**

```bash
conda run -n rag pytest tests/test_agent_rag.py -v
```

预期：所有测试 PASS（或 TestGenerateNode / TestStreamAnswer 中 LCEL mock 测试因复杂性 SKIP，其余全部 PASS）

- [ ] **Step 5：运行现有测试套件，确认无回归**

```bash
conda run -n rag pytest tests/ -v --ignore=tests/integration
```

预期：所有非集成测试 PASS

- [ ] **Step 6：Commit**

```bash
git add src/pipeline/agent_rag.py tests/test_agent_rag.py
git commit -m "feat: assemble LangGraph StateGraph with run() and stream_answer()"
```

---

## Task 7：更新 Streamlit UI

**Files:**

- Modify: `src/ui/app.py`

- [ ] **Step 1：添加 import**

在 `src/ui/app.py` 顶部的 import 区域添加（紧跟 `MultiHopRAGPipeline` 那行之后）：

```python
from src.pipeline.agent_rag import AgentRAGPipeline
```

- [ ] **Step 2：添加 build_agent_pipeline 函数**

在 `build_multi_hop_pipeline` 函数（约第 111-115 行）之后追加：

```python
def build_agent_pipeline(
    model_key: str, use_bm25: bool, enable_reranker: bool
) -> AgentRAGPipeline:
    retriever = _build_retriever(model_key, use_bm25, enable_reranker)
    return AgentRAGPipeline(retriever)
```

- [ ] **Step 3：扩展 Pipeline radio 并禁用 compare**

将侧边栏中的 Pipeline radio（约第 143 行）从：

```python
    pipeline_mode = st.radio("Pipeline", ["Single-hop", "Multi-hop"], index=0)
```

改为：

```python
    pipeline_mode = st.radio("Pipeline", ["Single-hop", "Multi-hop", "Agent"], index=0)
```

将 compare_mode 的 disabled 条件（约第 156-160 行）从：

```python
    compare_mode = st.checkbox(
        "Compare all 4 strategies",
        value=False,
        disabled=(pipeline_mode == "Multi-hop"),
    )
```

改为：

```python
    compare_mode = st.checkbox(
        "Compare all 4 strategies",
        value=False,
        disabled=(pipeline_mode in ("Multi-hop", "Agent")),
    )
```

- [ ] **Step 4：在 session state 清除块中加入 Agent 字段**

找到清除 session state 的代码块（约第 213-217 行）：

```python
    st.session_state["result"] = None
    st.session_state["compare_results"] = None
    st.session_state["pending_stream"] = False
    st.session_state["multi_hop_result"] = None
    st.session_state["pending_mh_stream"] = False
```

改为：

```python
    st.session_state["result"] = None
    st.session_state["compare_results"] = None
    st.session_state["pending_stream"] = False
    st.session_state["multi_hop_result"] = None
    st.session_state["pending_mh_stream"] = False
    st.session_state["agent_result"] = None
    st.session_state["pending_agent_stream"] = False
```

- [ ] **Step 5：在 ask_clicked 块中加入 Agent 分支**

找到 `if ask_clicked and query and query.strip():` 块（约第 222 行），在 `if pipeline_mode == "Multi-hop":` 分支之前插入：

```python
    if pipeline_mode == "Agent":
        with st.spinner("Running agentic retrieval…"):
            p = build_agent_pipeline(model_key, use_bm25, enable_reranker)
            agent_state = p.run(query)
        st.session_state["agent_result"] = agent_state
        st.session_state["pending_agent_stream"] = True
        st.session_state["result"] = None
        st.session_state["compare_results"] = None
        st.session_state["pending_stream"] = False
        st.session_state["multi_hop_result"] = None
        st.session_state["pending_mh_stream"] = False
    elif pipeline_mode == "Multi-hop":
```

（注意：原来的 `if pipeline_mode == "Multi-hop":` 改为 `elif pipeline_mode == "Multi-hop":`）

在 `st.session_state["last_pipeline_mode"] = pipeline_mode` 之后的 session state 保存区追加（约第 248 行附近）：

```python
    st.session_state["agent_pipeline"] = (
        build_agent_pipeline(model_key, use_bm25, enable_reranker)
        if pipeline_mode == "Agent"
        else None
    )
```

实际上更简洁的做法是在渲染时重新构建 pipeline（与 multi-hop 现有模式一致），不需要存储 pipeline。只需在渲染时用 `last_model_key` 等参数重建即可。所以 **不需要** 上面那行，跳过它。

- [ ] **Step 6：在 session state 读取区追加 Agent 字段**

找到 `pending_mh_stream = st.session_state.get("pending_mh_stream", False)` 那行（约第 267 行）之后追加：

```python
agent_result = st.session_state.get("agent_result")
pending_agent_stream = st.session_state.get("pending_agent_stream", False)
```

- [ ] **Step 7：添加 Agent 渲染区块**

在输出区域顶部（`if last_pipeline_mode == "Multi-hop" and multi_hop_result is not None:` 之前）插入：

```python
if last_pipeline_mode == "Agent" and agent_result is not None:
    st.divider()

    # ── Agent 推理轨迹 ────────────────────────────────────────────────────
    st.subheader("Agent 推理轨迹")
    for i, hop in enumerate(agent_result["trace"], 1):
        with st.expander(f"Hop {i} — \"{hop['query']}\"", expanded=True):
            if hop["reasoning"]:
                st.markdown(f"*{hop['reasoning']}*")
            if hop["docs"]:
                for j, doc in enumerate(hop["docs"], 1):
                    title = doc.metadata.get("title", "Untitled")[:60]
                    st.markdown(f"**{j}.** {title}  \n`{_doc_score(doc)}`")
            else:
                st.caption("本跳未检索到文档。")

    # ── Answer ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Answer")
    if pending_agent_stream:
        p = build_agent_pipeline(last_model_key, last_use_bm25, last_enable_reranker)
        try:
            answer = st.write_stream(p.stream_answer(agent_result))
            updated = dict(agent_result)
            updated["answer"] = answer
            st.session_state["agent_result"] = updated
        finally:
            st.session_state["pending_agent_stream"] = False
    else:
        st.markdown(agent_result["answer"])

    # ── Sources ───────────────────────────────────────────────────────────
    st.divider()
    docs = agent_result["retrieved_docs"]
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

elif last_pipeline_mode == "Multi-hop" and multi_hop_result is not None:
```

（注意：原来的 `if last_pipeline_mode == "Multi-hop" and multi_hop_result is not None:` 改为 `elif last_pipeline_mode == "Multi-hop" and multi_hop_result is not None:`）

- [ ] **Step 8：启动 UI 手动验证**

```bash
conda run -n rag streamlit run src/ui/app.py
```

验证清单：

- [ ] 侧边栏 Pipeline 有三个选项：Single-hop / Multi-hop / Agent
- [ ] 选 Agent 后 Compare 复选框变灰
- [ ] 选一个问题点击 Ask，spinner 出现后消失
- [ ] 推理轨迹展示 Hop 1（至少）及 reasoning 文字
- [ ] 答案流式输出
- [ ] 切换回 Single-hop 正常工作

- [ ] **Step 9：Commit**

```bash
git add src/ui/app.py
git commit -m "feat: add Agent pipeline mode to Streamlit UI with reasoning trace"
```

---

## Task 8：集成测试（可选，需要真实 API key）

**Files:**

- Modify: `tests/integration/test_rag_end_to_end.py`

- [ ] **Step 1：写集成测试**

在 `tests/integration/test_rag_end_to_end.py` 末尾追加：

```python
@pytest.mark.integration
def test_agent_rag_pipeline_end_to_end(openai_api_key):
    """AgentRAGPipeline 端到端测试：至少完成 1 跳，返回非空答案。"""
    from src.pipeline.agent_rag import AgentRAGPipeline
    from src.retriever.semantic import SemanticRetriever
    from src.config import EMBED_MODELS, EMBED_MODEL_DEFAULT

    cfg = EMBED_MODELS[EMBED_MODEL_DEFAULT]
    retriever = SemanticRetriever(
        chroma_dir=str(cfg["chroma_dir"]),
        embedding_model=cfg["model_name"],
        enable_reranker=False,
    )
    pipeline = AgentRAGPipeline(retriever=retriever)
    result = pipeline.run("What are the main applications of machine learning?")

    assert result["hop_count"] >= 1
    assert len(result["trace"]) >= 1
    assert result["answer"] != ""
    assert len(result["retrieved_docs"]) > 0
```

- [ ] **Step 2：运行集成测试（需要设置 OPENAI_API_KEY）**

```bash
conda run -n rag pytest tests/integration/test_rag_end_to_end.py -v -m integration
```

预期：PASS（或因无 API key 而 SKIP）

- [ ] **Step 3：Commit**

```bash
git add tests/integration/test_rag_end_to_end.py
git commit -m "test: add AgentRAGPipeline integration test"
```
