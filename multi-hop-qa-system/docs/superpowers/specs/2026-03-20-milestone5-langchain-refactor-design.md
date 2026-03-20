# Milestone 5 · LangChain 规范化改造 设计文档

**日期**: 2026-03-20
**范围**: `src/retriever/semantic.py`, `src/retriever/hybrid.py`, `src/pipeline/rag.py`
**不涉及**: `src/ui/app.py`（UI 适配留后续）、`src/retriever/bm25_retriever.py`

---

## 背景与目标

当前 retriever 完全绕开 LangChain 生态，pipeline 仅用了 prompt 格式化。本次改造将：

1. `SemanticRetriever` / `HybridRetriever` 继承 `BaseRetriever`，接入标准 LangChain 接口
2. `rag.py` 用 LCEL pipe 语法构造完整 chain，包括 retriever 在内
3. 新增 `stream()` 方法，支持逐 token 流式输出

---

## 第一节：Retriever 层

### 改动文件
- `src/retriever/semantic.py`
- `src/retriever/hybrid.py`

### 设计

两个类继承 `langchain_core.retrievers.BaseRetriever`。由于 `BaseRetriever` 是 Pydantic v2 model，原有的实例变量需要声明为 Pydantic 字段。

**做法**：用 `model_post_init()` 替代 `__init__`，显式声明字段类型。需加 `model_config = ConfigDict(arbitrary_types_allowed=True)` 允许非原生 Pydantic 类型（如 `Chroma`、`CrossEncoderReranker`）。

实现 `_get_relevant_documents(query: str) -> list[Document]`，内部调用现有 `retrieve()` 逻辑，保留 `retrieve()` 方法不删（`HybridRetriever` 内部仍调用 `SemanticRetriever.retrieve()`）。

```python
# semantic.py 示意
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, model_validator

class SemanticRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pydantic 字段（构造参数）
    chroma_dir: str = ...
    embedding_model: str = ...
    top_k: int = TOP_K
    enable_reranker: bool = False

    # 运行时对象（model_post_init 中初始化）
    vectorstore: Any = None
    reranker: Any = None

    def model_post_init(self, __context) -> None:
        # 初始化 HuggingFaceEmbeddings、Chroma、CrossEncoderReranker
        ...

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.retrieve(query)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        # 现有逻辑不变
        ...
```

`HybridRetriever` 同理，Pydantic 字段包括 `semantic: SemanticRetriever`、`bm25: BM25Retriever` 等，`model_post_init` 中初始化 `reranker`。

### 不改动
- `BM25Retriever`：仅被 `HybridRetriever` 内部消费，不插入 LCEL chain，不需要继承 `BaseRetriever`

---

## 第二节：RAG Pipeline 层

### 改动文件
- `src/pipeline/rag.py`

### LCEL Chain 结构

用 `RunnablePassthrough.assign()` 构造完整 chain，单次 `invoke` 同时产出 `docs`、`context`、`answer`：

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

chain = (
    RunnablePassthrough.assign(
        docs=retriever                                      # retriever.invoke(question) → list[Document]
    )
    | RunnablePassthrough.assign(
        context=lambda x: format_docs(x["docs"])           # list[Document] → str
    )
    | RunnablePassthrough.assign(
        answer=(
            {"question": itemgetter("question"), "context": itemgetter("context")}
            | PROMPT_TEMPLATE
            | llm
            | StrOutputParser()
        )
    )
)
```

`chain.invoke({"question": query})` 返回：
```python
{
    "question": str,
    "docs": list[Document],
    "context": str,
    "answer": str,
}
```

### `run()` 方法

```python
def run(self, query: str) -> dict:
    return self.chain.invoke({"question": query})
```

返回 dict 包含 `answer`、`docs`、`context`、`question`，调用方按需取用。原有 `sources` 字段不再由 pipeline 生成（调用方自行从 `docs` 提取 metadata）。

---

## 第三节：流式输出

### `stream()` 方法

`RunnablePassthrough.assign()` chain 的 stream 语义是每个步骤完成时推一次完整 dict，不是逐 token。为实现真正的逐 token 流式，`stream()` 将 chain 拆成两段：

```python
def stream(self, query: str):
    # 前半段同步：retriever + format_docs
    docs = self.retriever.invoke(query)
    context = format_docs(docs)
    # 后半段流式：逐 token yield
    yield from (PROMPT_TEMPLATE | self.llm | StrOutputParser()).stream(
        {"context": context, "question": query}
    )
```

返回 generator，yield str token，供 `st.write_stream()` 消费。

`run()` 和 `stream()` 并存，分别服务同步（结构化输出）和流式（逐 token）两个场景。

---

## 变更范围汇总

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/retriever/semantic.py` | 改造 | 继承 BaseRetriever，Pydantic 字段化 |
| `src/retriever/hybrid.py` | 改造 | 继承 BaseRetriever，Pydantic 字段化 |
| `src/retriever/bm25_retriever.py` | 不动 | 内部组件，无需接入 LCEL |
| `src/pipeline/rag.py` | 改造 | LCEL chain + stream() |
| `src/ui/app.py` | 不动 | 留后续 milestone 适配 |

---

## 不在本次范围内

- UI 流式展示（`st.write_stream()`）
- Compare mode 流式输出
- Milestone 6 多跳逻辑
