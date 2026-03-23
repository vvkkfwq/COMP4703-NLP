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

**做法**：用 `model_post_init()` 替代 `__init__`，显式声明字段类型。`BaseRetriever` 已设置 `arbitrary_types_allowed=True`，子类无需重复添加 `model_config`，除非需要追加其他配置项。

实现 `_get_relevant_documents(query, *, run_manager)`，内部调用现有 `retrieve()` 逻辑，保留 `retrieve()` 方法不删（`HybridRetriever` 内部仍调用 `SemanticRetriever.retrieve()`）。

```python
# semantic.py 示意
from typing import Any
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

class SemanticRetriever(BaseRetriever):
    # BaseRetriever 已包含 arbitrary_types_allowed=True，无需重复声明 model_config

    # Pydantic 字段（构造参数）
    chroma_dir: str = ...        # 注意：调用方传入 Path 时需先 str() 转换
    embedding_model: str = ...
    top_k: int = TOP_K
    enable_reranker: bool = False

    # 运行时对象（model_post_init 中初始化，默认 None）
    vectorstore: Any = None
    reranker: Any = None

    def model_post_init(self, __context: Any) -> None:
        # 初始化 HuggingFaceEmbeddings、Chroma
        # 仅当 enable_reranker=True 时才加载 CrossEncoderReranker（避免无谓的模型下载）
        ...
        super().model_post_init(__context)   # 必须调用，确保 LangChain 回调等内部逻辑执行

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.retrieve(query)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        # 现有逻辑不变
        ...
```

**注意事项：**

- `_get_relevant_documents` 必须包含 `run_manager` 关键字参数（LangChain 内部按名传入），实现中可忽略不用
- `model_post_init` 末尾必须调用 `super().model_post_init(__context)`，否则跳过 LangChain 的回调注册逻辑
- `SemanticRetriever` 当前代码无条件加载 reranker 模型（即使 `enable_reranker=False`）——迁移时在 `model_post_init` 中加 `if self.enable_reranker:` 守卫修复此问题
- `chroma_dir` 字段类型为 `str`，`app.py` 传入 `cfg["chroma_dir"]`（`Path` 类型）时需在调用处做 `str()` 转换

`HybridRetriever` 同理，Pydantic 字段包括 `semantic: SemanticRetriever`、`bm25: BM25Retriever` 等，`model_post_init` 中初始化 `reranker`（同样加 `enable_reranker` 守卫）。`app.py` 传入已构造好的实例，Pydantic v2 接受 model 实例作为字段值，调用约定不变。

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

返回 dict 包含 `answer`、`docs`、`context`、`question`，调用方按需取用。

**`top_k` 参数处理：** 原 `RAGPipeline.__init__` 接收 `top_k` 并在 `run()` 中传给 `self.retriever.retrieve(query, top_k=self.top_k)`。新设计中 chain 通过 `retriever.invoke()` 调用 retriever，无法在调用时动态传入 `top_k`。因此 `top_k` 改为在 retriever 构造时固定，`RAGPipeline.__init__` 的 `top_k` 参数移除。调用方如需不同 `top_k`，构造不同的 retriever 实例即可（现有 `app.py` 已通过 `build_pipeline()` 工厂函数管理 retriever 构造，行为不变）。

**⚠️ Breaking change（与旧接口的差异）：**

| 旧键             | 新键 / 状态 | 影响                                                        |
| ---------------- | ----------- | ----------------------------------------------------------- |
| `retrieved_docs` | 改为 `docs` | `app.py` 多处访问 `result["retrieved_docs"]` 会 `KeyError`  |
| `sources`        | 移除        | `app.py` 不再直接使用，调用方自行从 `docs` 的 metadata 提取 |
| `answer`         | 保留        | 不变                                                        |

`app.py` 的适配（将 `result["retrieved_docs"]` 改为 `result["docs"]`、移除 `sources` 依赖）**推迟到 UI 适配 milestone**，本 milestone 合入后 app.py 会有 KeyError，属已知可接受的中间状态。

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

**注意**：`stream()` 只 yield 文本 token，不返回 `docs`。UI 接入流式时需单独调用 `retriever.invoke()` 获取来源文档——这是 UI 适配 milestone 需要处理的问题。

`run()` 和 `stream()` 并存，分别服务同步（结构化输出）和流式（逐 token）两个场景。

---

## 变更范围汇总

| 文件                              | 变更类型 | 说明                                |
| --------------------------------- | -------- | ----------------------------------- |
| `src/retriever/semantic.py`       | 改造     | 继承 BaseRetriever，Pydantic 字段化 |
| `src/retriever/hybrid.py`         | 改造     | 继承 BaseRetriever，Pydantic 字段化 |
| `src/retriever/bm25_retriever.py` | 不动     | 内部组件，无需接入 LCEL             |
| `src/pipeline/rag.py`             | 改造     | LCEL chain + stream()               |
| `src/ui/app.py`                   | 不动     | 留后续 milestone 适配               |

---

## 不在本次范围内

- UI 流式展示（`st.write_stream()`）及 `result["docs"]` key 适配
- Compare mode 流式输出
- Milestone 6 多跳逻辑

## 已知风险

- **`@st.cache_resource` 与 Pydantic model**：`SemanticRetriever` / `HybridRetriever` 变为 Pydantic model 后，Streamlit 按字段值 hash，可能出现意外 cache miss，但不影响正确性，留观察。
