# Milestone 6 · 多跳查询分解 — 设计规格说明

**日期：** 2026-03-23
**范围：** 实现 Query Decomposition 多跳 pipeline，并适配 Streamlit UI。
**不在范围内：** 单跳 vs 多跳的定量指标对比，迭代检索（M7）。

---

## 目标

- 实现 `MultiHopRAGPipeline`：将复杂问题分解为 2–3 个子问题，各子问题独立检索，合并去重 context，流式输出最终答案。
- 侧边栏新增 **Pipeline 模式** 切换（Single-hop / Multi-hop）。
- 多跳模式下，在流式答案前展示子问题分解过程和各跳检索来源。

---

## 架构

### 新增文件：`src/pipeline/multi_hop_rag.py`

与 `RAGPipeline` 完全独立，不继承它。

**数据类：**

```python
@dataclass          # 不使用 frozen=True — UI 在流式输出后需要回写 answer 字段
class HopResult:
    sub_question: str
    docs: list[Document]

@dataclass          # 不使用 frozen=True — UI 在流式输出后需要回写 answer 字段
class MultiHopResult:
    question: str
    sub_questions: list[str]
    hop_results: list[HopResult]   # 每跳的子问题 + 检索结果
    merged_docs: list[Document]    # 所有跳合并去重后的文档
    answer: str                    # retrieve() 后为空字符串；UI 完成 stream() 后回写
```

**公共接口：**

```python
class MultiHopRAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL): ...

    def retrieve(self, query: str) -> MultiHopResult:
        """分解问题 → 各子问题独立检索 → 合并文档。
        不调用 LLM 生成答案。返回 answer="" 的 MultiHopResult。
        在 UI 的 spinner 块中调用此方法。"""

    def stream(self, result: MultiHopResult) -> Iterator[str]:
        """接收已填充的 MultiHopResult，流式输出最终答案。
        prompt 由 result.question、result.sub_questions 和 result.merged_docs 构建。
        在 retrieve() 之后调用，驱动 st.write_stream()。"""
```

**接口拆分原因：** 与 `app.py` 中现有的单跳模式完全对称 —— `p.retriever.invoke(query)` 在 spinner 中运行，然后 `p.stream(query, docs=docs)` 用于 `st.write_stream()`。多跳模式下，`pipeline.retrieve(query)` 在 spinner 中运行，然后 `pipeline.stream(result)` 驱动 `st.write_stream()`。

**内部函数（模块级，纯函数或接近纯函数）：**

| 函数               | 签名                                                 | 说明                                                                                                                                     |
| ------------------ | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `decompose_query`  | `(query: str, llm) -> list[str]`                     | LLM 调用，返回 JSON 数组；解析失败时降级为 `[query]`                                                                                     |
| `merge_docs`       | `(hop_docs: list[list[Document]]) -> list[Document]` | 按 `metadata["url"]` 再按 `metadata["title"]` 去重；保留分数最高的副本                                                                   |
| `_doc_score_value` | `(doc: Document) -> float`                           | 返回数值分数：`_rerank_score` → `_rrf_score` → `_score` → 0.0。仅内部使用，不导出；`app.py` 中有独立的 `_doc_score()` 返回格式化字符串。 |

**分解 LLM temperature：** `decompose_query` 使用独立的 `ChatOpenAI` 实例，`temperature=0.0`（结构化提取任务），不复用生成用的 `llm`（`temperature=0.1`）。两者使用相同的 `model` 名称（来自 `__init__` 参数）。

**`stream()` prompt 约定：**
生成 prompt 中包含子问题，为 LLM 提供显式推理上下文：

```
System: You are a helpful assistant. Answer using ONLY the provided context.
Human:
  Sub-questions used for retrieval:
  1. <sub_q1>
  2. <sub_q2>

  Context:
  <merged_docs 格式化内容>

  Question: <原始问题>
  Answer:
```

**分解 prompt 约定：**
返回字符串 JSON 数组，例如 `["sub_q1", "sub_q2"]`。若 LLM 输出无法解析为列表，`decompose_query` 记录警告并返回 `[query]`（降级为单跳）。prompt 中限制最多 3 个子问题。

**合并去重优先级：**

1. `metadata["url"]`（优先，全局唯一）
2. `metadata["title"]`（URL 缺失时的备选）

同一 key 的重复文档，保留分数较高者（`_rerank_score` → `_rrf_score` → `_score` → 0）。结果按分数降序排列。

---

### 修改文件：`src/ui/app.py`

#### 侧边栏新增

在 **Settings** 区块中，Mode radio 下方新增 Pipeline radio：

```
Settings
  Mode:        ○ Preset questions  ○ Free-form
  Pipeline:    ○ Single-hop  ● Multi-hop       ← 新增
─────────────────────────────────────────────
Configuration
  Embedding Model  [下拉框]
  BM25 + RRF       [开关]
  Reranker         [开关]
  Compare all 4    [复选框 — Pipeline = Multi-hop 时禁用]
```

`Compare all 4 strategies` 在多跳模式下禁用（`st.checkbox(..., disabled=pipeline_mode=="Multi-hop")`），因为对比模式仅适用于单跳。

#### Session state 新增键

| Key                    | 类型                     | 用途                                                            |
| ---------------------- | ------------------------ | --------------------------------------------------------------- |
| `"multi_hop_result"`   | `MultiHopResult \| None` | 存储分解结果 + 各跳文档，跨 rerun 保持                          |
| `"pending_mh_stream"`  | `bool`                   | 触发下一次 rerun 的流式答案渲染                                 |
| `"last_pipeline_mode"` | `str`                    | `"Single-hop"` 或 `"Multi-hop"` — 点击时捕获，渲染 rerun 时读取 |

以下读取语句须添加到现有读取块中（位于 `if ask_clicked` 树之后，与 `last_model_key`、`last_use_bm25` 等类似）：

```python
last_pipeline_mode = st.session_state.get("last_pipeline_mode", "Single-hop")  # 新增
multi_hop_result = st.session_state.get("multi_hop_result")                     # 新增
pending_mh_stream = st.session_state.get("pending_mh_stream", False)            # 新增
```

#### 过期状态清除（更新）

切换问题但未点击 Ask 时触发清除，需扩展以包含新的多跳 key：

```python
if previous_query_key != current_query_key and not ask_clicked:
    st.session_state["result"] = None
    st.session_state["compare_results"] = None
    st.session_state["pending_stream"] = False
    st.session_state["multi_hop_result"] = None      # 新增
    st.session_state["pending_mh_stream"] = False    # 新增
```

#### Ask 按钮执行 — 分支顺序

```python
if ask_clicked and query and query.strip():
    if pipeline_mode == "Multi-hop":
        # 多跳分支（优先判断；此模式下 compare_mode 已禁用）
        with st.spinner("Decomposing question and retrieving…"):
            p = build_multi_hop_pipeline(model_key, use_bm25, enable_reranker)
            mh_result = p.retrieve(query)
        st.session_state["multi_hop_result"] = mh_result
        st.session_state["pending_mh_stream"] = True
        st.session_state["result"] = None
        st.session_state["compare_results"] = None
        st.session_state["pending_stream"] = False
    elif compare_mode:
        # 现有对比分支 — 不变
        ...
    else:
        # 现有单跳分支 — 不变
        ...
    # 公共：存储 last_* 键
    st.session_state["last_pipeline_mode"] = pipeline_mode
    st.session_state["last_query"] = query
    st.session_state["last_mode"] = mode
    st.session_state["last_qa"] = selected_qa
    st.session_state["last_model_key"] = model_key
    st.session_state["last_use_bm25"] = use_bm25
    st.session_state["last_enable_reranker"] = enable_reranker
```

`build_multi_hop_pipeline(model_key, use_bm25, enable_reranker)` 是类比 `build_pipeline()` 的新辅助函数，使用相同的 `_build_retriever()` 逻辑构建 `MultiHopRAGPipeline`。**不**加 `@st.cache_resource` 装饰，与 `build_pipeline` 相同——`_build_retriever()` 内部已通过 `@st.cache_resource` 缓存 retriever 和 vectorstore，pipeline wrapper 每次重新实例化。

#### 多跳输出渲染

当 `last_pipeline_mode == "Multi-hop"` 且 `multi_hop_result is not None` 时触发。

```
── 子问题分解 ──────────────────────────────
  [1] <sub_question_1>
      st.expander → 来源列表（标题 + _doc_score()）
  [2] <sub_question_2>
      st.expander → 来源列表（标题 + _doc_score()）

── Answer ──────────────────────────────────
  if pending_mh_stream:
      p = build_multi_hop_pipeline(last_model_key, last_use_bm25, last_enable_reranker)
      try:
          answer = st.write_stream(p.stream(mh_result))
          mh_result.answer = answer          # 回写到已存储的结果中
          session_state["multi_hop_result"] = mh_result
      finally:
          session_state["pending_mh_stream"] = False   # 无论成功与否都清除，防止出错后重复流式
  else:
      st.markdown(mh_result.answer)

── Retrieved Sources (N) ───────────────────
  docs = mh_result.merged_docs
  [复用现有的逐文档 expander 渲染逻辑，不做修改]

── Evaluation ──────────────────────────────
  answer_str = mh_result.answer
  docs = mh_result.merged_docs               # 注意：使用 merged_docs，不是某一跳的 docs
  f1 = token_f1(answer_str, ground_truth)    # 与单跳相同的调用
  [复用现有的 evidence hit-check 逻辑，读取上方的 docs]
```

Sources 区块展示 `merged_docs`（去重合并后的联合集）。各跳的独立文档仅在子问题分解的 expander 内可见。Evaluation 区块与单跳结构完全相同，仅变量名不同（`mh_result.answer` vs `result["answer"]`）。

---

## 错误处理

| 失败点                                                        | 行为                                                       |
| ------------------------------------------------------------- | ---------------------------------------------------------- |
| `decompose_query` JSON 解析失败                               | 记录警告；降级为 `[query]`（等效单跳）                     |
| 某子问题检索结果为空                                          | `HopResult.docs = []`；从合并中排除；不崩溃                |
| 所有子问题检索结果均为空                                      | `merged_docs = []`；生成继续执行，LLM 返回 "I don't know." |
| `stream()` 期间 OpenAI API 报错                               | 异常传递到 Streamlit；通过 `st.error()` 展示               |
| `compare_mode=True` + `pipeline_mode="Multi-hop"`（过期状态） | 多跳分支在 `if ask_clicked` 树中优先判断；对比分支不可达   |

---

## 不变的部分

- `src/pipeline/rag.py` — 不修改
- `src/retriever/` — 不修改
- `src/config.py` — 无需新增常量（复用 `LLM_MODEL`）
- 评估指标逻辑 — 不修改；使用 `token_f1(mh_result.answer, ground_truth)`

---

## M7（LangGraph）接缝

`decompose_query`、`merge_docs` 和 `retrieve()` 内的逐跳检索循环均为无状态函数，天然对应 LangGraph 节点（`decompose_node`、`retrieve_node`、`generate_node`）。注意 `MultiHopResult` 是 dataclass，而 M7 的 `RAGState` 将是 `TypedDict`——迁移时需要在两种容器之间映射字段，但 retriever 层和 UI 渲染层预计无需修改。
