# Milestone 7 · LangGraph Agentic RAG — 设计规格

**日期：** 2026-03-26
**状态：** 已批准

---

## 概述

将多跳检索逻辑升级为基于 LangGraph `StateGraph` 的真正 Agentic RAG。Agent 从原始问题出发，先检索一次，然后用 LLM 判断当前累积的 context 是否足够回答问题。若不足，则生成一个有针对性的补充检索 query 并再次检索，最多循环 3 跳。这与 M6（静态问题分解）的本质区别在于：每一跳都由真实的信息缺口驱动，而非预先拆分。

---

## 数据结构

### `HopTrace`（TypedDict）

```python
class HopTrace(TypedDict):
    query: str            # 本跳使用的检索 query
    docs: list[Document]  # 本跳检索到的文档
    reasoning: str        # judge_node 的判断理由
```

### `RAGState`（TypedDict）

```python
class RAGState(TypedDict):
    question: str
    current_query: str          # 下一次 retrieve_node 使用的 query；初始值等于 question
    retrieved_docs: list[Document]  # 所有跳累积、去重后的文档
    hop_count: int
    sufficient: bool            # judge_node 的判断结果
    follow_up_query: str        # judge_node 生成的下一跳 query（信息充足时为空字符串）
    trace: list[HopTrace]       # 每跳完成后追加一条记录
    answer: str                 # generate_node 填入最终答案
```

---

## 图拓扑

```
START → retrieve_node → judge_node ──（信息充足 或 hop_count ≥ 3）──→ generate_node → END
                            │
                  （信息不足 且 hop_count < 3）
                            ↓
                       retrieve_node（current_query = follow_up_query）
```

条件边函数：

```python
def should_continue(state: RAGState) -> str:
    if state["sufficient"] or state["hop_count"] >= 3:
        return "generate"
    return "retrieve"
```

---

## 节点规格

### `retrieve_node`

- 调用 `retriever.invoke(state["current_query"])`
- 用 `multi_hop_rag.py` 中已有的 `merge_docs()` 将本跳结果合并去重后追加到 `retrieved_docs`
- 向 `trace` 追加一条 `HopTrace`（query、本跳 docs、reasoning 暂填空字符串）
- `hop_count` +1
- 仅当 `follow_up_query` 非空时，将 `current_query` 更新为 `follow_up_query`（首跳不触发此更新）

### `judge_node`

单次 LLM 调用，使用结构化输出 prompt。输入：原始 `question` + 所有 `retrieved_docs` 的 `page_content` 拼接（截断以适应上下文窗口）。期望输出 JSON：

```json
{
  "sufficient": true,
  "follow_up_query": "",
  "reasoning": "Context 已包含 X、Y、Z，足以回答问题。"
}
```

- 解析 JSON；解析失败时降级为 `sufficient=True`（防止死循环）
- 更新 `state["sufficient"]`、`state["follow_up_query"]`
- 将 `reasoning` 回写至 `state["trace"]` 最后一条记录

### `generate_node`

- 用已有的 `_format_docs()` 将 `retrieved_docs` 格式化为 context 字符串
- 调用 `chain.invoke()`（复用 M6 的 `_ANSWER_PROMPT`），将完整答案存入 `state["answer"]`
- 此处为同步调用；流式输出由 `AgentRAGPipeline.stream_answer()` 在 UI 层单独处理

---

## `AgentRAGPipeline` 接口（`src/pipeline/agent_rag.py`）

```python
class AgentRAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL) -> None: ...

    def run(self, question: str) -> RAGState:
        """执行完整图，返回包含 trace 和 answer 的最终 state。"""

    def stream_answer(self, state: RAGState) -> Iterator[str]:
        """基于 retrieved_docs + question 流式生成答案，供 UI 调用。"""
```

内部构建 `StateGraph[RAGState]`，添加节点和条件边，编译为 `app`，通过 `app.invoke(initial_state)` 执行。

---

## Streamlit UI 改动（`src/ui/app.py`）

1. Pipeline radio 新增第三个选项 `"Agent"`（Single-hop / Multi-hop 保持不变）
2. `pipeline_mode == "Agent"` 时禁用 `compare_mode` 复选框（与 Multi-hop 处理方式一致）
3. Agent 模式点击 Ask 后：
   - spinner 文案：`"Running agentic retrieval…"`
   - 调用 `AgentRAGPipeline.run(query)`，结果存入 `st.session_state["agent_result"]`
   - 设置 `st.session_state["pending_agent_stream"] = True`
4. 渲染区域（不区分预设/自由输入模式，均展示）：
   - 区块标题：`"Agent 推理轨迹"`
   - 每跳一个 `st.expander`，默认展开，标题格式：`Hop N — "{query}"`
     - judge reasoning（斜体文本）
     - 检索来源列表（标题 + 分数，样式与现有来源展示保持一致）
   - 轨迹下方：通过 `st.write_stream(pipeline.stream_answer(state))` 流式输出最终答案

---

## 复用与不改动范围

| 组件 | 处理方式 |
|---|---|
| `multi_hop_rag.py` 中的 `merge_docs()` | 直接复用 |
| `multi_hop_rag.py` 中的 `_format_docs()` | 直接复用 |
| `multi_hop_rag.py` 中的 `_ANSWER_PROMPT` | 直接复用 |
| `rag.py` | 不改动 |
| `multi_hop_rag.py` | 不改动 |
| 检索器层 | 不改动 |
| 评估 / metrics | 不在本次范围内 |

---

## 错误处理

- **judge 解析失败**：降级为 `sufficient=True`，正常退出循环
- **某跳检索结果为空**：视为信息充足（无新内容可检索）
- **达到最大跳数**：`hop_count >= 3` 时强制进入 `generate_node`，忽略 judge 结果

---

## 测试

- 单元测试：向各节点传入 mock `RAGState` dict，独立测试 `retrieve_node`、`judge_node`、`generate_node`
- 集成测试：用真实 retriever 对一条样本问题调用 `AgentRAGPipeline.run()`，断言 `hop_count >= 1`、`answer != ""`
- 现有测试无需改动
