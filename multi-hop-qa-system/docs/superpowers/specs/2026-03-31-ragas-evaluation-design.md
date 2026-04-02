# Milestone 8 · RAGAS 评估接入设计文档

**日期**：2026-03-31
**状态**：已批准

---

## 背景与目标

现有评估体系包含 token-F1（答案质量）和 NDCG/MAP/MRR/Hits（检索质量），缺少 LLM-based 的端到端评估维度。Milestone 8 引入 RAGAS，补充以下四个指标：

| 指标              | 含义                               | 依赖 ground truth | 依赖 LLM |
| ----------------- | ---------------------------------- | :---------------: | :------: |
| faithfulness      | 答案是否完全基于检索上下文，不捏造 |        ❌         |    ✅    |
| answer_relevancy  | 答案是否切题                       |        ❌         |    ✅    |
| context_precision | 相关上下文是否排在前面             |        ✅         |    ✅    |
| context_recall    | 所需信息是否被检索回来             |        ✅         |    ✅    |

---

## 范围决策

| 决策点           | 结论                                | 理由                                                         |
| ---------------- | ----------------------------------- | ------------------------------------------------------------ |
| 触发方式         | UI 按需单条评估                     | MVP 优先，费用可控                                           |
| 生效模式         | Preset questions 模式               | free-form 无 ground truth，context_precision/recall 无法计算 |
| Pipeline 支持    | Single-hop、Multi-hop、Agent 均支持 | 三种模式输出结构一致（answer + retrieved_docs）              |
| UI 交互          | Spinner 阻塞等待                    | 与现有 pipeline 调用风格一致，实现最简                       |
| build_metrics.py | 不改动                              | 批量 RAGAS 费用高，留作后续 Milestone                        |

---

## 架构

### 新增：`src/evaluation/ragas_eval.py`

**公开接口**：

```python
def run_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float | None]:
    ...
```

**行为**：

- 针对 `ragas>=0.2`（新版 API）：使用 `EvaluationDataset` + `evaluate(dataset, metrics=[...])`；实现时先 `import ragas; assert ragas.__version__ >= "0.2"` 确认版本，并在 `requirements.txt` / 注释中锁定 `ragas>=0.2,<0.3`
- 内部组装单条数据集，调用 `evaluate(dataset, metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()])`
- 返回 `{"faithfulness": float, "answer_relevancy": float, "context_precision": float, "context_recall": float}`
- 若 RAGAS 未安装、API Key 缺失、或调用抛出异常，所有值返回 `None`，打印警告，不 raise（避免崩溃 UI）

**入参来源**（由 UI 负责组装）：

| 参数           | 来源                                                                      |
| -------------- | ------------------------------------------------------------------------- |
| `question`     | `last_query`                                                              |
| `answer`       | `result["answer"]` / `multi_hop_result.answer` / `agent_result["answer"]` |
| `contexts`     | `[doc.page_content for doc in retrieved_docs]`                            |
| `ground_truth` | `last_qa["answer"]`                                                       |

### 改动：`src/ui/app.py`

在三个 pipeline 分支的 Preset 评估区块（`if last_mode == "Preset questions"`）中统一加入：

1. **"Run RAGAS Evaluation" 按钮**：仅在 `answer` 已生成时显示（`pending_stream == False` 等）
2. **点击逻辑**：
   ```python
   with st.spinner("Running RAGAS evaluation…"):
       result = run_ragas(question, answer, contexts, ground_truth)
   st.session_state["ragas_result"] = result
   ```
3. **结果展示**：4 个 `st.metric`，与 Token-F1 同行展示：
   ```
   Token-F1 | faithfulness | answer_relevancy | context_precision | context_recall
   ```
4. **状态清理**：切换问题时 `ragas_result` 一并清除（加入现有的 session_state 清除逻辑）

---

## 数据流

```
用户点击 "Run RAGAS"
  │
  ▼
组装入参
  question  ← last_query
  answer    ← session_state 中对应 pipeline 的 answer
  contexts  ← retrieved_docs[*].page_content
  ground_truth ← last_qa["answer"]
  │
  ▼
ragas_eval.run_ragas()
  │  内部：构建 ragas.Dataset → evaluate()
  │
  ▼
dict[str, float | None]
  │
  ▼
st.session_state["ragas_result"]
  │
  ▼
st.metric × 4 渲染
```

---

## 错误处理

- RAGAS 库未安装：捕获 `ImportError`，返回全 `None`，UI 显示 "RAGAS not installed"
- OpenAI 调用失败（网络/Key问题）：捕获通用 `Exception`，返回全 `None`，UI 显示错误信息
- 部分指标失败：返回该指标 `None`，其他正常显示

---

## 不在范围内

- `build_metrics.py` 批量 RAGAS（后续 Milestone）
- free-form 模式下的 RAGAS 评估
- RAGAS 结果持久化到文件
- 自定义 evaluator LLM（使用 RAGAS 默认，即 OpenAI）
