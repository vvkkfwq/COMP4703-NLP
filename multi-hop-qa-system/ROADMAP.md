# ROADMAP · Multi-Hop RAG 问答评估系统

## 项目概述

基于 HotpotQA 风格数据集的 Agentic RAG 问答评估系统。语料库和测试集固定，支持单跳 / 多跳 / Agent 三种检索模式，预设问题（含 ground truth 评分）和自由问答两种交互模式。

**技术栈**：LangChain · LangGraph · BGE-Large · ChromaDB · BM25 · OpenAI API · RAGAS · Streamlit

---

## Milestone 1 · 数据层 ✅ 已完成

corpus.json 入库，建立检索索引。

- [x] 解析 corpus.json，按 `body` 字段分块，metadata 保留 `url / title / source`
- [x] BGE-Large embedding 入库 ChromaDB（支持多 embedding 模型独立目录）
- [x] BM25 索引同步构建

---

## Milestone 2 · 检索与生成 ✅ 已完成

完整单跳 RAG pipeline 跑通。

- [x] 语义检索（BGE-Large + ChromaDB）
- [x] Cross-encoder Reranker（bge-reranker-base）
- [x] BM25 + 语义混合检索（RRF 融合）
- [x] LangChain 串联检索 → Prompt → OpenAI API 生成

---

## Milestone 3 · 问答界面 ✅ 已完成

系统可交互使用。

- [x] 预设问题模式：从 sample-rag.json 下拉选题
- [x] 自由输入模式：用户自己输入任意问题
- [x] 展示回答 + 检索原文片段（来源、相似度、原文）
- [x] 预设问题额外展示：ground truth 答案 + 命中文章对比 + F1 评分

---

## Milestone 4 · 配置切换 ✅ 已完成

在基础问答之上支持不同检索配置对比。

- [x] 侧边栏切换检索模式（语义 / 混合）
- [x] 侧边栏开关 Reranker
- [x] 同一问题 4 种策略结果并排展示（Compare mode）
- [x] 侧边栏 Retrieval Metrics 面板（NDCG@10 / MAP@10 / MRR@10 / Hits@10）

---

## Milestone 5 · LangChain 规范化改造 🔧 待做

**背景**：当前 retriever 完全绕开 LangChain 生态，pipeline 仅用了 prompt 格式化，
面试层面无法体现 LangChain 深度。本 milestone 将检索层接入标准接口，统一用 LCEL 串联。

- [x] `SemanticRetriever` / `HybridRetriever` 继承 `BaseRetriever`，实现 `_get_relevant_documents()`
- [x] `rag.py` 改用 LCEL pipe 语法：`retriever | format_docs` → `PROMPT` → `LLM` → `StrOutputParser()`
- [x] 支持流式输出（`chain.stream()`），Streamlit 用 `st.write_stream()` 展示

---

## Milestone 6 · 多跳检索实现 🔧 待做

**背景**：项目名为 Multi-Hop RAG，但当前 pipeline 是单次检索。本 milestone 实现真正的多跳逻辑，
解决名实不符问题，并建立单跳 vs 多跳的对比实验。

### 方案 A：Query Decomposition（先做）

- [x] 新增 `src/pipeline/multi_hop_rag.py`
  - LLM 将复杂问题分解为 2-3 个子问题
  - 每个子问题独立检索，合并去重 context
  - 最终 LLM 综合子问题和 context 生成答案
- [x] Streamlit 侧边栏新增 Pipeline 模式切换（Single-hop / Multi-hop）
- [x] 预设问题模式下展示：子问题分解过程 + 各子问题检索来源

### 方案 B：Iterative Retrieval（在 M7 LangGraph 中实现）

LLM 判断当前 context 是否充足，不够则生成 follow-up query 继续检索，
天然与 Agent 循环结合，在 Milestone 7 一并完成。

---

## Milestone 7 · LangGraph Agentic RAG 🔧 待做

**背景**：将多跳检索的循环逻辑升级为 LangGraph StateGraph，实现真正的 Agentic RAG，
解锁简历 Agent 方向定位。

- [x] 定义 `RAGState`（TypedDict）：`question / sub_questions / retrieved_docs / answer / hop_count`
- [x] 实现四个节点：
  - `retrieve_node`：检索（可被循环调用）
  - `judge_node`：判断信息是否充足（初期用规则，后期换 LLM）
  - `generate_node`：生成最终答案
- [x] `judge_node` → 条件边：`hop_count < 3` 且信息不足 → 回到 `retrieve_node`；否则 → `generate_node`
- [x] 新增 `src/pipeline/agent_rag.py`，封装 LangGraph 调用接口
- [x] Streamlit Pipeline 模式新增 Agent 选项，展示推理轨迹（每跳的 query 和来源）

---

## Milestone 8 · RAGAS 评估接入 🔧 待做

**背景**：补全 LLM-based 评估维度，与现有 token-F1 / NDCG 形成完整评估体系。

- [ ] 新增 `src/evaluation/ragas_eval.py`
  - 封装 `faithfulness` / `answer_relevancy` / `context_precision` / `context_recall`
  - 入参对齐现有数据结构（question / answer / contexts / ground_truth）
- [ ] Streamlit Evaluation 区块新增"Run RAGAS"按钮（异步，避免阻塞界面）
- [ ] 评估结果与 token-F1 并排展示
- [ ] `build_metrics.py` 支持批量跑 RAGAS，结果写入 `data/ragas_metrics.json`

> **注意**：RAGAS 的 `faithfulness` / `answer_relevancy` 依赖 LLM 评判，会产生额外 token 费用，
> 结果随 evaluator 模型版本漂移。`context_precision` / `context_recall` 依赖 ground truth，
> 更稳定，优先作为主要指标展示。

---

## Milestone 9 · 工程保障 🔧 待做

- [ ] 请求缓存（diskcache，key 包含 `question + pipeline_mode + retrieval_config + model`）
- [ ] 结构化日志（queries.jsonl，记录 pipeline 模式 / token 用量 / 耗时 / hop 数）
- [ ] 成本统计面板（累计费用 / cache 命中率，区分单跳 / 多跳 / Agent 成本）
- [ ] Docker 一键部署

---

## Milestone 10 · 交付 🔧 待做

- [ ] README：项目简介 + 架构图（单跳 vs 多跳 vs Agent 三条路径）+ Quick Start + 评估结果对比表
- [ ] 评估结果表：6 种检索策略 × 3 种 pipeline 模式 × F1 / NDCG / RAGAS 指标
- [ ] 更新简历条目（Multi-Hop 实现 + LangGraph Agent + RAGAS 评估）

---

## 执行顺序建议

```
M5（LangChain 规范化）→ M6-A（Query Decomposition）→ M8（RAGAS）→ M7（LangGraph）→ M9 → M10
```

M5 是基础，改完之后 M6 / M7 的代码都能直接复用标准 LCEL 接口。
M8 可以在 M6 完成后立即接入，不依赖 LangGraph。
M7 最复杂，放在 RAGAS 跑通之后做，有完整评估体系支撑实验。
