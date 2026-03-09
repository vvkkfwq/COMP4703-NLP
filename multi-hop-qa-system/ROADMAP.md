# ROADMAP · Multi-Hop RAG 问答评估系统

## 项目概述

基于 HotpotQA 风格数据集的 RAG 问答评估系统。语料库和测试集固定，支持预设问题（含 ground truth 评分）和自由问答两种模式。

**技术栈**：LangChain · BGE-Large · ChromaDB · BM25 · OpenAI API · Streamlit

---

## Milestone 1 · 数据层

corpus.json 入库，建立检索索引。

- [x] 解析 corpus.json，按 `body` 字段分块，其余都在 metadata 保留
- [x] BGE-Large embedding 入库 ChromaDB
- [x] BM25 索引同步构建

---

## Milestone 2 · 检索与生成

完整 RAG pipeline 跑通。

- [x] 语义检索（BGE-Large + ChromaDB）
- [x] Cross-encoder Reranker
- [x] BM25 + 语义混合检索（RRF 融合）
- [x] LangChain 串联检索 → Prompt → OpenAI API 生成

---

## Milestone 3 · 问答界面

系统可交互使用。

- [ ] 预设问题模式：从 sample-rag.json 下拉选题
- [ ] 自由输入模式：用户自己输入任意问题
- [ ] 展示回答 + 检索原文片段（来源、相似度、原文）
- [ ] 预设问题额外展示：ground truth 答案 + 命中文章对比 + F1 评分

---

## Milestone 4 · 配置切换

在基础问答之上支持不同检索配置对比。

- [ ] 侧边栏切换检索模式（语义 / 混合）
- [ ] 侧边栏开关 Reranker
- [ ] 同一问题不同配置的结果并排展示

---

## Milestone 5 · 工程保障

- [ ] 请求缓存（diskcache，相同参数不重复调用 API）
- [ ] 结构化日志（queries.jsonl，记录每次查询的 token 用量和耗时）
- [ ] 成本统计面板（累计费用 / cache 命中率）
- [ ] Docker 一键部署

---

## Milestone 6 · 交付

- [ ] README（项目简介 + 架构图 + Quick Start + 评估结果）
- [ ] 更新简历条目
