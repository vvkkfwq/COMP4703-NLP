# COMP4703 Natural Language Processing - Assignment 2

## 项目需求文档 (Project Requirements Document)

### 📋 基本信息

- **课程**: COMP4703 Natural Language Processing
- **作业**: Project Assignment 2
- **学期**: Semester 2, 2025
- **分值**: 25 marks (占总成绩 25%)
- **截止日期**: Friday 24 October 2025, 3:00PM (Week 12)
- **提交方式**: Blackboard

---

## 🎯 项目目标 (Objective)

实验研究 Retrieval Augmented Generation Systems (RAG) 在解决 "Multi-Hop Retrieval" 问题时的性能权衡。

### 核心任务

Multi-Hop Retrieval 是一个事实性问答问题,需要:

1. 检索多个相关文档
2. 使用 Large Language Model (LLM) 综合信息生成答案
3. 系统返回简洁答案或 "Insufficient Data"

### RAG 系统架构

```
Stage 1: First Stage Retrieval (检索 top-k 文档)
   ↓
Stage 2: [Optional] Reranking (重排序优化)
   ↓
Stage 3: LLM Generation (生成最终答案)
```

---

## 📦 提供的文件清单

### 数据文件

1. **data/corpus.json** - 完整语料库
2. **data/sample-corpus.json** - 样本语料库(用于测试)
3. **data/rag.json** - 完整查询集和金标准
4. **data/sample-rag.json** - 5 个查询样本(用于测试)

### 示例代码

5. **example_retrieval.py** - Dense retrieval 示例
6. **example_reranker.py** - Reranking 示例
7. **example_rag.py** - RAG 完整示例(使用 Llama 2 7B)
8. **evaluate_stage_1.py** - Top-k 检索评估脚本
9. **evaluate_rag.py** - RAG 评估脚本(Recall, Precision, F1)

### 文档

10. **A2Spec-v2.0.pdf** - 作业规格说明
11. **README.md** - 环境配置说明
12. **llamaindex-tutorial.md** - Llama Index API 教程

---

## ⚙️ 技术要求

### 必须完成的实验组合

- **4 个 Stage 1 检索算法**
- **4 个 LLM 模型**
- **实验策略**: 4 个检索算法全部评估,选择 2 个最优的分别与 4 个 LLM 组合

### 时间预算

- Dense retrieval: ~1 小时/模型
- Reranking: ~1.5 小时
- LLM generation: ~2 小时/模型
- **总计算时间**: 约 24 小时
- **GPU 配额**: 40 小时
- **预留时间**: 16 小时(用于调试和测试)

### 运行时间估算

```
4个检索模型评估: 4 × 1小时 = 4小时
2个检索 × 4个LLM: 2 × 2小时 × 4 = 16小时
Reranking(可选): 2 × 1.5小时 = 3小时
-----------------------------------
总计: ~24小时
```

---

## 🖥️ GPU 使用要求

### 访问地址

<https://coursemgr.uqcloud.net/comp4703>

### 重要注意事项

1. ⚠️ **禁止使用 Jupyter Notebook** - 运行时间过长,会导致超时
2. ✅ **推荐使用 Terminal + tmux** - 支持断线后继续运行
3. 🔍 **监控 GPU 使用**: 使用 `nvidia-smi` 检查 GPU 状态
4. 💰 **成本意识**: GPU 时间宝贵,每分钟都有实际成本

### 环境管理

- 使用提供的预配置环境
- 可以安装新库: `pip install <package>`
- 生成依赖列表: `pip freeze > requirements.txt`

### 文件下载

```bash
wget http://wight.seg.rmit.edu.au/jsc/A2-v2.0.zip
```

---

## 📊 评估指标

### Stage 1 Retrieval 评估

- Top-k Recall
- Precision
- Mean Reciprocal Rank (MRR)
- 其他自定义指标(需自行实现)

### RAG 系统评估

- Recall
- Precision
- F1 Score
- 至少 2 个额外的自定义评估指标

---

## 📝 提交要求

### 报告要求 (20 marks)

5 页实验报告,包含:

1. **Introduction** (1 mark) - 问题背景和研究目标
2. **Methodology** (2 marks) - 详细解释使用的方法
3. **Experiments** (3 marks) - 实验设计和执行
4. **Conclusions** (1 mark) - 结论和发现
5. **References** (1 mark) - 参考文献

### 报告质量评分

- Clarity of Writing (2 marks) - 写作清晰度
- Sufficient algorithms (3 marks) - 算法数量充足性
- Evaluation metrics (1 mark) - 评估指标完整性
- Proper visuals (2 marks) - 图表使用(至少 3 个)
- Analysis (3 marks) - 对比分析的深度
- Format (1 mark) - 格式规范

### 环境提交 (5 marks)

- requirements.txt (1 mark)
- 所有代码文件 (2 marks)
- 所有中间结果 (2 marks)

### 必须提交的文件

```
S<学号>/
├── report.pdf                    # 5页实验报告
├── MyREADME.md                  # 运行说明
├── requirements.txt              # 依赖列表
├── ranker[A-D].py               # 4个检索算法
├── reranker[A-B].py             # 重排序算法(可选)
├── rag[A-D].py                  # 4个LLM生成模型
├── MyRetEval.py                 # 自定义检索评估(可选)
├── MyRAGEval.py                 # 自定义RAG评估(可选)
└── output/
    ├── ranker[A-D].json         # 检索结果
    ├── reranker[A-B].json       # 重排序结果
    └── rag[A-D].json            # RAG结果
```

---

## ⚠️ 重要限制和警告

### 学术诚信

- ❌ **禁止使用 AI 写报告或 Python 代码**
- ✅ **允许使用 AI 写 Shell 脚本**
- ⚠️ **严格禁止抄袭** - 使用反抄袭软件检查
- 📌 **个人项目** - 不允许小组合作

### 时间管理建议

1. 早期开始,逐步完成各个模型配置
2. 使用 tmux 运行长时间任务,可以断线后继续
3. 分批运行实验,避免一次性占用所有配额
4. 预留充足时间整理结果和撰写报告

### 扩展申请政策

- 必须通过 UQ 官方渠道申请
- 教师不直接批准扩展
- 需要提供正当理由(疾病、个人悲剧等)

---

## 🆘 获取帮助

### 办公时间

- **时间**: 每周二下午 4 点
- **实验室**: 周三实验课

### 其他渠道

- Lab Demonstrator
- 直接发邮件给教师
- Ed 论坛提问

---

## 📚 参考资源

### 技术文档

- Llama Index API: <https://nanonets.com/blog/llamaindex/>
- Anthropic Documentation: <https://docs.claude.com>

### 支持资源

- Claude Support: <https://support.claude.com>
- UQ Policy on Plagiarism: <https://policies.uq.edu.au/document/view-current.php?id=149>

---

**最后更新**: 2025-10-05
**文档版本**: v1.0
