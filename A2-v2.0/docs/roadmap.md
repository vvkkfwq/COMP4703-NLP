# COMP4703 Assignment 2 - 任务路线图 (Task Roadmap)

## Multi-Hop RAG System 实验项目

**截止日期**: 2025 年 10 月 24 日 15:00
**当前日期**: 2025 年 10 月 5 日
**剩余时间**: 19 天

---

## 📅 整体时间规划

### Week 1: 环境搭建和熟悉阶段 (10 月 5 日-10 月 11 日)

**目标**: 完成环境配置,理解代码框架,完成初步测试

### Week 2: 实验执行阶段 (10 月 12 日-10 月 18 日)

**目标**: 运行所有实验组合,收集结果数据

### Week 3: 分析和撰写阶段 (10 月 19 日-10 月 24 日)

**目标**: 数据分析,撰写报告,准备提交

---

## 🎯 详细任务分解

## Phase 1: 准备阶段 (Day 1-3: 10 月 5 日-10 月 7 日)

### ✅ Day 1: 环境配置和理解 (10 月 5 日)

**预计时间**: 4-6 小时**GPU 时间**: 1 小时

- [x] **Task 1.1**: 访问 GPU 实例

  - 登录 https://coursemgr.uqcloud.net/comp4703
  - 熟悉 Web Terminal 界面
  - 学习使用 tmux (会话管理)

- [x] **Task 1.2**: 下载和解压项目文件

  ```bash
  wget http://wight.seg.rmit.edu.au/jsc/A2-v2.0.zip
  unzip A2-v2.0.zip
  cd A2-v2.0
  ```

- [x] **Task 1.3**: 验证环境

  - 检查 Python 环境
  - 运行 `nvidia-smi` 确认 GPU 可用
  - 查看预装模型: `ls $HOME/.cache/huggingface/hub`

- [x] **Task 1.4**: 理解数据结构

  - 查看 `data/sample-corpus.json` 结构
  - 查看 `data/sample-rag.json` 查询格式
  - 理解金标准答案格式

- [x] **Task 1.5**: 测试示例代码

  ```bash
  # 使用样本数据测试
  python example_retrieval.py
  python evaluate_stage_1.py
  ```

**输出文档**:

- 环境配置笔记
- 数据结构理解文档

---

### ✅ Day 2: 代码理解和修改准备 (10 月 6 日)

**预计时间**: 6-8 小时**GPU 时间**: 2 小时

- [ ] **Task 2.1**: 深入理解示例代码

  - 阅读 `example_retrieval.py`
  - 阅读 `example_reranker.py`
  - 阅读 `example_rag.py`
  - 理解 Llama Index API 调用方式

- [ ] **Task 2.2**: 研究可用模型

  - 检索模型选择 (4 个)
    - 查看已下载的 embedding 模型
    - 研究不同 dense retrieval 算法
  - LLM 模型选择 (4 个)
    - 查看可用的 7B-13B 参数模型
    - 确认模型在 24GB GPU 内存限制下可运行

- [ ] **Task 2.3**: 阅读 Llama Index 文档

  - 阅读 `llamaindex-tutorial.md`
  - 理解如何切换不同的 retrieval 算法
  - 理解如何切换不同的 LLM 模型

- [ ] **Task 2.4**: 完整测试流程

  ```bash
  # 测试完整RAG流程
  python example_rag.py
  python evaluate_rag.py
  ```

**输出文档**:

- 模型选择方案文档
- API 调用笔记

---

### ✅ Day 3: 修改代码准备正式实验 (10 月 7 日)

**预计时间**: 6-8 小时**GPU 时间**: 3 小时

- [x] **Task 3.1**: 创建 4 个检索算法变体

  - [x] 复制并修改为 `rankerA.py` (例如: BAAI/bge-base-en-v1.5)
  - [x] 复制并修改为 `rankerB.py` (例如: sentence-transformers/all-MiniLM-L6-v2)
  - [x] 复制并修改为 `rankerC.py` (例如: intfloat/e5-base-v2)
  - [x] 复制并修改为 `rankerD.py` (例如: BAAI/bge-small-en-v1.5)

- [x] **Task 3.2**: 使用样本数据测试所有 ranker

  ```bash
  python rankerA.py  # 测试并记录时间
  python rankerB.py
  python rankerC.py
  python rankerD.py
  ```

- [ ] **Task 3.3**: 评估检索结果

  ```bash
  python evaluate_stage_1.py --input output/rankerA.json
  python evaluate_stage_1.py --input output/rankerB.json
  python evaluate_stage_1.py --input output/rankerC.json
  python evaluate_stage_1.py --input output/rankerD.json
  ```

- [ ] **Task 3.4**: 选择最优的 2 个 ranker

  - 基于 Recall@k, Precision, MRR 等指标
  - 记录选择理由(用于报告的 Methodology 部分)

**输出文档**:

- 初步检索结果分析
- 选择最优 ranker 的理由文档

---

## Phase 2: 实验执行阶段 (Day 4-10: 10 月 8 日-10 月 14 日)

### ✅ Day 4-5: 完整检索实验 (10 月 8 日-10 月 9 日)

**预计时间**: 每天 4 小时**GPU 时间**: 8 小时 (4 个模型 × 2 小时)

- [ ] **Task 4.1**: 在完整数据集上运行 4 个 ranker

  - 修改所有 ranker 使用 `data/rag.json`
  - 编写批处理脚本运行所有实验

  ```bash
  #!/bin/bash
  python rankerA.py > logs/rankerA.log 2>&1
  python rankerB.py > logs/rankerB.log 2>&1
  python rankerC.py > logs/rankerC.log 2>&1
  python rankerD.py > logs/rankerD.log 2>&1
  ```

- [ ] **Task 4.2**: 使用 tmux 运行长时间任务

  ```bash
  tmux new -s experiments
  bash run_all_rankers.sh
  # Ctrl+B, D 分离会话
  ```

- [ ] **Task 4.3**: 定期检查进度

  - 每 4 小时检查一次日志
  - 确认没有错误
  - 监控 GPU 使用: `nvidia-smi`

- [ ] **Task 4.4**: 评估所有检索结果

  - 运行评估脚本
  - 生成对比表格
  - 确认选择的 2 个最优 ranker

**输出文件**:

- `output/rankerA.json`
- `output/rankerB.json`
- `output/rankerC.json`
- `output/rankerD.json`
- 检索结果对比表格

---

### ✅ Day 6-7: 准备 LLM 实验 (10 月 10 日-10 月 11 日)

**预计时间**: 每天 6 小时**GPU 时间**: 4 小时

- [ ] **Task 5.1**: 创建 4 个 LLM 变体

  - [ ] `ragA.py` - Llama 2 7B (baseline)
  - [ ] `ragB.py` - Llama 3.1 8B
  - [ ] `ragC.py` - Mistral 7B v0.1
  - [ ] `ragD.py` - Phi-3 Mini 4K

- [ ] **Task 5.2**: 配置 prompt template

  - 设计统一的 instruction prompt
  - 确保所有模型使用相同的 prompt 格式
  - 添加 "Insufficient Data" 回退逻辑

- [ ] **Task 5.3**: 样本数据测试

  ```bash
  python ragA.py  # 记录时间和GPU使用
  python ragB.py
  python ragC.py
  python ragD.py
  ```

- [ ] **Task 5.4**: 验证输出格式

  - 确认 JSON 格式正确
  - 验证答案质量
  - 检查 "Insufficient Data" 情况处理

**输出文档**:

- Prompt 设计文档
- LLM 配置参数记录

---

### ✅ Day 8-10: RAG 完整实验 (10 月 12 日-10 月 14 日)

**预计时间**: 每天 2 小时监控**GPU 时间**: 16 小时 (2 rankers × 4 LLMs × 2 小时)

- [ ] **Task 6.1**: 组合实验设计

  - Best Ranker A + 4 LLMs
  - Best Ranker B + 4 LLMs
  - 总共 8 个组合

- [ ] **Task 6.2**: 编写批处理脚本

  ```bash
  #!/bin/bash
  # 使用 rankerA 结果
  python ragA.py --retrieval output/rankerA.json
  python ragB.py --retrieval output/rankerA.json
  python ragC.py --retrieval output/rankerA.json
  python ragD.py --retrieval output/rankerA.json

  # 使用 rankerB 结果
  python ragA.py --retrieval output/rankerB.json
  python ragB.py --retrieval output/rankerB.json
  python ragC.py --retrieval output/rankerB.json
  python ragD.py --retrieval output/rankerB.json
  ```

- [ ] **Task 6.3**: 分批运行实验

  - Day 8: 运行前 4 个组合
  - Day 9: 运行后 4 个组合
  - Day 10: 备用时间(如有失败重跑)

- [ ] **Task 6.4**: 收集所有结果

  - 验证 8 个输出 JSON 文件
  - 确认所有结果完整

**输出文件**:

- `output/ragA_rankerA.json`
- `output/ragB_rankerA.json`
- ... (共 8 个文件)

---

## Phase 3: 分析和撰写阶段 (Day 11-17: 10 月 15 日-10 月 21 日)

### ✅ Day 11-12: 数据分析 (10 月 15 日-10 月 16 日)

**预计时间**: 每天 8 小时**GPU 时间**: 0 小时

- [ ] **Task 7.1**: 评估所有 RAG 结果

  ```bash
  for file in output/rag*.json; do
    python evaluate_rag.py --input $file
  done
  ```

- [ ] **Task 7.2**: 实现额外评估指标 (至少 2 个)

  - [ ] 例如: Exact Match (EM) accuracy
  - [ ] 例如: BLEU score
  - [ ] 例如: Answer coverage

  - 创建 `MyRAGEval.py`

- [ ] **Task 7.3**: 生成对比表格

  - Stage 1 检索性能表格
  - RAG end-to-end 性能表格
  - 各 LLM 在不同 retriever 下的表现

- [ ] **Task 7.4**: 创建可视化图表

  - [ ] 图表 1: 检索算法 Recall@k 对比
  - [ ] 图表 2: LLM 性能对比(按 retriever 分组)
  - [ ] 图表 3: F1 score 热力图
  - [ ] 图表 4+: 其他分析图表

**输出文档**:

- 实验结果分析文档
- 所有表格和图表

---

### ✅ Day 13-15: 撰写报告 (10 月 17 日-10 月 19 日)

**预计时间**: 每天 8 小时**GPU 时间**: 0 小时

- [ ] **Day 13: Introduction + Methodology**

  - [ ] Introduction (1 mark)
    - Multi-Hop QA 问题背景
    - RAG 系统重要性
    - 研究目标和贡献
  - [ ] Methodology (2 marks)
    - 详细解释 4 个检索算法工作原理
    - 详细解释 4 个 LLM 模型特点
    - 评估指标说明
    - 实验设计 rationale

- [ ] **Day 14: Experiments + Results**

  - [ ] Experiments (3 marks)
    - 实验设置详细描述
    - 数据集描述
    - 超参数配置
    - 每个表格和图表的详细分析
    - 对比分析:为什么 System A 优于 System B
    - Failure analysis: 错误案例分析

- [ ] **Day 15: Conclusions + References + Polish**

  - [ ] Conclusions (1 mark)
    - 主要发现总结
    - 最佳系统推荐
    - 局限性讨论
    - 未来工作方向
  - [ ] References (1 mark)
    - 至少 10 篇参考文献
    - 使用标准引用格式
  - [ ] 格式检查
    - 确保正好 5 页
    - 检查语法和拼写
    - 确保图表清晰

**输出文档**:

- `report.pdf` (初稿)

---

### ✅ Day 16-17: 最终检查和打包 (10 月 20 日-10 月 21 日)

**预计时间**: 每天 4 小时**GPU 时间**: 0 小时

- [ ] **Task 8.1**: 代码整理

  - 添加代码注释
  - 删除调试代码
  - 确认所有文件命名规范

- [ ] **Task 8.2**: 创建 MyREADME.md

  ```markdown
  # 运行说明

  ## 环境要求

  见 requirements.txt

  ## 运行检索实验

  python rankerA.py
  ...

  ## 运行 RAG 实验

  python ragA.py --retrieval output/rankerA.json
  ...

  ## 评估

  python evaluate_stage_1.py
  python evaluate_rag.py
  python MyRAGEval.py
  ```

- [ ] **Task 8.3**: 生成 requirements.txt

  ```bash
  pip freeze > requirements.txt
  ```

- [ ] **Task 8.4**: 最终检查清单

  - [ ] report.pdf (5 页,PDF 格式)
  - [ ] MyREADME.md
  - [ ] requirements.txt
  - [ ] rankerA.py, rankerB.py, rankerC.py, rankerD.py
  - [ ] ragA.py, ragB.py, ragC.py, ragD.py
  - [ ] MyRAGEval.py (自定义评估)
  - [ ] output/rankerA.json ... rankerD.json
  - [ ] output/ragA_rankerA.json ... (8 个文件)
  - [ ] 其他脚本和结果文件

- [ ] **Task 8.5**: 打包提交

  ```bash
  mkdir S<学号>
  cp <所有需要的文件> S<学号>/
  zip -r S<学号>.zip S<学号>/
  ```

---

### ✅ Day 18-19: 缓冲时间和最终提交 (10 月 22 日-10 月 24 日)

- [ ] **Day 18 (10 月 22 日)**: 报告最终修改

  - 同伴互审
  - 教师 office hour 咨询
  - 最终润色

- [ ] **Day 19 (10 月 23 日)**: 提交前验证

  - 解压 zip 文件测试
  - 验证所有文件完整
  - 最后检查 plagiarism

- [ ] **Day 19 晚上**: 提交到 Blackboard

  - 在截止日期前提交
  - 确认提交成功
  - 保存提交确认邮件

---

## 📊 GPU 时间预算监控

| 任务阶段      | 预计 GPU 时间 | 实际使用 | 剩余配额    |
| ------------- | ------------- | -------- | ----------- |
| 初始配额      | -             | -        | 40h         |
| 环境测试      | 1h            | [ ]      | -           |
| Ranker 测试   | 3h            | [ ]      | -           |
| LLM 测试      | 4h            | [ ]      | -           |
| 完整检索实验  | 8h            | [ ]      | -           |
| 完整 RAG 实验 | 16h           | [ ]      | -           |
| **总计**      | **32h**       | **[ ]**  | **8h 缓冲** |

---

## ⚠️ 风险管理

### 高风险项

1. **GPU 超时**: 任务运行时间超过预期
   - 缓解: 先用样本数据测试,确认运行时间
2. **内存溢出**: 模型太大导致 OOM
   - 缓解: 选择 7B-13B 参数范围内的模型
3. **时间管理失败**: 拖延导致最后匆忙完成
   - 缓解: 严格遵循每日任务清单

### 应急计划

- 如果某个 LLM 模型不可用,准备备选模型列表
- 如果 GPU 配额不足,优先保证最小实验组合(2 ranker × 3 LLM)
- 保持 daily backup 所有结果文件

---

## 📋 每日检查清单模板

```markdown
### 日期: \_**\_年**月\_\_日

#### 今日目标

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

#### 今日完成

- [ ] 完成 Task 1 (用时: \_\_小时)
- [ ] 完成 Task 2 (用时: \_\_小时)
- [ ] 部分完成 Task 3

#### GPU 使用

- 今日使用: \_\_小时
- 累计使用: \_\_小时
- 剩余配额: \_\_小时

#### 遇到的问题

1. 问题描述...
   解决方案...

#### 明日计划

- [ ] 继续 Task 3
- [ ] 开始 Task 4
```

---

## 🎓 学习目标检查

通过完成此项目,你应该能够:

- [ ] 理解 RAG 系统的完整架构
- [ ] 掌握多个检索算法的实现和对比
- [ ] 熟练使用 LLM 进行文本生成
- [ ] 设计和执行全面的实验对比研究
- [ ] 撰写规范的学术实验报告
- [ ] 管理 GPU 资源和大规模实验
- [ ] 使用 Llama Index API 进行快速原型开发

---

## 📞 紧急联系

- **Office Hours**: 每周二下午 4 点
- **Lab Session**: 每周三
- **Ed Forum**: 在线提问
- **Email**: 直接联系教师

---

**创建日期**: 2025-10-05
**最后更新**: 2025-10-05
**路线图版本**: v1.0

---

## 💡 成功提示

1. **早开始,多迭代**: 不要等到最后一周才开始
2. **充分测试**: 在样本数据上验证所有代码
3. **详细记录**: 记录每个实验的配置和结果
4. **定期备份**: 每天备份代码和结果到本地
5. **寻求帮助**: 遇到问题及时 ask,不要独自挣扎
6. **时间缓冲**: 总是预留额外时间应对意外
7. **质量优先**: 与其做 10 个粗糙实验,不如做 6 个高质量实验

**Good luck! 🚀**
