# 检索器内存优化技术报告

## 一、报告信息

- 报告日期：2026-03-23
- 主题：检索器实例化与缓存策略优化
- 适用模块：Streamlit UI、SemanticRetriever、HybridRetriever

## 二、执行摘要

本次优化针对“切换配置后内存持续上涨”问题，核心做法是将缓存策略从“配置组合级”调整为“组件级”，并将 reranker 改为按需加载。优化后，重资源（embedding/chroma/bm25）复用率提升，重复创建显著减少，测试通过且功能行为保持一致。

## 三、问题描述

在 UI 中频繁切换以下配置后，进程内存持续上升：

- Embedding 模型切换
- BM25 开关切换
- Reranker 开关切换
- 多策略对比模式切换

问题本质是：旧实例被缓存保留，同时新实例不断创建，尤其是重模型对象（cross-encoder reranker）与检索器子组件重复初始化。

## 四、根因分析

### 4.1 SemanticRetriever 无条件创建 reranker

- 关闭 reranker 时仍初始化 cross-encoder。
- 导致语义检索与混合检索（不重排）路径也承担额外内存。

### 4.2 HybridRetriever 覆盖外部注入依赖

- model_post_init 中重建 semantic/bm25，破坏 UI 层缓存复用。
- 每次策略切换都可能产生额外实例。

### 4.3 UI 缓存键设计过宽

- semantic 缓存键绑定 reranker 开关，导致同一模型出现多份 semantic 缓存条目。

### 4.4 Pipeline 长期缓存收益低

- pipeline 对象本身并非最重，但配置组合缓存会增加驻留对象总量。

## 五、优化目标与原则

1. 重资源按最小必要维度缓存。
2. 依赖注入优先，不在 post-init 覆盖外部实例。
3. reranker 按需加载。
4. pipeline 轻量重建，重状态下沉到 retriever 组件。

## 六、实施方案

### 6.1 语义检索器优化

相对路径：src/retriever/semantic.py

- vectorstore 仅在未注入时初始化。
- reranker 仅在 enable_reranker=True 时创建。
- 关闭 reranker 时主动置空引用。
- 执行 rerank 前同时检查开关与实例。

### 6.2 混合检索器优化

相对路径：src/retriever/hybrid.py

- semantic/bm25 仅在为空时初始化。
- reranker 仅在启用且为空时初始化。
- 运行重排前检查 reranker 实例存在。

### 6.3 UI 构建与缓存优化

相对路径：src/ui/app.py

- \_get_semantic_retriever 仅按 model_key 缓存。
- 基础 semantic 固定不启用 reranker，作为共享核心。
- semantic+rerank 路径通过轻量包装复用已缓存 vectorstore。
- 移除 build_pipeline 的资源缓存，避免按配置组合常驻。

## 七、核心代码摘录

### 7.1 src/retriever/semantic.py（按需初始化与复用）

```python
def model_post_init(self, context: Any) -> None:
    if self.vectorstore is None:
        embed = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": EMBED_DEVICE},
        )
        self.vectorstore = Chroma(
            persist_directory=str(self.chroma_dir), embedding_function=embed
        )

    if self.enable_reranker and self.reranker is None:
        self.reranker = CrossEncoderReranker()
    if not self.enable_reranker:
        self.reranker = None

    super().model_post_init(context)
```

```python
if self.enable_reranker and self.reranker is not None:
    docs = self.reranker.rerank(query, docs, top_k=k)
```

### 7.2 src/retriever/hybrid.py（不覆盖注入实例）

```python
def model_post_init(self, context: Any) -> None:
    if self.semantic is None:
        self.semantic = SemanticRetriever()
    if self.bm25 is None:
        self.bm25 = BM25Retriever()
    if self.enable_reranker and self.reranker is None:
        self.reranker = CrossEncoderReranker()

    super().model_post_init(context)
```

```python
if self.enable_reranker and self.reranker is not None:
    fused = self.reranker.rerank(query, fused, top_k=k)
```

### 7.3 src/ui/app.py（组件级缓存）

```python
@st.cache_resource(show_spinner="Loading embedding model…")
def _get_semantic_retriever(model_key: str) -> SemanticRetriever:
    cfg = EMBED_MODELS[model_key]
    return SemanticRetriever(
        chroma_dir=str(cfg["chroma_dir"]),
        embedding_model=cfg["model_name"],
        enable_reranker=False,
    )
```

```python
def _build_retriever(model_key: str, use_bm25: bool, enable_reranker: bool):
    semantic = _get_semantic_retriever(model_key)
    if use_bm25:
        return HybridRetriever(
            semantic=semantic,
            bm25=_get_bm25_retriever(),
            enable_reranker=enable_reranker,
        )
    if enable_reranker:
        return SemanticRetriever(
            chroma_dir=semantic.chroma_dir,
            embedding_model=semantic.embedding_model,
            top_k=semantic.top_k,
            enable_reranker=True,
            vectorstore=semantic.vectorstore,
        )
    return semantic
```

## 八、验证结果

执行命令：

```bash
conda run -n rag pytest -q tests/test_semantic_retriever.py tests/test_hybrid_retriever.py tests/test_rag_pipeline.py
```

结果：

- 57 passed
- 相关变更文件静态诊断无错误

## 九、结论

本次优化通过“缓存键收敛 + 依赖注入复用 + reranker 按需加载”三项措施，显著降低了配置切换引发的额外内存占用。系统仍会在部分场景创建新对象（预期），但重资源复用路径已建立，整体内存增长趋势得到抑制。

## 十、后续建议

1. 在 UI 增加“一键清缓存”按钮（resource/data cache）。
2. 对 cache_resource 增加 max_entries 或 ttl，防止长时运行累计。
3. 增加运行时内存监测面板，支持配置切换前后对比。
4. 增加回归测试，确保 HybridRetriever 不覆盖注入 semantic/bm25。
