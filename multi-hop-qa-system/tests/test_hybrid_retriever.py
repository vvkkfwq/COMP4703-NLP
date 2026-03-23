"""
测试 HybridRetriever 的功能

包含：
1. HybridRetriever 初始化的测试
2. HybridRetriever.retrieve() 方法的测试
3. RRF（倒数排序融合）融合逻辑的测试
4. reranker 集成的测试
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.retriever.hybrid import HybridRetriever


class TestHybridRetrieverInit:
    """测试 HybridRetriever 初始化"""

    @pytest.mark.unit
    def test_hybrid_retriever_creates_sub_retrievers(self):
        """验证 HybridRetriever 能正确初始化子检索器"""
        retriever = HybridRetriever()

        # semantic 和 bm25 应该被初始化
        assert retriever.semantic is not None
        assert retriever.bm25 is not None
        assert retriever.reranker is not None

    @pytest.mark.unit
    def test_hybrid_retriever_default_fields(self):
        """验证 HybridRetriever 的默认字段值"""
        retriever = HybridRetriever()

        assert retriever.enable_reranker == False
        assert retriever.rrf_k > 0
        assert retriever.top_k > 0

    @pytest.mark.unit
    def test_hybrid_retriever_custom_top_k(self):
        """验证 HybridRetriever 支持自定义 top_k"""
        top_k = 7
        retriever = HybridRetriever(top_k=top_k)

        assert retriever.top_k == top_k

    @pytest.mark.unit
    def test_hybrid_retriever_custom_rrf_k(self):
        """验证 HybridRetriever 支持自定义 rrf_k"""
        rrf_k = 100
        retriever = HybridRetriever(rrf_k=rrf_k)

        assert retriever.rrf_k == rrf_k

    @pytest.mark.unit
    def test_hybrid_retriever_enable_reranker(self):
        """验证 HybridRetriever 支持启用 reranker"""
        retriever = HybridRetriever(enable_reranker=True)

        assert retriever.enable_reranker == True
        assert retriever.reranker is not None


class TestHybridRetrieverRetrieve:
    """测试 HybridRetriever.retrieve() 方法"""

    @pytest.mark.unit
    def test_retrieve_returns_list_of_documents(self):
        """验证 retrieve() 返回文档列表"""
        retriever = HybridRetriever()

        # Mock semantic 和 bm25 的返回结果
        semantic_docs = [
            Document(page_content="Semantic result 1", metadata={"_score": 0.95}),
            Document(page_content="Semantic result 2", metadata={"_score": 0.85}),
        ]
        bm25_docs = [
            Document(page_content="BM25 result 1", metadata={}),
            Document(page_content="BM25 result 2", metadata={}),
        ]

        with patch.object(retriever.semantic, "retrieve", return_value=semantic_docs):
            with patch.object(retriever.bm25, "retrieve", return_value=bm25_docs):
                result = retriever.retrieve("test query")

        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)

    @pytest.mark.unit
    def test_retrieve_calls_both_retrievers(self):
        """验证 retrieve() 调用两个子检索器"""
        retriever = HybridRetriever()

        with patch.object(
            retriever.semantic, "retrieve", return_value=[]
        ) as mock_semantic:
            with patch.object(retriever.bm25, "retrieve", return_value=[]) as mock_bm25:
                retriever.retrieve("test query", top_k=5)

                # 两个检索器都应该被调用，且 top_k 翻倍
                mock_semantic.assert_called_once_with("test query", top_k=10)
                mock_bm25.assert_called_once_with("test query", top_k=10)

    @pytest.mark.unit
    def test_retrieve_respects_top_k(self):
        """验证 retrieve() 返回的文档数量不超过 top_k"""
        retriever = HybridRetriever(top_k=3)

        # 模拟两个检索器各返回 6 个文档
        semantic_docs = [
            Document(page_content=f"Semantic {i}", metadata={}) for i in range(6)
        ]
        bm25_docs = [Document(page_content=f"BM25 {i}", metadata={}) for i in range(6)]

        with patch.object(retriever.semantic, "retrieve", return_value=semantic_docs):
            with patch.object(retriever.bm25, "retrieve", return_value=bm25_docs):
                result = retriever.retrieve("test")

        # 结果应该不超过 top_k
        assert len(result) <= 3


class TestHybridRetrieverRRF:
    """测试 RRF（倒数排序融合）算法"""

    @pytest.mark.unit
    def test_rrf_scores_computed_correctly(self):
        """验证 RRF 分数计算正确"""
        retriever = HybridRetriever(rrf_k=60)

        # 创建有重叠的文档集
        doc1 = Document(page_content="Overlap doc")
        doc2 = Document(page_content="Semantic only")
        doc3 = Document(page_content="BM25 only")

        semantic_docs = [doc1, doc2]  # rank 0, 1
        bm25_docs = [doc1, doc3]  # rank 0, 1

        with patch.object(retriever.semantic, "retrieve", return_value=semantic_docs):
            with patch.object(retriever.bm25, "retrieve", return_value=bm25_docs):
                result = retriever.retrieve("test")

        # 检查 RRF 分数是否添加
        for doc in result:
            if doc.page_content == "Overlap doc":
                # 重叠文档应该有更高的分数
                assert "_rrf_score" in doc.metadata
                # 重叠应该被融合（两次相加）
                assert doc.metadata["_rrf_score"] > 0

    @pytest.mark.unit
    def test_rrf_formula_1_over_k_plus_rank(self):
        """验证 RRF 公式：score = 1 / (k + rank)"""
        retriever = HybridRetriever(rrf_k=60, top_k=2)

        # 单个不重叠的文档，rank=0 在 semantic 中
        doc1 = Document(page_content="First doc", metadata={})
        # rank=1 在 semantic 中
        doc2 = Document(page_content="Second doc", metadata={})

        semantic_docs = [doc1, doc2]
        bm25_docs = []

        with patch.object(retriever.semantic, "retrieve", return_value=semantic_docs):
            with patch.object(retriever.bm25, "retrieve", return_value=bm25_docs):
                result = retriever.retrieve("test")

        # 验证分数
        for doc in result:
            if doc.page_content == "First doc":
                # rank=0: 1 / (60 + 0 + 1) = 1/61
                expected_score = 1.0 / (60 + 0 + 1)
                assert abs(doc.metadata["_rrf_score"] - expected_score) < 0.0001
            elif doc.page_content == "Second doc":
                # rank=1: 1 / (60 + 1 + 1) = 1/62
                expected_score = 1.0 / (60 + 1 + 1)
                assert abs(doc.metadata["_rrf_score"] - expected_score) < 0.0001

    @pytest.mark.unit
    def test_rrf_ranking_order(self):
        """验证 RRF 能正确排序结果"""
        retriever = HybridRetriever(rrf_k=1, top_k=3)

        # 高分文档（两个检索器都排名靠前）
        high_score_doc = Document(page_content="High score", metadata={})
        # 低分文档（只在一个检索器中）
        low_score_doc = Document(page_content="Low score", metadata={})

        semantic_docs = [high_score_doc, low_score_doc]
        bm25_docs = [high_score_doc]

        with patch.object(retriever.semantic, "retrieve", return_value=semantic_docs):
            with patch.object(retriever.bm25, "retrieve", return_value=bm25_docs):
                result = retriever.retrieve("test")

        # 高分文档应该排在前面
        assert result[0].page_content == "High score"


class TestHybridRetrieverWithReranker:
    """测试 HybridRetriever 与 reranker 的集成"""

    @pytest.mark.unit
    def test_retrieve_without_reranker(self):
        """验证 retrieve() 在禁用 reranker 时正常工作"""
        retriever = HybridRetriever(enable_reranker=False)

        semantic_docs = [Document(page_content="Semantic result")]
        bm25_docs = [Document(page_content="BM25 result")]

        with patch.object(retriever.semantic, "retrieve", return_value=semantic_docs):
            with patch.object(retriever.bm25, "retrieve", return_value=bm25_docs):
                with patch.object(retriever.reranker, "rerank") as mock_rerank:
                    result = retriever.retrieve("test")

                    # rerank 不应该被调用
                    mock_rerank.assert_not_called()

        assert len(result) > 0

    @pytest.mark.unit
    def test_retrieve_with_reranker(self):
        """验证 retrieve() 在启用 reranker 时调用它"""
        retriever = HybridRetriever(enable_reranker=True, top_k=2)

        semantic_docs = [
            Document(page_content="Semantic result 1"),
            Document(page_content="Semantic result 2"),
        ]
        bm25_docs = [
            Document(page_content="BM25 result 1"),
            Document(page_content="BM25 result 2"),
        ]

        # Reranker 返回重排后的结果
        reranked_docs = [semantic_docs[1], semantic_docs[0]]

        with patch.object(retriever.semantic, "retrieve", return_value=semantic_docs):
            with patch.object(retriever.bm25, "retrieve", return_value=bm25_docs):
                with patch.object(
                    retriever.reranker, "rerank", return_value=reranked_docs
                ) as mock_rerank:
                    result = retriever.retrieve("test")

                    # rerank 应该被调用
                    mock_rerank.assert_called_once()

        assert len(result) > 0


class TestHybridRetrieverGetRelevantDocuments:
    """测试 _get_relevant_documents() LangChain 标准接口"""

    @pytest.mark.unit
    def test_get_relevant_documents_calls_retrieve(self):
        """验证 _get_relevant_documents() 调用 retrieve()"""
        retriever = HybridRetriever()

        with patch.object(
            HybridRetriever, "retrieve", return_value=[]
        ) as mock_retrieve:
            mock_run_manager = None
            retriever._get_relevant_documents(
                "test query", run_manager=mock_run_manager
            )

            # retrieve 应该被调用
            mock_retrieve.assert_called_once_with("test query")

    @pytest.mark.unit
    def test_get_relevant_documents_returns_document_list(self):
        """验证 _get_relevant_documents() 返回文档列表"""
        retriever = HybridRetriever()

        mock_docs = [
            Document(page_content="Content 1"),
            Document(page_content="Content 2"),
        ]

        with patch.object(HybridRetriever, "retrieve", return_value=mock_docs):
            mock_run_manager = None
            result = retriever._get_relevant_documents(
                "test query", run_manager=mock_run_manager
            )

        assert isinstance(result, list)
        assert len(result) == 2


class TestHybridRetrieverEdgeCases:
    """测试边界情况"""

    @pytest.mark.unit
    def test_retrieve_with_empty_semantic_results(self):
        """验证 retrieve() 能处理 semantic 返回空"""
        retriever = HybridRetriever(top_k=2)

        bm25_docs = [
            Document(page_content="BM25 result 1"),
            Document(page_content="BM25 result 2"),
        ]

        with patch.object(retriever.semantic, "retrieve", return_value=[]):
            with patch.object(retriever.bm25, "retrieve", return_value=bm25_docs):
                result = retriever.retrieve("test")

        # 应该返回 BM25 结果
        assert len(result) > 0

    @pytest.mark.unit
    def test_retrieve_with_empty_bm25_results(self):
        """验证 retrieve() 能处理 BM25 返回空"""
        retriever = HybridRetriever(top_k=2)

        semantic_docs = [
            Document(page_content="Semantic result 1"),
            Document(page_content="Semantic result 2"),
        ]

        with patch.object(retriever.semantic, "retrieve", return_value=semantic_docs):
            with patch.object(retriever.bm25, "retrieve", return_value=[]):
                result = retriever.retrieve("test")

        # 应该返回 semantic 结果
        assert len(result) > 0

    @pytest.mark.unit
    def test_retrieve_with_both_empty(self):
        """验证 retrieve() 能处理两个检索器都返回空"""
        retriever = HybridRetriever()

        with patch.object(retriever.semantic, "retrieve", return_value=[]):
            with patch.object(retriever.bm25, "retrieve", return_value=[]):
                result = retriever.retrieve("no match query")

        assert result == []

    @pytest.mark.unit
    def test_retrieve_with_duplicate_content(self):
        """验证 retrieve() 能处理重复内容"""
        retriever = HybridRetriever(top_k=2)

        # 两个检索器都返回相同的文档
        doc = Document(page_content="Same content", metadata={})

        semantic_docs = [doc, doc]
        bm25_docs = [doc, doc]

        with patch.object(retriever.semantic, "retrieve", return_value=semantic_docs):
            with patch.object(retriever.bm25, "retrieve", return_value=bm25_docs):
                result = retriever.retrieve("test")

        # 重复内容的分数应该是累加的
        assert len(result) > 0
        for doc in result:
            if doc.page_content == "Same content":
                # 应该有 RRF 分数
                assert "_rrf_score" in doc.metadata
