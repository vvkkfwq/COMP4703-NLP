"""
测试 SemanticRetriever 的功能

包含：
1. SemanticRetriever 初始化的测试
2. SemanticRetriever.retrieve() 方法的测试
3. _get_relevant_documents() 的测试
4. reranker 配置的测试
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.retriever.semantic import SemanticRetriever


class TestSemanticRetrieverInit:
    """测试 SemanticRetriever 初始化"""

    @pytest.mark.unit
    def test_semantic_retriever_has_default_values(self):
        """验证 SemanticRetriever 有合理的默认值"""
        retriever = SemanticRetriever()

        assert retriever.top_k > 0
        assert retriever.enable_reranker == False
        assert retriever.vectorstore is not None
        assert retriever.reranker is not None

    @pytest.mark.unit
    def test_semantic_retriever_custom_top_k(self):
        """验证 SemanticRetriever 支持自定义 top_k"""
        top_k = 5
        retriever = SemanticRetriever(top_k=top_k)

        assert retriever.top_k == top_k

    @pytest.mark.unit
    def test_semantic_retriever_enable_reranker(self):
        """验证 SemanticRetriever 支持启用 reranker"""
        retriever = SemanticRetriever(enable_reranker=True)

        assert retriever.enable_reranker == True
        assert retriever.reranker is not None

    @pytest.mark.unit
    def test_semantic_retriever_vectorstore_initialized(self):
        """验证 vectorstore 已正确初始化"""
        retriever = SemanticRetriever()

        # vectorstore 不应该为 None
        assert retriever.vectorstore is not None
        # vectorstore 应该有搜索方法
        assert hasattr(retriever.vectorstore, "similarity_search_with_score")

    @pytest.mark.unit
    def test_semantic_retriever_reranker_always_loaded(self):
        """验证 reranker 总是被加载"""
        # enable_reranker=False
        retriever1 = SemanticRetriever(enable_reranker=False)
        assert retriever1.reranker is not None

        # enable_reranker=True
        retriever2 = SemanticRetriever(enable_reranker=True)
        assert retriever2.reranker is not None


class TestSemanticRetrieverRetrieve:
    """测试 SemanticRetriever.retrieve() 方法"""

    @pytest.mark.unit
    def test_retrieve_returns_list_of_documents(self):
        """验证 retrieve() 返回文档列表"""
        retriever = SemanticRetriever()

        # Mock vectorstore 的搜索响应
        mock_docs = [
            (Document(page_content="Content 1"), 0.95),
            (Document(page_content="Content 2"), 0.85),
        ]

        with patch.object(
            retriever.vectorstore,
            "similarity_search_with_score",
            return_value=mock_docs,
        ):
            result = retriever.retrieve("test query")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(doc, Document) for doc in result)

    @pytest.mark.unit
    def test_retrieve_respects_top_k(self):
        """验证 retrieve() 尊重 top_k 参数"""
        retriever = SemanticRetriever(top_k=3)

        mock_docs = [
            (Document(page_content=f"Content {i}"), 0.9 - i * 0.1) for i in range(5)
        ]

        with patch.object(
            retriever.vectorstore,
            "similarity_search_with_score",
            return_value=mock_docs[:3],
        ):
            result = retriever.retrieve("test", top_k=3)

        assert len(result) <= 3

    @pytest.mark.unit
    def test_retrieve_adds_score_metadata(self):
        """验证 retrieve() 添加 _score 到 metadata"""
        retriever = SemanticRetriever()

        mock_docs = [
            (Document(page_content="Content 1", metadata={}), 0.95),
            (Document(page_content="Content 2", metadata={}), 0.85),
        ]

        with patch.object(
            retriever.vectorstore,
            "similarity_search_with_score",
            return_value=mock_docs,
        ):
            result = retriever.retrieve("test query")

        # 每个文档都应该有 _score
        for doc in result:
            assert "_score" in doc.metadata
            assert isinstance(doc.metadata["_score"], float)

    @pytest.mark.unit
    def test_retrieve_without_reranker(self):
        """验证 retrieve() 在禁用 reranker 时正常工作"""
        retriever = SemanticRetriever(enable_reranker=False)

        mock_docs = [
            (Document(page_content="Content 1"), 0.95),
            (Document(page_content="Content 2"), 0.85),
        ]

        with patch.object(
            retriever.vectorstore,
            "similarity_search_with_score",
            return_value=mock_docs,
        ):
            with patch.object(retriever.reranker, "rerank") as mock_rerank:
                result = retriever.retrieve("test query")

                # rerank 不应该被调用
                mock_rerank.assert_not_called()

        assert len(result) > 0

    @pytest.mark.unit
    def test_retrieve_with_reranker(self):
        """验证 retrieve() 在启用 reranker 时调用它"""
        retriever = SemanticRetriever(enable_reranker=True)

        mock_docs = [
            (Document(page_content="Content 1"), 0.95),
            (Document(page_content="Content 2"), 0.85),
        ]

        # 模拟 reranker 返回重排后的结果
        reranked_docs = [mock_docs[1][0], mock_docs[0][0]]  # 反序

        with patch.object(
            retriever.vectorstore,
            "similarity_search_with_score",
            return_value=mock_docs,
        ):
            with patch.object(
                retriever.reranker, "rerank", return_value=reranked_docs
            ) as mock_rerank:
                result = retriever.retrieve("test query")

                # rerank 应该被调用
                mock_rerank.assert_called_once()

        assert len(result) > 0


class TestSemanticRetrieverGetRelevantDocuments:
    """测试 _get_relevant_documents() LangChain 标准接口"""

    @pytest.mark.unit
    def test_get_relevant_documents_calls_retrieve(self):
        """验证 _get_relevant_documents() 调用 retrieve()"""
        retriever = SemanticRetriever()

        with patch.object(
            SemanticRetriever, "retrieve", return_value=[]
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
        retriever = SemanticRetriever()

        mock_docs = [
            Document(page_content="Content 1"),
            Document(page_content="Content 2"),
        ]

        with patch.object(SemanticRetriever, "retrieve", return_value=mock_docs):
            mock_run_manager = None
            result = retriever._get_relevant_documents(
                "test query", run_manager=mock_run_manager
            )

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(doc, Document) for doc in result)


class TestSemanticRetrieverFetchKLogic:
    """测试 fetch_k 逻辑（reranker 时多取候选）"""

    @pytest.mark.unit
    def test_fetch_k_without_reranker(self):
        """验证 reranker 禁用时 fetch_k = k"""
        retriever = SemanticRetriever(top_k=5, enable_reranker=False)

        mock_docs = [
            (Document(page_content=f"Content {i}"), 0.9 - i * 0.1) for i in range(5)
        ]

        with patch.object(
            retriever.vectorstore,
            "similarity_search_with_score",
            return_value=mock_docs,
        ) as mock_search:
            retriever.retrieve("test")

            # 应该查询 k 个文档
            call_args = mock_search.call_args
            assert call_args[1]["k"] == 5

    @pytest.mark.unit
    def test_fetch_k_with_reranker(self):
        """验证 reranker 启用时 fetch_k = k * 3"""
        retriever = SemanticRetriever(top_k=5, enable_reranker=True)

        mock_docs = [
            (Document(page_content=f"Content {i}"), 0.9 - i * 0.1) for i in range(15)
        ]

        with patch.object(
            retriever.vectorstore,
            "similarity_search_with_score",
            return_value=mock_docs,
        ) as mock_search:
            with patch.object(retriever.reranker, "rerank", return_value=mock_docs[:5]):
                retriever.retrieve("test")

                # 应该查询 k*3 个文档
                call_args = mock_search.call_args
                assert call_args[1]["k"] == 15


class TestSemanticRetrieverEdgeCases:
    """测试边界情况"""

    @pytest.mark.unit
    def test_retrieve_with_empty_results(self):
        """验证 retrieve() 能处理空搜索结果"""
        retriever = SemanticRetriever()

        with patch.object(
            retriever.vectorstore, "similarity_search_with_score", return_value=[]
        ):
            result = retriever.retrieve("no match query")

        assert result == []

    @pytest.mark.unit
    def test_retrieve_with_zero_top_k(self):
        """验证 retrieve() 使用默认 top_k"""
        retriever = SemanticRetriever(top_k=10)

        mock_docs = [(Document(page_content=f"Content {i}"), 0.9) for i in range(10)]

        with patch.object(
            retriever.vectorstore,
            "similarity_search_with_score",
            return_value=mock_docs,
        ):
            result = retriever.retrieve("test", top_k=None)

            # 应该使用默认 top_k
            assert len(result) <= 10

    @pytest.mark.unit
    def test_retrieve_score_type_conversion(self):
        """验证 _score 转换为 float"""
        retriever = SemanticRetriever()

        # 模拟返回的 score 是其他类型（e.g. numpy.float64）
        mock_docs = [
            (Document(page_content="Content 1"), 0.95),
        ]

        with patch.object(
            retriever.vectorstore,
            "similarity_search_with_score",
            return_value=mock_docs,
        ):
            result = retriever.retrieve("test")

        # _score 应该是 float
        assert isinstance(result[0].metadata["_score"], float)
