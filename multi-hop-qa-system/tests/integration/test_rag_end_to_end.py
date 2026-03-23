"""
端到端集成测试 — 验证完整 RAG 系统的功能

这个文件测试整个系统的集成，包括：
1. 从真实数据库检索（或 mock）
2. 完整的 run() 流程
3. 完整的 stream() 流程
4. 端到端的质量

提示：这些测试需要 OPENAI_API_KEY 才能运行真实 LLM。
      如果没有 API key，会自动跳过。
"""

import os

import pytest
from unittest.mock import patch

from src.retriever.hybrid import HybridRetriever
from src.pipeline.rag import ChatOpenAI, RAGPipeline


@pytest.fixture
def fake_openai_api_key(monkeypatch):
    """Provide a placeholder API key for tests that mock LLM calls."""
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "test-key"))


class TestRAGEndToEndWithMock:
    """使用 mock 的端到端测试（快速，无网络对 LLM 的依赖）"""

    @pytest.mark.integration
    def test_end_to_end_run_returns_complete_result(
        self, mock_retriever, fake_openai_api_key
    ):
        """验证 run() 返回完整的结构化结果"""
        del fake_openai_api_key
        # 你需要做什么：
        # 1. 创建 RAGPipeline，使用 mock_retriever
        pipeline = RAGPipeline(retriever=mock_retriever)

        # 2. 使用 mock/patch 来模拟 LLM 的响应，避免真实 API 调用
        # （参考 test_rag_pipeline.py 中的例子）
        mock_response = "Artificial intelligence is a rapidly evolving field."

        with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
            # 3. 调用 run()
            result = pipeline.run("What is artificial intelligence?")

        # 4. 验证结果结构（提示：使用 conftest.py 中的 assert_pipeline_output_structure 函数）
        from tests.conftest import assert_pipeline_output_structure

        assert_pipeline_output_structure(result)

        # 5. 验证特定内容
        assert "artificial intelligence" in result["answer"].lower()

    @pytest.mark.integration
    def test_end_to_end_stream_produces_complete_response(
        self, mock_retriever, fake_openai_api_key
    ):
        """验证 stream() 能完整生成响应"""
        del fake_openai_api_key
        # 你需要做什么：
        # 1. 创建 RAGPipeline
        pipeline = RAGPipeline(retriever=mock_retriever)

        mock_tokens = [
            "Machine",
            " learning",
            " is",
            " a",
            " subset",
            " of",
            " artificial",
            " intelligence",
            " that",
            " focuses",
            " on",
            " learning",
            " from",
            " data",
            ".",
        ]

        with patch.object(ChatOpenAI, "stream", return_value=iter(mock_tokens)):
            # 3. 调用 stream() 并收集所有 token
            tokens = list(pipeline.stream("What is machine learning?"))

        # 4. 验证
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.integration
    def test_end_to_end_multiple_queries(self, mock_retriever, fake_openai_api_key):
        """验证系统能处理多个连续查询"""
        del fake_openai_api_key
        # 你需要做什么：
        # 测试多个不同的查询，确保系统的健壮性

        pipeline = RAGPipeline(retriever=mock_retriever)

        queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are neural networks?",
        ]

        for query in queries:
            mock_response = f"Answer to: {query}"

            with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
                # 4. 调用 run() 并验证
                result = pipeline.run(query)

                # 5. 每个查询都应该返回有效结果
                from tests.conftest import assert_pipeline_output_structure

                assert_pipeline_output_structure(result)


class TestRAGEndToEndWithRealData:
    """使用真实数据的端到端测试"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_with_real_hybrid_retriever(self, openai_api_key):
        """验证使用真实 HybridRetriever 的完整流程

        这需要：
        - OPENAI_API_KEY 环境变量
        - 已构建的 Chroma 数据库（chroma_db/）
        - 已构建的 BM25 索引（bm25_index.pkl）
        """
        # 你需要做什么：
        # 1. 创建真实的 HybridRetriever（不用 mock）
        retriever = HybridRetriever(top_k=3)

        # 2. 创建 RAGPipeline（这次 LLM 会真实调用 OpenAI）
        pipeline = RAGPipeline(retriever=retriever)

        # 3. 调用 run()
        result = pipeline.run("What is machine learning?")

        # 4. 验证结果
        from tests.conftest import assert_pipeline_output_structure

        assert_pipeline_output_structure(result)

        # 5. 可选：验证特定的质量指标
        assert len(result["answer"]) > 50, "Answer should not be too short"
        assert len(result["docs"]) > 0, "Should retrieve documents"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_stream_with_real_data(self, openai_api_key):
        """验证使用真实数据的流式响应

        这需要：
        - OPENAI_API_KEY 环境变量
        - 真实的检索系统
        """
        # 你需要做什么：
        # 1. 创建真实的检索器和 pipeline
        retriever = HybridRetriever(top_k=3)
        pipeline = RAGPipeline(retriever=retriever)

        # 2. 调用 stream()
        tokens = []
        for token in pipeline.stream("What is artificial intelligence?"):
            tokens.append(token)
            # 安全限制：最多收集 100 个 token
            if len(tokens) >= 100:
                break

        # 3. 验证
        assert len(tokens) > 0, "Should produce at least one token"
        full_response = "".join(tokens)
        assert len(full_response) > 20, "Response should not be too short"


class TestRAGWithDifferentRetrievalConfigs:
    """测试不同的检索配置"""

    @pytest.mark.integration
    def test_hybrid_retriever_with_reranker_disabled(self, mock_retriever):
        """验证混合检索不使用 reranker"""
        # 你需要做什么：
        # 创建禁用 reranker 的检索器
        retriever = HybridRetriever(enable_reranker=False, top_k=5)

        # 验证配置
        assert retriever.enable_reranker == False

        # 可选：构建 pipeline 并测试

    @pytest.mark.integration
    def test_hybrid_retriever_with_reranker_enabled(self, mock_retriever):
        """验证混合检索使用 reranker"""
        # 你需要做什么：
        # 创建启用 reranker 的检索器
        retriever = HybridRetriever(enable_reranker=True, top_k=5)

        # 验证配置
        assert retriever.enable_reranker == True

        # 可选：构建 pipeline 并测试

    @pytest.mark.integration
    def test_different_top_k_values(self, mock_retriever, fake_openai_api_key):
        """验证系统支持不同的 top_k 值"""
        del mock_retriever, fake_openai_api_key
        # 你需要做什么：
        # 测试不同的 top_k 值

        for top_k in [1, 3, 5, 10]:
            retriever = HybridRetriever(top_k=top_k)
            pipeline = RAGPipeline(retriever=retriever)

            mock_response = "Test response"

            with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
                result = pipeline.run("Test query")

            # 验证返回的文档数量不超过 top_k
            assert len(result["docs"]) <= top_k


class TestRAGErrorHandling:
    """测试错误处理和边界情况"""

    @pytest.mark.integration
    def test_run_with_empty_query(self, mock_retriever, fake_openai_api_key):
        """验证系统能处理空查询"""
        del fake_openai_api_key
        # 你需要做什么：
        # 验证空字符串查询是否被正确处理

        pipeline = RAGPipeline(retriever=mock_retriever)
        mock_response = "Please provide a valid question"

        with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
            # 这应该不会崩溃
            result = pipeline.run("")

        assert isinstance(result, dict)

    @pytest.mark.integration
    def test_run_with_very_long_query(self, mock_retriever, fake_openai_api_key):
        """验证系统能处理长查询"""
        del fake_openai_api_key
        # 你需要做什么：
        # 创建一个很长的查询字符串

        pipeline = RAGPipeline(retriever=mock_retriever)
        long_query = " ".join(["question"] * 100)

        mock_response = "Response to long query"

        with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
            result = pipeline.run(long_query)

        assert result["question"] == long_query


class TestRAGPerformance:
    """性能和资源相关的测试"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_multiple_sequential_calls(self, mock_retriever, fake_openai_api_key):
        """验证系统能处理多个连续调用"""
        del fake_openai_api_key
        # 你需要做什么：
        # 测试系统在多次调用下的性能和内存

        pipeline = RAGPipeline(retriever=mock_retriever)

        for i in range(5):
            mock_response = f"Response {i}"

            with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
                result = pipeline.run(f"Query {i}")

            assert isinstance(result, dict)

    @pytest.mark.integration
    def test_retrieve_response_size(self, mock_retriever, fake_openai_api_key):
        """验证检索结果的合理大小"""
        del fake_openai_api_key
        # 你需要做什么：
        # 确保检索不会返回过多文档

        pipeline = RAGPipeline(retriever=mock_retriever)

        # mock_retriever 应该始终返回合理数量的文档
        assert len(mock_retriever.invoke("test")) <= 10
