"""
测试 RAG Pipeline 的功能

包含：
1. format_docs 函数的测试
2. RAGPipeline 初始化的测试
3. RAGPipeline.run() 方法的测试
4. RAGPipeline.stream() 方法的测试
"""

import pytest
from unittest.mock import patch
from langchain_core.documents import Document

from src.pipeline.rag import ChatOpenAI, RAGPipeline, format_docs
from tests.conftest import MockRetriever


class TestFormatDocs:
    """测试 format_docs 辅助函数"""

    def test_format_docs_joins_documents(self, sample_documents):
        """验证 format_docs 能正确拼接文档"""
        result = format_docs(sample_documents)

        # 应该包含所有文档内容
        assert "Artificial intelligence is transforming industries" in result
        assert "Machine learning requires large amounts" in result
        assert "Neural networks are inspired by" in result

        # 应该用 --- 分隔
        assert "\n\n---\n\n" in result

    def test_format_docs_empty_list(self):
        """验证 format_docs 能处理空列表"""
        result = format_docs([])
        assert result == ""

    def test_format_docs_single_document(self):
        """验证 format_docs 能处理单个文档"""
        docs = [Document(page_content="Test content")]
        result = format_docs(docs)
        assert result == "Test content"

    def test_format_docs_preserves_page_content(self):
        """验证 format_docs 保留原始内容"""
        docs = [
            Document(page_content="First doc"),
            Document(page_content="Second doc"),
        ]
        result = format_docs(docs)
        assert result == "First doc\n\n---\n\nSecond doc"

    def test_format_docs_ignores_metadata(self):
        """验证 format_docs 只关注 page_content，忽略 metadata"""
        docs = [
            Document(
                page_content="Content 1", metadata={"source": "doc1", "score": 0.9}
            ),
            Document(
                page_content="Content 2", metadata={"source": "doc2", "score": 0.8}
            ),
        ]
        result = format_docs(docs)

        # metadata 不应该出现在结果中
        assert "source" not in result
        assert "doc1" not in result
        assert "doc2" not in result


class TestRAGPipelineInit:
    """测试 RAGPipeline 初始化"""

    @pytest.mark.unit
    def test_pipeline_initialization_with_mock_retriever(self, mock_retriever):
        """验证 pipeline 能正确初始化"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        assert pipeline.retriever is mock_retriever
        assert pipeline.llm is not None
        assert pipeline.chain is not None

    @pytest.mark.unit
    def test_pipeline_llm_configuration(self, mock_retriever):
        """验证 pipeline 的 LLM 配置"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        # LLM 应该有正确的配置
        assert pipeline.llm.temperature == 0.1
        assert pipeline.llm.model_name is not None

    @pytest.mark.unit
    def test_pipeline_chain_structure(self, mock_retriever):
        """验证 chain 的结构正确"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        # chain 应该存在且是 runnable
        assert hasattr(pipeline.chain, "invoke")
        assert hasattr(pipeline.chain, "stream")


class TestRAGPipelineRun:
    """测试 RAGPipeline.run() 方法"""

    @pytest.mark.unit
    def test_run_returns_dict_with_expected_keys(self, mock_retriever):
        """验证 run() 返回包含正确键的字典"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        # Mock LLM 的响应
        mock_response = "This is a test answer about AI."

        with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
            result = pipeline.run("What is AI?")

        # 验证返回的是字典
        assert isinstance(result, dict)

        # 验证有所有必需的键
        assert "question" in result
        assert "docs" in result
        assert "context" in result
        assert "answer" in result

    @pytest.mark.unit
    def test_run_question_field_correct(self, mock_retriever):
        """验证 run() 返回的问题字段正确"""
        pipeline = RAGPipeline(retriever=mock_retriever)
        question = "What is machine learning?"

        mock_response = "Machine learning is a subset of AI..."

        with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
            result = pipeline.run(question)

        assert result["question"] == question

    @pytest.mark.unit
    def test_run_docs_is_list_of_documents(self, mock_retriever):
        """验证 run() 返回的 docs 是文档列表"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        mock_response = "Test answer"

        with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
            result = pipeline.run("What is AI?")

        # docs 应该是列表
        assert isinstance(result["docs"], list)
        # 文档列表不应该为空（mock_retriever 返回 3 个文档）
        assert len(result["docs"]) > 0
        # 每个元素应该是 Document
        assert all(isinstance(doc, Document) for doc in result["docs"])

    @pytest.mark.unit
    def test_run_context_is_formatted_string(self, mock_retriever):
        """验证 run() 返回的 context 是格式化的字符串"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        mock_response = "Test answer"

        with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
            result = pipeline.run("What is AI?")

        # context 应该是字符串
        assert isinstance(result["context"], str)
        # context 不应该为空
        assert len(result["context"]) > 0
        # context 应该包含文档内容（由 format_docs 格式化）
        assert "Machine learning" in result["context"]
        assert "Deep learning" in result["context"]

    @pytest.mark.unit
    def test_run_answer_is_llm_response(self, mock_retriever):
        """验证 run() 返回的 answer 来自 LLM"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        expected_answer = "AI is the future of computing."
        mock_response = expected_answer

        with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
            result = pipeline.run("What is AI?")

        # answer 应该来自 LLM 响应
        assert result["answer"] == expected_answer

    @pytest.mark.unit
    def test_run_with_different_queries(self, mock_retriever):
        """验证 run() 能处理不同的查询"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are neural networks?",
        ]

        for query in queries:
            mock_response = "Test response"

            with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
                result = pipeline.run(query)

            # 每次调用都应该成功并返回正确结构
            assert result["question"] == query
            assert len(result["docs"]) > 0


class TestRAGPipelineStream:
    """测试 RAGPipeline.stream() 方法"""

    @pytest.mark.unit
    def test_stream_yields_tokens(self, mock_retriever, mock_streaming_tokens):
        """验证 stream() 能产生 token"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        with patch.object(
            ChatOpenAI, "stream", return_value=iter(mock_streaming_tokens)
        ):
            tokens = list(pipeline.stream("What is AI?"))

        # 应该产生 token
        assert len(tokens) > 0
        # 所有 token 应该是字符串
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.unit
    def test_stream_returns_generator(self, mock_retriever, mock_streaming_tokens):
        """验证 stream() 返回生成器"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        with patch.object(
            ChatOpenAI, "stream", return_value=iter(mock_streaming_tokens)
        ):
            result = pipeline.stream("What is AI?")

        # 结果应该是生成器
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    @pytest.mark.unit
    def test_stream_retrieves_documents_first(
        self, mock_retriever, mock_streaming_tokens
    ):
        """验证 stream() 先检索文档再流式生成"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        # 跟踪 retriever 是否被调用
        with patch.object(
            MockRetriever, "invoke", wraps=pipeline.retriever.invoke
        ) as mock_invoke:
            with patch.object(
                ChatOpenAI, "stream", return_value=iter(mock_streaming_tokens)
            ):
                # 消费生成器
                list(pipeline.stream("What is AI?"))

            # retriever 应该被调用
            mock_invoke.assert_called_once()

    @pytest.mark.unit
    def test_stream_partial_consumption(self, mock_retriever, mock_streaming_tokens):
        """验证 stream() 支持部分消费（取前 N 个 token）"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        with patch.object(
            ChatOpenAI, "stream", return_value=iter(mock_streaming_tokens)
        ):
            stream_gen = pipeline.stream("What is AI?")

            # 只取前 5 个 token
            partial_tokens = [next(stream_gen) for _ in range(5)]

        assert len(partial_tokens) == 5
        assert all(isinstance(t, str) for t in partial_tokens)

    @pytest.mark.unit
    def test_stream_empty_response(self, mock_retriever):
        """验证 stream() 能处理空响应"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        with patch.object(ChatOpenAI, "stream", return_value=iter([])):
            tokens = list(pipeline.stream("What is AI?"))

        assert tokens == []


class TestRAGPipelineIntegration:
    """RAG Pipeline 的集成场景"""

    @pytest.mark.unit
    def test_run_and_stream_use_same_retriever(self, mock_retriever):
        """验证 run() 和 stream() 使用同一个 retriever"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        # 两个方法都应该使用同一个 retriever
        assert pipeline.retriever is mock_retriever

    @pytest.mark.unit
    def test_run_and_stream_have_same_data_source(self, mock_retriever):
        """验证 run() 和 stream() 的数据源一致"""
        pipeline = RAGPipeline(retriever=mock_retriever)

        mock_response = "Answer"
        mock_tokens = ["Answer"]

        with patch.object(ChatOpenAI, "invoke", return_value=mock_response):
            run_result = pipeline.run("Test query")

        with patch.object(ChatOpenAI, "stream", return_value=iter(mock_tokens)):
            stream_tokens = list(pipeline.stream("Test query"))

        # 两者都应该有相同数量的文档（都来自同一个 retriever）
        assert len(run_result["docs"]) == len(mock_retriever.invoke("Test query"))
