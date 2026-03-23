"""Shared test fixtures and utilities"""

import os
import pytest
from typing import Any, List
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


# ============ Mock Retriever（不依赖真实数据库） ============


class MockRetriever(BaseRetriever):
    """模拟检索器，返回固定文档，用于单元测试"""

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """返回固定的测试文档"""
        docs = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
                metadata={"source": "test_doc_1", "_score": 0.95},
            ),
            Document(
                page_content="Deep learning uses neural networks with multiple layers to process complex patterns in data.",
                metadata={"source": "test_doc_2", "_score": 0.87},
            ),
            Document(
                page_content="Natural language processing helps computers understand, interpret, and generate human language.",
                metadata={"source": "test_doc_3", "_score": 0.79},
            ),
        ]
        return docs


class _DummyVectorStore:
    """Lightweight vectorstore stub for unit tests."""

    def similarity_search_with_score(self, query: str, k: int):
        del query
        return [
            (
                Document(
                    page_content=f"Stub semantic doc {i}", metadata={"source": "stub"}
                ),
                1.0 - (i * 0.01),
            )
            for i in range(k)
        ]


class _DummyReranker:
    """Lightweight reranker stub for unit tests."""

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        del query
        return docs[:top_k]


class _DummyBM25:
    """Lightweight BM25 stub for unit tests."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        del query
        k = top_k or 5
        return [
            Document(page_content=f"Stub bm25 doc {i}", metadata={"source": "stub"})
            for i in range(k)
        ]


class _DummySemantic:
    """Plain semantic retriever stub for unit tests."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        del query
        k = top_k or 5
        return [
            Document(page_content=f"Stub semantic doc {i}", metadata={"source": "stub"})
            for i in range(k)
        ]


# ============ Fixtures ============


@pytest.fixture
def mock_retriever():
    """提供 mock retriever 给所有测试"""
    return MockRetriever()


@pytest.fixture(autouse=True)
def _set_default_openai_api_key(request, monkeypatch):
    """Ensure unit tests do not fail when OPENAI_API_KEY is missing."""
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "test-key"))


@pytest.fixture(autouse=True)
def _patch_heavy_retriever_init_for_unit_tests(request, monkeypatch):
    """Skip model downloads and DB setup in non-integration tests."""
    if request.node.get_closest_marker("integration"):
        return

    from src.retriever.semantic import SemanticRetriever
    from src.retriever.hybrid import HybridRetriever

    def _semantic_model_post_init(self, context: Any) -> None:
        self.vectorstore = _DummyVectorStore()
        self.reranker = _DummyReranker()
        super(SemanticRetriever, self).model_post_init(context)

    def _hybrid_model_post_init(self, context: Any) -> None:
        self.semantic = _DummySemantic()
        self.bm25 = _DummyBM25()
        self.reranker = _DummyReranker()
        super(HybridRetriever, self).model_post_init(context)

    monkeypatch.setattr(SemanticRetriever, "model_post_init", _semantic_model_post_init)
    monkeypatch.setattr(HybridRetriever, "model_post_init", _hybrid_model_post_init)


@pytest.fixture
def sample_documents():
    """提供示例文档"""
    return [
        Document(
            page_content="Artificial intelligence is transforming industries across the world.",
            metadata={"source": "doc1", "relevance": 0.9},
        ),
        Document(
            page_content="Machine learning requires large amounts of quality data.",
            metadata={"source": "doc2", "relevance": 0.8},
        ),
        Document(
            page_content="Neural networks are inspired by biological neurons.",
            metadata={"source": "doc3", "relevance": 0.75},
        ),
    ]


@pytest.fixture
def sample_query():
    """提供示例查询"""
    return "What is artificial intelligence?"


@pytest.fixture
def openai_api_key():
    """
    提供 API key，如果不存在则跳过需要它的测试

    使用方法：
        def test_something(openai_api_key):
            # 如果 OPENAI_API_KEY 未设置，此测试会被跳过
            ...
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set in environment")
    return key


@pytest.fixture
def mock_llm_response():
    """提供模拟的 LLM 响应"""
    response = Mock()
    response.content = (
        "Artificial intelligence (AI) is the simulation of human intelligence processes by computer systems. "
        "It includes learning, reasoning, and self-correction. AI is transforming industries including healthcare, "
        "finance, transportation, and more."
    )
    return response


@pytest.fixture
def mock_streaming_tokens():
    """提供模拟的流式 token"""
    return [
        "Artificial",
        " intelligence",
        " is",
        " the",
        " simulation",
        " of",
        " human",
        " intelligence",
        " processes",
        " by",
        " computer",
        " systems",
        ".",
    ]


# ============ 辅助函数 ============


def assert_document_list_structure(documents, expected_min_length=1):
    """
    验证文档列表的结构

    Args:
        documents: 要验证的文档列表
        expected_min_length: 期望的最小长度
    """
    assert isinstance(documents, list), "Documents should be a list"
    assert (
        len(documents) >= expected_min_length
    ), f"Expected at least {expected_min_length} documents"

    for doc in documents:
        assert isinstance(
            doc, Document
        ), f"Each item should be a Document, got {type(doc)}"
        assert hasattr(doc, "page_content"), "Document should have page_content"
        assert hasattr(doc, "metadata"), "Document should have metadata"
        assert isinstance(doc.page_content, str), "page_content should be string"
        assert isinstance(doc.metadata, dict), "metadata should be dict"


def assert_pipeline_output_structure(output):
    """
    验证 RAG Pipeline 输出的结构

    Args:
        output: RAGPipeline.run() 的返回值
    """
    assert isinstance(output, dict), "Output should be a dictionary"

    required_keys = {"question", "docs", "context", "answer"}
    assert required_keys.issubset(
        output.keys()
    ), f"Missing required keys. Got {output.keys()}"

    assert isinstance(output["question"], str), "question should be string"
    assert isinstance(output["docs"], list), "docs should be list"
    assert isinstance(output["context"], str), "context should be string"
    assert isinstance(output["answer"], str), "answer should be string"

    if output["docs"]:
        assert_document_list_structure(output["docs"], expected_min_length=1)
