import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.pipeline.multi_hop_rag import HopResult, MultiHopResult, _doc_score_value
from tests.conftest import MockRetriever
from src.pipeline.multi_hop_rag import merge_docs
from src.pipeline.multi_hop_rag import decompose_query
from src.pipeline.multi_hop_rag import MultiHopRAGPipeline


class TestDataClasses:
    def test_hop_result_fields(self):
        doc = Document(page_content="test")
        hr = HopResult(sub_question="q", docs=[doc])
        assert hr.sub_question == "q"
        assert hr.docs == [doc]

    def test_multi_hop_result_fields(self):
        hr = HopResult(sub_question="q", docs=[])
        mhr = MultiHopResult(
            question="orig",
            sub_questions=["q"],
            hop_results=[hr],
            merged_docs=[],
            answer="",
        )
        assert mhr.question == "orig"
        assert mhr.answer == ""

    def test_multi_hop_result_is_mutable(self):
        mhr = MultiHopResult(
            question="q", sub_questions=[], hop_results=[], merged_docs=[], answer=""
        )
        mhr.answer = "patched"
        assert mhr.answer == "patched"


class TestDocScoreValue:
    def test_prefers_rerank_score(self):
        doc = Document(page_content="x", metadata={"_rerank_score": 0.9, "_score": 0.5})
        assert _doc_score_value(doc) == 0.9

    def test_falls_back_to_rrf_score(self):
        doc = Document(page_content="x", metadata={"_rrf_score": 0.7, "_score": 0.5})
        assert _doc_score_value(doc) == 0.7

    def test_falls_back_to_score(self):
        doc = Document(page_content="x", metadata={"_score": 0.6})
        assert _doc_score_value(doc) == 0.6

    def test_returns_zero_when_no_score(self):
        doc = Document(page_content="x", metadata={})
        assert _doc_score_value(doc) == 0.0


class TestMergeDocs:
    def test_deduplicates_by_url(self):
        doc_a = Document(
            page_content="v1", metadata={"url": "http://x.com", "_score": 0.9}
        )
        doc_b = Document(
            page_content="v2", metadata={"url": "http://x.com", "_score": 0.5}
        )
        result = merge_docs([[doc_a], [doc_b]])
        assert len(result) == 1
        assert result[0].page_content == "v1"  # 保留分数高的

    def test_deduplicates_by_title_when_no_url(self):
        doc_a = Document(page_content="v1", metadata={"title": "Doc A", "_score": 0.4})
        doc_b = Document(page_content="v2", metadata={"title": "Doc A", "_score": 0.8})
        result = merge_docs([[doc_a], [doc_b]])
        assert len(result) == 1
        assert result[0].page_content == "v2"  # 保留分数高的

    def test_keeps_unique_docs_from_all_hops(self):
        doc1 = Document(page_content="d1", metadata={"url": "http://a.com"})
        doc2 = Document(page_content="d2", metadata={"url": "http://b.com"})
        doc3 = Document(page_content="d3", metadata={"url": "http://c.com"})
        result = merge_docs([[doc1, doc2], [doc2, doc3]])
        assert len(result) == 3

    def test_empty_hops_excluded(self):
        doc1 = Document(page_content="d1", metadata={"url": "http://a.com"})
        result = merge_docs([[doc1], [], []])
        assert len(result) == 1

    def test_all_empty_returns_empty(self):
        assert merge_docs([[], []]) == []

    def test_result_ordered_by_descending_score(self):
        doc_low = Document(
            page_content="low", metadata={"url": "http://a.com", "_score": 0.3}
        )
        doc_high = Document(
            page_content="high", metadata={"url": "http://b.com", "_score": 0.9}
        )
        result = merge_docs([[doc_low, doc_high]])
        assert result[0].page_content == "high"


class TestDecomposeQuery:
    def _make_llm(self, response_text: str):
        llm = MagicMock()
        llm.invoke.return_value.content = response_text
        return llm

    def test_returns_list_of_strings(self):
        llm = self._make_llm('["What is A?", "What is B?"]')
        result = decompose_query("Complex question about A and B", llm)
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    def test_parses_two_sub_questions(self):
        llm = self._make_llm('["Sub Q1", "Sub Q2"]')
        result = decompose_query("question", llm)
        assert result == ["Sub Q1", "Sub Q2"]

    def test_fallback_on_invalid_json(self):
        llm = self._make_llm("I cannot decompose this")
        result = decompose_query("original question", llm)
        assert result == ["original question"]

    def test_fallback_on_non_list_json(self):
        llm = self._make_llm('{"key": "value"}')
        result = decompose_query("original question", llm)
        assert result == ["original question"]

    def test_llm_is_called_once(self):
        llm = self._make_llm('["Q1", "Q2"]')
        decompose_query("test", llm)
        assert llm.invoke.call_count == 1


class TestMultiHopRAGPipelineRetrieve:
    def test_retrieve_returns_multi_hop_result(self):
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch(
            "src.pipeline.multi_hop_rag.decompose_query", return_value=["Q1", "Q2"]
        ):
            result = pipeline.retrieve("complex question")
        assert isinstance(result, MultiHopResult)

    def test_retrieve_answer_is_empty_string(self):
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch("src.pipeline.multi_hop_rag.decompose_query", return_value=["Q1"]):
            result = pipeline.retrieve("question")
        assert result.answer == ""

    def test_retrieve_fills_sub_questions(self):
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch(
            "src.pipeline.multi_hop_rag.decompose_query", return_value=["Q1", "Q2"]
        ):
            result = pipeline.retrieve("question")
        assert result.sub_questions == ["Q1", "Q2"]

    def test_retrieve_hop_results_count_matches_sub_questions(self):
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch(
            "src.pipeline.multi_hop_rag.decompose_query",
            return_value=["Q1", "Q2", "Q3"],
        ):
            result = pipeline.retrieve("question")
        assert len(result.hop_results) == 3

    def test_retrieve_merged_docs_not_empty(self):
        # MockRetriever 每次返回 3 个不同 page_content 的文档（无 url/title），
        # merge_docs 用 page_content[:60] 作为 key，两个子问题各返回相同 3 个内容 → 去重后仍为 3
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        with patch(
            "src.pipeline.multi_hop_rag.decompose_query", return_value=["Q1", "Q2"]
        ):
            result = pipeline.retrieve("question")
        assert len(result.merged_docs) > 0


class TestMultiHopRAGPipelineStream:
    def _make_mhr(self, docs=None):
        return MultiHopResult(
            question="q",
            sub_questions=["Q1"],
            hop_results=[HopResult("Q1", docs or [])],
            merged_docs=docs or [Document(page_content="ctx", metadata={})],
            answer="",
        )

    def test_stream_yields_strings(self):
        """stream() 通过 LCEL chain 流式输出，这里直接 patch pipeline.stream。"""
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        mhr = self._make_mhr()
        tokens = ["Hello", " world"]

        with patch.object(pipeline, "stream", return_value=iter(tokens)):
            result = list(pipeline.stream(mhr))
        assert result == tokens

    def test_stream_is_generator(self):
        import types
        pipeline = MultiHopRAGPipeline(retriever=MockRetriever())
        mhr = self._make_mhr()
        # 用 MagicMock 替换 _gen_llm，让 LCEL chain 不真正调用 OpenAI
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter([])
        pipeline._gen_llm = mock_llm
        gen = pipeline.stream(mhr)
        assert isinstance(gen, types.GeneratorType)
