import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.pipeline.multi_hop_rag import HopResult, MultiHopResult, _doc_score_value
from tests.conftest import MockRetriever


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
