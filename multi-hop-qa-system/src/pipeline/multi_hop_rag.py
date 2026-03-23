import json
import logging
import os
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dataclasses import dataclass
from src.config import LLM_MODEL

logger = logging.getLogger(__name__)


# 数据类
@dataclass
class HopResult:
    sub_question: str
    docs: list[Document]


@dataclass
class MultiHopResult:
    question: str
    sub_questions: list[str]
    hop_results: list[HopResult]
    merged_docs: list[Document]
    answer: str


def _doc_score_value(doc: Document) -> float:
    """返回文档的数值分数，用于 merge_docs 去重比较。内部使用，不导出。"""
    meta = doc.metadata
    if "_rerank_score" in meta:
        return float(meta["_rerank_score"])
    if "_rrf_score" in meta:
        return float(meta["_rrf_score"])
    if "_score" in meta:
        return float(meta["_score"])
    return 0.0


def merge_docs(hop_docs: list[list[Document]]) -> list[Document]:
    """合并多跳检索结果，按 url → title 去重，保留分数最高的副本，按分数降序返回。"""
    seen: dict[str, Document] = {}
    for hop in hop_docs:
        for doc in hop:
            key = (
                doc.metadata.get("url")
                or doc.metadata.get("title")
                or doc.page_content[:60]
            )
            if key not in seen or _doc_score_value(doc) > _doc_score_value(seen[key]):
                seen[key] = doc
    return sorted(seen.values(), key=_doc_score_value, reverse=True)


_DECOMPOSE_PROMPT = (
    "Break the following complex question into 2-3 independent sub-questions "
    "that together cover all the information needed to answer it. "
    'Return ONLY a JSON array of strings, e.g. ["sub_q1", "sub_q2"]. '
    "No extra text.\n\nQuestion: {question}"
)


def decompose_query(query: str, llm) -> list[str]:
    """LLM 将复杂问题分解为子问题列表。解析失败时降级返回 [query]。"""
    prompt = _DECOMPOSE_PROMPT.format(question=query)
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        parsed = json.loads(text.strip())
        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
            return parsed
        logger.warning("decompose_query: LLM returned non-list JSON, falling back")
    except Exception as exc:
        logger.warning(
            "decompose_query: parse error (%s), falling back to original query", exc
        )
    return [query]


_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            'If the answer cannot be found in the context, respond with "I don\'t know."',
        ),
        (
            "human",
            "Sub-questions used for retrieval:\n{sub_questions}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\nAnswer:",
        ),
    ]
)


def _format_sub_questions(sub_questions: list[str]) -> str:
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class MultiHopRAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL):
        self.retriever = retriever
        self._decompose_llm = ChatOpenAI(
            model=model, temperature=0.0, api_key=os.environ["OPENAI_API_KEY"]
        )
        self._gen_llm = ChatOpenAI(
            model=model, temperature=0.1, api_key=os.environ["OPENAI_API_KEY"]
        )

    def retrieve(self, query: str) -> MultiHopResult:
        """分解问题 → 各子问题独立检索 → 合并去重。不调用生成 LLM。"""
        sub_questions = decompose_query(query, self._decompose_llm)
        hop_results = []
        for sub_q in sub_questions:
            docs = self.retriever.invoke(sub_q)
            hop_results.append(HopResult(sub_question=sub_q, docs=docs))
        merged = merge_docs([hr.docs for hr in hop_results])
        return MultiHopResult(
            question=query,
            sub_questions=sub_questions,
            hop_results=hop_results,
            merged_docs=merged,
            answer="",
        )

    def stream(self, result: MultiHopResult) -> Iterator[str]:
        """流式输出最终答案。prompt 中包含子问题以提供推理上下文。"""
        yield from (_ANSWER_PROMPT | self._gen_llm | StrOutputParser()).stream(
            {
                "question": result.question,
                "sub_questions": _format_sub_questions(result.sub_questions),
                "context": _format_docs(result.merged_docs),
            }
        )
