import os
import json
import logging
from typing import TypedDict, Iterator
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.config import LLM_MODEL
from src.pipeline.multi_hop_rag import _ANSWER_PROMPT, _format_docs, merge_docs

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 6000


class HopTrace(TypedDict):
    query: str
    docs: list[Document]
    reasoning: str


class RAGState(TypedDict):
    question: str
    current_query: str
    retrieved_docs: list[Document]
    hop_count: int
    sufficient: bool
    follow_up_query: str
    trace: list[HopTrace]
    answer: str


def _build_judge_context(docs: list[Document]) -> str:
    text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return text[:_MAX_CONTEXT_CHARS]


_JUDGE_PROMPT = (
    "You are evaluating whether the provided context is sufficient to answer the question.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Return ONLY a JSON object with this exact structure:\n"
    '{{"sufficient": true or false, '
    '"follow_up_query": "specific question to retrieve missing information, or empty string if sufficient", '
    '"reasoning": "brief explanation of what information is present or missing"}}'
)


class AgentRAGPipeline:
    def __init__(self, retriever, model: str = LLM_MODEL) -> None:
        self.retriever = retriever
        self._judge_llm = ChatOpenAI(
            model=model, temperature=0.0, api_key=os.environ["OPENAI_API_KEY"]
        )
        self._gen_llm = ChatOpenAI(
            model=model, temperature=0.1, api_key=os.environ["OPENAI_API_KEY"]
        )
        self._graph = self._build_graph()

    def _retrieve_node(self, state: RAGState) -> dict:
        query = (
            state["follow_up_query"]
            if state["follow_up_query"]
            else state["current_query"]
        )
        docs = self.retriever.invoke(query)
        merged = merge_docs([state["retrieved_docs"], docs])
        trace = list(state["trace"]) + [HopTrace(query=query, docs=docs, reasoning="")]
        return {
            "current_query": query,
            "retrieved_docs": merged,
            "hop_count": state["hop_count"] + 1,
            "trace": trace,
        }

    def _judge_node(self, state: RAGState) -> dict:
        context = _build_judge_context(state["retrieved_docs"])
        prompt = _JUDGE_PROMPT.format(question=state["question"], context=context)
        sufficient = True
        follow_up_query = ""
        reasoning = ""
        try:
            response = self._judge_llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text.strip())
            sufficient = bool(parsed.get("sufficient", True))
            follow_up_query = str(parsed.get("follow_up_query", ""))
            reasoning = str(parsed.get("reasoning", ""))
        except Exception as exc:
            logger.warning(
                "judge_node: parse error (%s), falling back to sufficient=True", exc
            )

        trace = list(state["trace"])
        if trace:
            last = trace[-1]
            trace[-1] = HopTrace(
                query=last["query"], docs=last["docs"], reasoning=reasoning
            )

        return {
            "sufficient": sufficient,
            "follow_up_query": follow_up_query,
            "trace": trace,
        }

    def _generate_node(self, state: RAGState) -> dict:
        sub_questions_str = "\n".join(
            f"{i+1}. {t['query']}" for i, t in enumerate(state["trace"])
        )
        context = _format_docs(state["retrieved_docs"])
        chain = _ANSWER_PROMPT | self._gen_llm | StrOutputParser()
        answer = chain.invoke(
            {
                "question": state["question"],
                "sub_questions": sub_questions_str,
                "context": context,
            }
        )
        return {"answer": answer}

    @staticmethod
    def _should_continue(state: RAGState) -> str:
        if state["sufficient"] or state["hop_count"] >= 3:
            return "generate"
        return "retrieve"

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", lambda s: self._retrieve_node(s))
        graph.add_node("judge", lambda s: self._judge_node(s))
        graph.add_node("generate", lambda s: self._generate_node(s))
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "judge")
        graph.add_conditional_edges(
            "judge",
            self._should_continue,
            {"retrieve": "retrieve", "generate": "generate"},
        )
        graph.add_edge("generate", END)
        return graph.compile()

    def run(self, question: str) -> RAGState:
        initial: RAGState = {
            "question": question,
            "current_query": question,
            "retrieved_docs": [],
            "hop_count": 0,
            "sufficient": False,
            "follow_up_query": "",
            "trace": [],
            "answer": "",
        }
        return self._graph.invoke(initial)

    def stream_answer(self, state: RAGState) -> Iterator[str]:
        sub_questions_str = "\n".join(
            f"{i+1}. {t['query']}" for i, t in enumerate(state["trace"])
        )
        context = _format_docs(state["retrieved_docs"])
        chain = _ANSWER_PROMPT | self._gen_llm | StrOutputParser()
        yield from chain.stream(
            {
                "question": state["question"],
                "sub_questions": sub_questions_str,
                "context": context,
            }
        )
