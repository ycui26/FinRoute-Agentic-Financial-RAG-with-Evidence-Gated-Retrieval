from __future__ import annotations

import copy
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from finroute.config import FinRouteConfig
from finroute.grading import EvidenceGrader
from finroute.planning import RulePlanner
from finroute.reranking import QueryCenteredReranker
from finroute.retrieval import PageRetriever
from finroute.routing import DocumentRouter
from finroute.schemas import EvidenceGrade, QueryPlan, RetrievalResult


class AgentState(TypedDict, total=False):
    question: str
    plan: QueryPlan
    allowed_documents: list[str]
    candidates: pd.DataFrame
    reranked: pd.DataFrame
    selected: pd.DataFrame
    grade: EvidenceGrade
    retry_count: int
    retry_queries: list[str]
    abstain: bool
    stop_reason: str
    trace: list[dict[str, Any]]


def trace(state: AgentState, node: str, **details: Any) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, **details}]


class FinRouteWorkflow:
    def __init__(
        self,
        planner: RulePlanner,
        router: DocumentRouter,
        retriever: PageRetriever,
        reranker: QueryCenteredReranker,
        grader: EvidenceGrader,
        config: FinRouteConfig,
    ):
        self.planner = planner
        self.router = router
        self.retriever = retriever
        self.reranker = reranker
        self.grader = grader
        self.config = config
        self.graph = self._compile()

    def _plan(self, state: AgentState) -> dict:
        plan = self.planner.plan(state["question"])
        return {
            "plan": plan,
            "retry_count": 0,
            "trace": trace(
                state,
                "plan",
                route=plan.route,
                confidence=plan.route_confidence,
                reason=plan.reason,
                operands=plan.operands,
            ),
        }

    def _route_documents(self, state: AgentState) -> dict:
        frame = self.router.route(
            state["question"], self.config.retrieval.routed_document_top_k
        )
        documents = frame["doc_name"].astype(str).tolist()
        return {
            "allowed_documents": documents,
            "trace": trace(state, "document_route", documents=documents),
        }

    def _retrieve(self, state: AgentState) -> dict:
        candidates = self.retriever.retrieve(
            state["question"], state["plan"], state["allowed_documents"]
        )
        return {
            "candidates": candidates,
            "trace": trace(
                state,
                "retrieve",
                route=state["plan"].route,
                candidate_pages=len(candidates),
            ),
        }

    def _rerank(self, state: AgentState) -> dict:
        reranked, selected = self.reranker.rerank(
            state["question"], state["plan"], state["candidates"]
        )
        return {
            "reranked": reranked,
            "selected": selected,
            "trace": trace(state, "rerank", selected_pages=len(selected)),
        }

    def _grade(self, state: AgentState) -> dict:
        grade = self.grader.grade(
            state["question"], state["plan"], state["candidates"], state["selected"]
        )
        return {
            "grade": grade,
            "trace": trace(
                state,
                "grade",
                complete=grade.complete,
                reason=grade.reason,
                retry_round=state.get("retry_count", 0),
            ),
        }

    def _after_grade(self, state: AgentState) -> str:
        if state["grade"].complete:
            return "finish"
        if state.get("retry_count", 0) < self.config.grading.maximum_retries:
            return "retry"
        return "refuse"

    def _retry(self, state: AgentState) -> dict:
        queries = self.grader.retry_queries(state["question"], state["plan"], state["grade"])
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "retry_queries": queries,
            "trace": trace(
                state,
                "retry",
                observation=state["grade"].to_dict(),
                queries=queries,
            ),
        }

    def _retry_retrieve(self, state: AgentState) -> dict:
        retry_plan = copy.copy(state["plan"])
        retry_plan.route = "calculation" if retry_plan.operands else "mixed"
        retry = self.retriever.retrieve(
            state["question"],
            retry_plan,
            state["allowed_documents"],
            extra_queries=state["retry_queries"],
        )
        candidates = self.retriever.merge(state["candidates"], retry)
        return {
            "candidates": candidates,
            "trace": trace(
                state,
                "retry_retrieve",
                new_candidate_pages=len(retry),
                combined_candidate_pages=len(candidates),
            ),
        }

    def _finish(self, state: AgentState) -> dict:
        return {
            "abstain": False,
            "stop_reason": "evidence_complete",
            "trace": trace(state, "finish"),
        }

    def _refuse(self, state: AgentState) -> dict:
        return {
            "abstain": True,
            "stop_reason": "insufficient_evidence_after_retry",
            "trace": trace(state, "refuse", reason=state["grade"].reason),
        }

    def _compile(self):
        graph = StateGraph(AgentState)
        for name, node in {
            "plan": self._plan,
            "document_route": self._route_documents,
            "retrieve": self._retrieve,
            "rerank": self._rerank,
            "grade": self._grade,
            "retry": self._retry,
            "retry_retrieve": self._retry_retrieve,
            "finish": self._finish,
            "refuse": self._refuse,
        }.items():
            graph.add_node(name, node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "document_route")
        graph.add_edge("document_route", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "grade")
        graph.add_conditional_edges(
            "grade",
            self._after_grade,
            {"finish": "finish", "retry": "retry", "refuse": "refuse"},
        )
        graph.add_edge("retry", "retry_retrieve")
        graph.add_edge("retry_retrieve", "rerank")
        graph.add_edge("finish", END)
        graph.add_edge("refuse", END)
        return graph.compile()

    def run(self, question: str) -> RetrievalResult:
        final = self.graph.invoke({"question": question, "trace": []})
        selected = final.get("selected", pd.DataFrame())
        page_columns = [
            "rank", "doc_name", "page_idx0", "page_num", "section",
            "combined_score", "rerank_excerpt", "candidate_sources",
        ]
        pages = selected[[column for column in page_columns if column in selected.columns]].to_dict(
            orient="records"
        )
        return RetrievalResult(
            question=question,
            plan=final["plan"],
            allowed_documents=final.get("allowed_documents", []),
            pages=pages,
            grade=final["grade"],
            abstain=final.get("abstain", True),
            stop_reason=final.get("stop_reason", "unknown"),
            trace=final.get("trace", []),
            candidate_count=len(final.get("candidates", [])),
        )
