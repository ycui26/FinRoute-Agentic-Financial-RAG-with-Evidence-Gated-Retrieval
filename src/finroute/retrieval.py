from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finroute.config import FinRouteConfig
from finroute.finance_terms import aliases_for
from finroute.indexing import HybridIndex
from finroute.schemas import QueryPlan


def deduplicate_pages(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.drop_duplicates(["doc_name", "page_idx0"]).head(top_k).copy()
    result["path_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def frame_to_documents(frame: pd.DataFrame):
    from langchain_core.documents import Document

    documents = []
    for row in frame.to_dict(orient="records"):
        metadata = {
            key: value
            for key, value in row.items()
            if key != "context_text" and not isinstance(value, (dict, list, tuple, set))
        }
        documents.append(
            Document(page_content=str(row.get("context_text", row.get("text", ""))), metadata=metadata)
        )
    return documents


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    allowed_docs: tuple[str, ...] = ()
    top_n: int = 120


class LangChainRetrieverAdapter:
    """Expose the hybrid index through LangChain's standard invoke interface."""

    def __init__(self, index: HybridIndex):
        self.index = index

    def invoke(self, request: RetrievalRequest):
        frame = self.index.search(
            request.query,
            allowed_docs=list(request.allowed_docs) or None,
            top_n=request.top_n,
        )
        return frame_to_documents(frame)


class PageRetriever:
    def __init__(
        self,
        child_index: HybridIndex,
        table_index: HybridIndex,
        pages: pd.DataFrame,
        config: FinRouteConfig,
    ):
        self.child_index = child_index
        self.table_index = table_index
        self.pages = pages.set_index(["doc_name", "page_idx0"], drop=False)
        self.config = config
        self.langchain_child = LangChainRetrieverAdapter(child_index)
        self.langchain_table = LangChainRetrieverAdapter(table_index)

    def _annotate(
        self,
        frame: pd.DataFrame,
        source: str,
        query: str,
        plan: QueryPlan,
        allowed_docs: list[str],
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        result = frame.copy().reset_index(drop=True)
        result["path_rank"] = np.arange(1, len(result) + 1)
        result["candidate_source"] = source
        result["matched_query"] = query
        route_rank = {doc: rank for rank, doc in enumerate(allowed_docs, 1)}
        result["doc_route_rank"] = result["doc_name"].map(route_rank).fillna(len(route_rank) + 1)
        result["statement_match"] = [
            int(
                not plan.statement_types
                or str(row.get("statement_type", "")) in plan.statement_types
                or any(
                    statement.replace("_", " ")
                    in f"{row.get('section', '')} {row.get('context_text', '')}".lower()
                    for statement in plan.statement_types
                )
            )
            for row in result.to_dict(orient="records")
        ]
        # Narrative frames carry no table_quality_score column; default to 0.5
        # instead of letting the scalar fallback flow through pd.to_numeric.
        if "table_quality_score" in result.columns:
            result["table_quality_score"] = pd.to_numeric(
                result["table_quality_score"], errors="coerce"
            ).fillna(0.5)
        else:
            result["table_quality_score"] = 0.5
        structure = 1.0 + 0.15 * result["statement_match"] + 0.10 * result["table_quality_score"]
        result["fusion_contribution"] = (
            1.0 / (self.config.retrieval.rrf_k + result["path_rank"])
        ) * structure
        return result

    def _aggregate_children(self, children: pd.DataFrame, top_pages: int = 40) -> pd.DataFrame:
        if children.empty:
            return pd.DataFrame()
        rows = []
        for (_, _), group in children.groupby(["doc_name", "page_idx0"], sort=False):
            group = group.sort_values("rrf_score", ascending=False)
            row = group.iloc[0].to_dict()
            scores = group["rrf_score"].head(3).to_numpy()
            row["score"] = float(np.dot(scores, np.array([1.0, 0.5, 0.25])[: len(scores)]))
            rows.append(row)
        return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_pages)

    def _narrative_candidates(
        self, question: str, allowed_docs: list[str], plan: QueryPlan
    ) -> list[pd.DataFrame]:
        children = self.child_index.search(question, allowed_docs)
        pages = self._aggregate_children(children)
        return [self._annotate(pages, "narrative_child", question, plan, allowed_docs)]

    def _table_candidates(
        self, queries: list[str], allowed_docs: list[str], plan: QueryPlan
    ) -> list[pd.DataFrame]:
        frames = []
        rc = self.config.retrieval
        for query_index, query in enumerate(dict.fromkeys(queries)):
            tables = self.table_index.search(query, allowed_docs)
            frames.append(
                self._annotate(
                    deduplicate_pages(tables, rc.table_blocks_per_query),
                    f"table_q{query_index}",
                    query,
                    plan,
                    allowed_docs,
                )
            )
            if query_index == 0:
                children = self.child_index.search(query, allowed_docs)
                frames.append(
                    self._annotate(
                        deduplicate_pages(children, rc.child_pages_per_query),
                        "child_fallback",
                        query,
                        plan,
                        allowed_docs,
                    )
                )
        return frames

    def _repair_document_quota(
        self, selected: pd.DataFrame, pool: pd.DataFrame, allowed_docs: list[str]
    ) -> pd.DataFrame:
        minimum = self.config.retrieval.minimum_document_candidates
        selected = selected.copy()
        for doc in allowed_docs:
            available = pool[(pool["doc_name"] == doc) & ~pool.index.isin(selected.index)]
            target = min(minimum, int((pool["doc_name"] == doc).sum()))
            while int((selected["doc_name"] == doc).sum()) < target and not available.empty:
                counts = selected["doc_name"].value_counts()
                removable = selected[selected["doc_name"].map(counts) > minimum]
                if removable.empty:
                    break
                drop_index = removable.sort_values("candidate_score").index[0]
                add_index = available.sort_values("candidate_score", ascending=False).index[0]
                selected = pd.concat([selected.drop(drop_index), pool.loc[[add_index]]])
                available = available.drop(add_index)
        return selected

    def _fuse(
        self,
        frames: list[pd.DataFrame],
        allowed_docs: list[str],
        table_first: bool,
    ) -> pd.DataFrame:
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True, sort=False)
        rows = []
        for (doc_name, page_idx0), group in combined.groupby(["doc_name", "page_idx0"], sort=False):
            group = group.sort_values("fusion_contribution", ascending=False)
            representative = group.iloc[0].to_dict()
            evidence = list(dict.fromkeys(group["context_text"].astype(str)))[:5]
            page = self.pages.loc[(doc_name, int(page_idx0))]
            if isinstance(page, pd.DataFrame):
                page = page.iloc[0]
            sources = sorted(set(group["candidate_source"].astype(str)))
            query_support = int(group["matched_query"].nunique())
            fusion_score = float(group["fusion_contribution"].sum())
            representative.update(
                {
                    "doc_name": doc_name,
                    "page_idx0": int(page_idx0),
                    "page_num": int(page_idx0) + 1,
                    "section": str(page.get("section", "")),
                    "parent_text": str(page["text"]),
                    "evidence_bundle": "\n\n".join(evidence),
                    "candidate_sources": sources,
                    "matched_queries": list(dict.fromkeys(group["matched_query"])),
                    "query_support": query_support,
                    "fusion_score": fusion_score,
                    "statement_match": int(group["statement_match"].max()),
                    "table_quality_score": float(group["table_quality_score"].max()),
                    "doc_route_rank": int(group["doc_route_rank"].min()),
                    "has_table_source": any(source.startswith("table_") for source in sources),
                }
            )
            representative["candidate_score"] = (
                fusion_score + 0.002 * math.log1p(query_support) + 0.001 / max(1, representative["doc_route_rank"])
            )
            rows.append(representative)
        pool = pd.DataFrame(rows).sort_values("candidate_score", ascending=False)
        limit = self.config.retrieval.page_shortlist
        if table_first:
            table = pool[pool["has_table_source"]].head(
                self.config.retrieval.minimum_table_candidates
            )
            child = pool[~pool.index.isin(table.index)].head(
                self.config.retrieval.maximum_child_only_candidates
            )
            selected = pd.concat([table, child])
            selected = pd.concat([selected, pool[~pool.index.isin(selected.index)]])
            selected = selected[~selected.index.duplicated()].head(limit)
        else:
            selected = pool.head(limit)
        selected = self._repair_document_quota(selected, pool, allowed_docs)
        selected = selected.sort_values("candidate_score", ascending=False).head(limit).reset_index(drop=True)
        selected["candidate_rank"] = np.arange(1, len(selected) + 1)
        return selected

    def retrieve(
        self,
        question: str,
        plan: QueryPlan,
        allowed_docs: list[str],
        extra_queries: list[str] | None = None,
    ) -> pd.DataFrame:
        suffix = " ".join([*plan.years, *[item.replace("_", " ") for item in plan.statement_types]])
        operand_queries = [f"{' '.join(aliases_for(item))} {suffix}".strip() for item in plan.operands]
        queries = list(dict.fromkeys([*plan.subqueries, *operand_queries, *(extra_queries or [])]))
        frames: list[pd.DataFrame] = []
        if plan.route in {"narrative", "mixed"}:
            frames.extend(self._narrative_candidates(question, allowed_docs, plan))
        if plan.route in {"table_lookup", "calculation", "mixed"}:
            frames.extend(self._table_candidates(queries, allowed_docs, plan))
        return self._fuse(
            frames,
            allowed_docs,
            table_first=plan.route in {"table_lookup", "calculation"},
        )

    def merge(self, initial: pd.DataFrame, retry: pd.DataFrame) -> pd.DataFrame:
        if initial.empty:
            return retry.head(self.config.retrieval.page_shortlist)
        if retry.empty:
            return initial.head(self.config.retrieval.page_shortlist)
        merged = (
            pd.concat([initial, retry], ignore_index=True, sort=False)
            .sort_values(["candidate_score", "fusion_score"], ascending=False)
            .drop_duplicates(["doc_name", "page_idx0"])
            .head(self.config.retrieval.page_shortlist)
            .reset_index(drop=True)
        )
        merged["candidate_rank"] = np.arange(1, len(merged) + 1)
        return merged
