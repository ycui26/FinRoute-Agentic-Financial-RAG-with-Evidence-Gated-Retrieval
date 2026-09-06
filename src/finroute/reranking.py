from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from finroute.config import FinRouteConfig
from finroute.finance_terms import aliases_for
from finroute.schemas import QueryPlan


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float32")
    if not len(values):
        return values
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if not math.isfinite(low) or not math.isfinite(high) or high - low < 1e-8:
        return np.ones_like(values)
    return (values - low) / (high - low)


def contains_operand(text: str, operand: str) -> bool:
    lower = text.lower()
    return any(alias in lower for alias in aliases_for(operand))


def query_centered_excerpt(
    question: str,
    row: dict,
    plan: QueryPlan,
    max_chars: int = 2000,
    line_window: int = 1,
) -> str:
    text = str(row.get("parent_text", row.get("evidence_bundle", row.get("context_text", ""))))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:max_chars]
    query_tokens = set(TOKEN_RE.findall(question.lower()))
    anchors = [*plan.operands, *plan.required_terms, *plan.years]
    scores = []
    for index, line in enumerate(lines):
        lower = line.lower()
        overlap = len(query_tokens & set(TOKEN_RE.findall(lower)))
        anchor_hits = sum(str(anchor).lower() in lower for anchor in anchors)
        numeric = int(bool(re.search(r"\d", line)))
        scores.append((2.0 * anchor_hits + 0.2 * overlap + 0.25 * numeric, index))
    selected_indices: set[int] = set()
    for _, center in sorted(scores, reverse=True)[:8]:
        selected_indices.update(
            range(max(0, center - line_window), min(len(lines), center + line_window + 1))
        )
    excerpt = "\n".join(lines[index] for index in sorted(selected_indices))
    return excerpt[:max_chars]


class QueryCenteredReranker:
    def __init__(self, model_name: str, config: FinRouteConfig):
        self.model_name = model_name
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                trust_remote_code=True,
                local_files_only=self.config.models.local_files_only,
            )
        return self._model

    def rerank(
        self, question: str, plan: QueryPlan, candidates: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if candidates.empty:
            return candidates.copy(), candidates.copy()
        rc = self.config.reranking
        ranked = candidates.copy().reset_index(drop=True)
        ranked["rerank_excerpt"] = [
            query_centered_excerpt(
                question,
                row,
                plan,
                max_chars=rc.excerpt_max_chars,
                line_window=rc.excerpt_line_window,
            )
            for row in ranked.to_dict(orient="records")
        ]
        scores = np.asarray(
            self.model.predict(
                [(question, text) for text in ranked["rerank_excerpt"].astype(str)],
                batch_size=16,
                show_progress_bar=False,
            ),
            dtype="float32",
        ).reshape(-1)
        ranked["reranker_score"] = scores
        ranked["reranker_norm"] = minmax(scores)
        ranked["fusion_norm"] = minmax(ranked["candidate_score"].to_numpy())
        ranked["structure_score"] = (
            0.6 * pd.to_numeric(ranked.get("statement_match", 0), errors="coerce").fillna(0)
            + 0.4 * pd.to_numeric(ranked.get("table_quality_score", 0.5), errors="coerce").fillna(0.5)
        )
        ranked["combined_score"] = (
            rc.reranker_weight * ranked["reranker_norm"]
            + rc.fusion_weight * ranked["fusion_norm"]
            + rc.structure_weight * ranked["structure_score"]
        )
        ranked = ranked.sort_values("combined_score", ascending=False).reset_index(drop=True)
        ranked["covered_operands"] = [
            [operand for operand in plan.operands if contains_operand(text, operand)]
            for text in ranked["rerank_excerpt"].astype(str)
        ]
        ranked["covered_statement_types"] = [
            [item for item in plan.statement_types if item == str(row.get("statement_type", ""))]
            for row in ranked.to_dict(orient="records")
        ]
        ranked["covered_years"] = [
            [year for year in plan.years if year in text]
            for text in ranked["rerank_excerpt"].astype(str)
        ]
        selected = self._soft_select(ranked, plan, self.config.retrieval.final_top_k)
        return ranked, selected

    def _soft_select(self, ranked: pd.DataFrame, plan: QueryPlan, top_k: int) -> pd.DataFrame:
        remaining = set(ranked.index)
        selected: list[int] = []
        operands: set[str] = set()
        statements: set[str] = set()
        years: set[str] = set()
        rc = self.config.reranking
        while remaining and len(selected) < top_k:
            best_index, best_utility = None, -float("inf")
            for index in remaining:
                row = ranked.loc[index]
                utility = float(row["combined_score"])
                utility += rc.operand_gain_weight * len(set(row["covered_operands"]) - operands)
                utility += rc.statement_gain_weight * len(
                    set(row["covered_statement_types"]) - statements
                )
                utility += rc.year_gain_weight * len(set(row["covered_years"]) - years)
                if utility > best_utility:
                    best_index, best_utility = index, utility
            selected.append(int(best_index))
            remaining.remove(best_index)
            operands.update(ranked.loc[best_index, "covered_operands"])
            statements.update(ranked.loc[best_index, "covered_statement_types"])
            years.update(ranked.loc[best_index, "covered_years"])
        result = ranked.loc[selected].copy().reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        return result
