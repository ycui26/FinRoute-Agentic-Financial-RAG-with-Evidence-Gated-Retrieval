from __future__ import annotations

import re

import pandas as pd

from finroute.config import FinRouteConfig
from finroute.finance_terms import aliases_for
from finroute.schemas import EvidenceGrade, QueryPlan


VALUE_RE = re.compile(r"(?<!\w)(?:\$|€|£)?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")


def has_operand(text: str, operand: str) -> bool:
    lower = text.lower()
    return any(alias in lower for alias in aliases_for(operand))


def number_near_operand(text: str, operand: str, window: int) -> bool:
    lower = text.lower()
    for alias in aliases_for(operand):
        for match in re.finditer(re.escape(alias), lower):
            start, stop = max(0, match.start() - window), min(len(text), match.end() + window)
            local = text[start:stop]
            if VALUE_RE.search(local):
                return True
    return False


class EvidenceGrader:
    def __init__(self, config: FinRouteConfig):
        self.config = config

    def grade(
        self,
        question: str,
        plan: QueryPlan,
        candidates: pd.DataFrame,
        selected: pd.DataFrame,
    ) -> EvidenceGrade:
        text = "\n".join(
            selected.get("rerank_excerpt", selected.get("evidence_bundle", pd.Series(dtype=str)))
            .astype(str)
            .tolist()
        ).lower()
        covered = [operand for operand in plan.operands if has_operand(text, operand)]
        missing_operands = [operand for operand in plan.operands if operand not in covered]
        missing_numeric = [
            operand
            for operand in covered
            if plan.requires_numeric
            and not number_near_operand(text, operand, self.config.grading.numeric_anchor_window)
        ]
        matched_terms = [term for term in plan.required_terms if term.lower() in text]
        missing_terms = [term for term in plan.required_terms if term not in matched_terms]
        term_coverage = len(matched_terms) / len(plan.required_terms) if plan.required_terms else 1.0
        missing_years = [year for year in plan.years if year not in text]
        numeric_anchor_ok = bool(
            not plan.requires_numeric
            or not self.config.grading.require_numeric_anchor
            or (
                not missing_numeric
                if plan.operands
                else bool(VALUE_RE.search(text))
            )
        )
        enough_candidates = len(candidates) >= self.config.grading.minimum_candidate_pages
        complete = bool(
            enough_candidates
            and not missing_operands
            and not missing_years
            and term_coverage >= self.config.grading.minimum_term_coverage
            and numeric_anchor_ok
        )
        reasons = []
        if not enough_candidates:
            reasons.append(f"only {len(candidates)} candidate pages")
        if missing_operands:
            reasons.append(f"missing operands={missing_operands}")
        if missing_numeric:
            reasons.append(f"operands without nearby values={missing_numeric}")
        if missing_years:
            reasons.append(f"missing years={missing_years}")
        if term_coverage < self.config.grading.minimum_term_coverage:
            reasons.append(f"term coverage={term_coverage:.2f}")
        if not numeric_anchor_ok:
            reasons.append("no numeric anchor")
        return EvidenceGrade(
            complete=complete,
            covered_operands=covered,
            missing_operands=missing_operands,
            missing_numeric_operands=missing_numeric,
            missing_terms=missing_terms,
            missing_years=missing_years,
            term_coverage=term_coverage,
            numeric_anchor_ok=numeric_anchor_ok,
            reason="evidence checks passed" if complete else "; ".join(reasons),
        )

    def retry_queries(self, question: str, plan: QueryPlan, grade: EvidenceGrade) -> list[str]:
        suffix = " ".join([*grade.missing_years, *plan.years, *plan.statement_types]).strip()
        targets = list(
            dict.fromkeys(
                [*grade.missing_numeric_operands, *grade.missing_operands, *plan.operands]
            )
        )
        queries = [f"{' '.join(aliases_for(target))} {suffix}".strip() for target in targets]
        if not queries:
            terms = " ".join(grade.missing_terms[:6])
            queries = [f"{terms} {suffix} {question}".strip()]
        return list(dict.fromkeys(query for query in queries if query))
