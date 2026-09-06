from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Route = Literal["narrative", "table_lookup", "calculation", "mixed"]


@dataclass
class QueryPlan:
    route: Route
    metric_name: str | None = None
    years: list[str] = field(default_factory=list)
    operands: list[str] = field(default_factory=list)
    required_terms: list[str] = field(default_factory=list)
    statement_types: list[str] = field(default_factory=list)
    formula: str | None = None
    subqueries: list[str] = field(default_factory=list)
    requires_multiple_pages: bool = False
    requires_numeric: bool = False
    route_confidence: float = 0.5
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceGrade:
    complete: bool
    covered_operands: list[str] = field(default_factory=list)
    missing_operands: list[str] = field(default_factory=list)
    missing_numeric_operands: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    missing_years: list[str] = field(default_factory=list)
    term_coverage: float = 0.0
    numeric_anchor_ok: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    question: str
    plan: QueryPlan
    allowed_documents: list[str]
    pages: list[dict[str, Any]]
    grade: EvidenceGrade
    abstain: bool
    stop_reason: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "plan": self.plan.to_dict(),
            "allowed_documents": self.allowed_documents,
            "pages": self.pages,
            "grade": self.grade.to_dict(),
            "abstain": self.abstain,
            "stop_reason": self.stop_reason,
            "trace": self.trace,
            "candidate_count": self.candidate_count,
        }
