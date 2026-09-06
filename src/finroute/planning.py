from __future__ import annotations

import re
from dataclasses import dataclass

from finroute.schemas import QueryPlan


@dataclass(frozen=True)
class MetricSpec:
    name: str
    pattern: re.Pattern[str]
    operands: tuple[str, ...]
    statement_types: tuple[str, ...]
    formula: str | None


METRIC_SPECS = (
    MetricSpec(
        "net_working_capital",
        re.compile(r"(?i)net working capital"),
        ("current assets", "current liabilities"),
        ("balance_sheet",),
        "current_assets - current_liabilities",
    ),
    MetricSpec(
        "return_on_assets",
        re.compile(r"(?i)return on assets|\bROA\b"),
        ("net income", "total assets"),
        ("income_statement", "balance_sheet"),
        "net_income / average_total_assets",
    ),
    MetricSpec(
        "quick_ratio",
        re.compile(r"(?i)quick ratio"),
        ("cash and cash equivalents", "marketable securities", "accounts receivable", "current liabilities"),
        ("balance_sheet",),
        "(cash + marketable_securities + accounts_receivable) / current_liabilities",
    ),
    MetricSpec(
        "current_ratio",
        re.compile(r"(?i)(?:working capital|current) ratio"),
        ("current assets", "current liabilities"),
        ("balance_sheet",),
        "current_assets / current_liabilities",
    ),
    MetricSpec(
        "operating_margin",
        re.compile(r"(?i)operating margin"),
        ("operating income", "revenue"),
        ("income_statement",),
        "operating_income / revenue",
    ),
    MetricSpec(
        "gross_margin",
        re.compile(r"(?i)gross margin"),
        ("gross profit", "revenue"),
        ("income_statement",),
        "gross_profit / revenue",
    ),
    MetricSpec(
        "revenue_growth",
        re.compile(r"(?i)(?:revenue|sales).{0,30}(?:growth|change|increase|decrease)"),
        ("revenue current period", "revenue prior period"),
        ("income_statement",),
        "revenue_current / revenue_prior - 1",
    ),
    MetricSpec(
        "free_cash_flow",
        re.compile(r"(?i)free cash flow"),
        ("cash from operations", "capital expenditures"),
        ("cash_flow_statement",),
        "cash_from_operations - capital_expenditures",
    ),
)

PREAMBLE_RE = re.compile(
    r"(?is)^\s*assume that you are a public equities analyst\.\s*"
    r"(?:answer the following question[^:?.]*[:.]\s*)?"
)
QUALITATIVE_RE = re.compile(r"(?i)why|explain|describe|discuss|driver|reason|impact")
NUMERIC_RE = re.compile(r"(?i)how much|what (?:was|is)|calculate|ratio|margin|percent|percentage|amount")
TABLE_RE = re.compile(r"(?i)balance sheet|income statement|cash flow|total assets|liabilities|revenue|net income")
STOPWORDS = {
    "what", "which", "when", "where", "according", "company", "report",
    "filing", "calculate", "determine", "from", "that", "this", "were",
    "was", "are", "and", "the", "for", "with", "how", "much", "many",
}
QUERY_EXPANSIONS = {
    "borrow": "revolving credit facility borrowing capacity commitment",
    "derivative": "derivative instruments notional amount swaps",
    "repurchase": "share repurchase stock buyback purchases",
    "proceeds": "cash proceeds consideration transaction",
    "customers": "major customers customer concentration",
}


def strip_preamble(question: str) -> str:
    cleaned = PREAMBLE_RE.sub("", str(question)).strip()
    return cleaned or str(question).strip()


def extract_years(question: str) -> list[str]:
    years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)
    return list(dict.fromkeys(years))


def extract_required_terms(question: str, limit: int = 10) -> list[str]:
    terms = []
    for word in re.findall(r"[a-z][a-z0-9&-]{2,}", strip_preamble(question).lower()):
        if word not in STOPWORDS and word not in terms:
            terms.append(word)
    return terms[:limit]


def expanded_query(question: str) -> str:
    lower = question.lower()
    additions = [value for key, value in QUERY_EXPANSIONS.items() if key in lower]
    return " ".join([question, *additions]).strip()


class RulePlanner:
    def plan(self, question: str) -> QueryPlan:
        core = strip_preamble(question)
        years = extract_years(core)
        required_terms = extract_required_terms(core)
        spec = next((item for item in METRIC_SPECS if item.pattern.search(core)), None)
        qualitative = bool(QUALITATIVE_RE.search(core))
        numerical = bool(NUMERIC_RE.search(core))

        if spec:
            route = "mixed" if qualitative else "calculation"
            confidence = 0.95
            operands = list(spec.operands)
            statement_types = list(spec.statement_types)
            metric_name, formula = spec.name, spec.formula
            requires_numeric = True
            reason = f"recognized financial metric: {spec.name}"
        elif numerical and TABLE_RE.search(core):
            route = "mixed" if qualitative else "table_lookup"
            confidence = 0.80
            operands, statement_types = [], []
            metric_name = formula = None
            requires_numeric = True
            reason = "financial statement lookup"
        elif qualitative:
            route, confidence = "narrative", 0.85
            operands, statement_types = [], []
            metric_name = formula = None
            requires_numeric = False
            reason = "qualitative disclosure"
        else:
            route = "mixed" if numerical else "narrative"
            confidence = 0.50
            operands, statement_types = [], []
            metric_name = formula = None
            requires_numeric = numerical
            reason = "conservative fallback"

        suffix = " ".join([*years, *[item.replace("_", " ") for item in statement_types]])
        subqueries = [core, expanded_query(core)]
        subqueries.extend(f"{operand} {suffix}".strip() for operand in operands)
        if required_terms:
            subqueries.append(" ".join([*required_terms, *years]))
        subqueries = list(dict.fromkeys(query for query in subqueries if query))
        return QueryPlan(
            route=route,
            metric_name=metric_name,
            years=years,
            operands=operands,
            required_terms=required_terms,
            statement_types=statement_types,
            formula=formula,
            subqueries=subqueries,
            requires_multiple_pages=len(operands) > 1 or len(statement_types) > 1,
            requires_numeric=requires_numeric,
            route_confidence=confidence,
            reason=reason,
        )
