import pandas as pd

from finroute.config import FinRouteConfig
from finroute.grading import EvidenceGrader
from finroute.schemas import QueryPlan


def test_complete_numeric_evidence():
    config = FinRouteConfig()
    config.grading.minimum_candidate_pages = 1
    plan = QueryPlan(
        route="calculation",
        operands=["operating income", "revenue"],
        years=["2023"],
        required_terms=["operating", "margin"],
        requires_numeric=True,
    )
    selected = pd.DataFrame(
        {
            "rerank_excerpt": [
                "In 2023, operating income was $20 and revenue was $100; operating margin increased."
            ]
        }
    )
    grade = EvidenceGrader(config).grade("question", plan, selected, selected)
    assert grade.complete
    assert not grade.missing_operands


def test_missing_operand_creates_retry_query():
    config = FinRouteConfig()
    config.grading.minimum_candidate_pages = 1
    plan = QueryPlan(route="calculation", operands=["revenue"], requires_numeric=True)
    selected = pd.DataFrame({"rerank_excerpt": ["The filing discusses customers."]})
    grader = EvidenceGrader(config)
    grade = grader.grade("What was revenue?", plan, selected, selected)
    assert not grade.complete
    assert "revenue" in grade.missing_operands
    assert grader.retry_queries("What was revenue?", plan, grade)
