import pandas as pd
import pytest

pytest.importorskip("rank_bm25")  # finroute.indexing needs it; absent in the fast CI job

from finroute.config import FinRouteConfig
from finroute.retrieval import PageRetriever
from finroute.schemas import QueryPlan


def make_retriever() -> PageRetriever:
    pages = pd.DataFrame(
        [
            {"doc_name": "doc_a", "page_idx0": 0, "page_num": 1, "text": "alpha"},
            {"doc_name": "doc_a", "page_idx0": 1, "page_num": 2, "text": "beta"},
        ]
    )
    return PageRetriever(child_index=None, table_index=None, pages=pages, config=FinRouteConfig())


def narrative_frame() -> pd.DataFrame:
    # Narrative candidates come from the child corpus, which has no
    # table_quality_score column.
    return pd.DataFrame(
        [
            {"doc_name": "doc_a", "page_idx0": 0, "context_text": "revenue 2023"},
            {"doc_name": "doc_a", "page_idx0": 1, "context_text": "revenue 2022"},
        ]
    )


def test_annotate_defaults_table_quality_for_narrative_frames():
    retriever = make_retriever()
    plan = QueryPlan(route="narrative")
    annotated = retriever._annotate(narrative_frame(), "narrative_child", "revenue", plan, ["doc_a"])
    assert annotated["table_quality_score"].eq(0.5).all()


def test_annotate_preserves_existing_table_quality_scores():
    retriever = make_retriever()
    plan = QueryPlan(route="table_lookup", statement_types=["income_statement"])
    frame = narrative_frame()
    frame["table_quality_score"] = [0.9, None]
    annotated = retriever._annotate(frame, "table_q0", "revenue", plan, ["doc_a"])
    assert annotated["table_quality_score"].tolist()[0] == 0.9
    assert annotated["table_quality_score"].isna().sum() == 0
