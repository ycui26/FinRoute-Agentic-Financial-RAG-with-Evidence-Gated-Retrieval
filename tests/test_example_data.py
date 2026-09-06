import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pymupdf")  # full CI runs in Docker where pymupdf is installed

from finroute.preprocessing import extract_table_blocks, parse_pdf_pages

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
_spec = importlib.util.spec_from_file_location("make_example_data", EXAMPLES_DIR / "make_example_data.py")
make_example_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_example_data)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("example_data")
    make_example_data.generate(data_dir)
    return data_dir


def test_metadata_and_gold_labels(dataset):
    metadata = pd.read_csv(dataset / "documents.csv")
    assert set(metadata.columns) >= {"doc_name", "file_name", "company", "year", "form"}
    assert len(metadata) == 4
    for file_name in metadata["file_name"]:
        assert (dataset / "pdfs" / file_name).exists()

    rows = [json.loads(line) for line in (dataset / "evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    known_docs = set(metadata["doc_name"])
    for row in rows:
        assert row["question"] and row["doc_name"] in known_docs
        for evidence in row["evidence"]:
            assert isinstance(evidence["evidence_page_num"], int)


def test_generated_pdfs_parse_with_expected_structure(dataset):
    metadata = pd.read_csv(dataset / "documents.csv")
    for record in metadata.to_dict(orient="records"):
        pages = parse_pdf_pages(dataset / "pdfs" / record["file_name"], record)
        assert len(pages) == len(make_example_data.PAGE_ORDER)
        assert pages["text"].str.contains(record["company"].split(",")[0]).all()

        income_page = pages.iloc[make_example_data.GOLD_PAGE["income_statement"]]
        assert "Consolidated Statements of Operations" in income_page["text"]
        slug, year, _ = record["doc_name"].split("_")
        expected = f"Revenue ${make_example_data.FINANCIALS[slug]['revenue'][int(year)]:,}"
        assert expected in income_page["text"]


def test_table_extraction_finds_statement_blocks(dataset):
    metadata = pd.read_csv(dataset / "documents.csv")
    frames = [
        parse_pdf_pages(dataset / "pdfs" / record["file_name"], record)
        for record in metadata.to_dict(orient="records")
    ]
    tables = extract_table_blocks(pd.concat(frames, ignore_index=True))
    statement_types = set(tables["statement_type"])
    assert {"income_statement", "balance_sheet", "cash_flow_statement"} <= statement_types


def test_derived_totals_are_consistent(dataset):
    for slug in make_example_data.FINANCIALS:
        for year in make_example_data.FILING_YEARS:
            gross = make_example_data.gross_profit(slug, year)
            operating = make_example_data.operating_income(slug, year)
            assert operating < gross
            assert make_example_data.total_current_assets(slug, year) > make_example_data.total_current_liabilities(slug, year)
