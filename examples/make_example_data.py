"""Generate a small synthetic FinanceBench-style dataset so FinRoute can be tried
end to end without downloading real filings.

Creates under --data-dir (default: data/):
    pdfs/*.pdf          four fictional 10-K filings (two companies x two years)
    documents.csv       the metadata table expected by `finroute build-index`
    evaluation.jsonl    four gold-labeled questions for `finroute evaluate`

All content is fictional and deterministic. Numbers are internally consistent
(gross profit, operating income, and balance-sheet totals are derived), so the
metric questions in evaluation.jsonl have real answers on real pages.

Requires pymupdf, which is already a FinRoute dependency.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import pandas as pd
import pymupdf

PAGE_WIDTH, PAGE_HEIGHT = 612, 792  # US Letter, points
MARGIN = 72
FONT_SIZE = 10
LINE_HEIGHT = 16
WRAP_WIDTH = 90

FORM = "10-K"

FINANCIALS = {
    "meridian": {
        "revenue": {2021: 9900, 2022: 10950, 2023: 12480},
        "cost_of_revenue": {2021: 6010, 2022: 6540, 2023: 7360},
        "research_and_development": {2021: 1400, 2022: 1560, 2023: 1790},
        "selling_general_administrative": {2021: 1090, 2022: 1420, 2023: 1470},
        "net_income": {2021: 1080, 2022: 1090, 2023: 1410},
        "interest_expense": {2021: 88, 2022: 92, 2023: 85},
        "cash_and_equivalents": {2021: 1720, 2022: 1880, 2023: 2140},
        "receivables": {2021: 1380, 2022: 1510, 2023: 1690},
        "inventory": {2021: 420, 2022: 390, 2023: 410},
        "other_current_assets": {2021: 330, 2022: 430, 2023: 620},
        "accounts_payable": {2021: 780, 2022: 790, 2023: 860},
        "accrued_expenses": {2021: 690, 2022: 610, 2023: 640},
        "short_term_debt": {2021: 710, 2022: 930, 2023: 910},
        "long_term_debt": {2021: 2350, 2022: 2250, 2023: 2200},
        "property_equipment": {2021: 2980, 2022: 3070, 2023: 3310},
        "goodwill": {2021: 900, 2022: 900, 2023: 940},
        "cash_from_operations": {2021: 2150, 2022: 2410, 2023: 2730},
        "capital_expenditures": {2021: 700, 2022: 820, 2023: 940},
        "depreciation": {2021: 450, 2022: 480, 2023: 520},
        "credit_facility": {2021: 500, 2022: 500, 2023: 750},
    },
    "cobalt": {
        "revenue": {2021: 7600, 2022: 8240, 2023: 8910},
        "cost_of_revenue": {2021: 4940, 2022: 5350, 2023: 5760},
        "research_and_development": {2021: 380, 2022: 410, 2023: 450},
        "selling_general_administrative": {2021: 860, 2022: 930, 2023: 1180},
        "net_income": {2021: 1010, 2022: 1090, 2023: 1180},
        "interest_expense": {2021: 140, 2022: 132, 2023: 128},
        "cash_and_equivalents": {2021: 2210, 2022: 2480, 2023: 2760},
        "receivables": {2021: 1150, 2022: 1240, 2023: 1310},
        "inventory": {2021: 980, 2022: 1040, 2023: 1110},
        "other_current_assets": {2021: 250, 2022: 270, 2023: 300},
        "accounts_payable": {2021: 940, 2022: 1010, 2023: 1080},
        "accrued_expenses": {2021: 780, 2022: 830, 2023: 890},
        "short_term_debt": {2021: 1230, 2022: 1280, 2023: 1340},
        "long_term_debt": {2021: 5100, 2022: 4950, 2023: 4800},
        "property_equipment": {2021: 6420, 2022: 6750, 2023: 7180},
        "goodwill": {2021: 1830, 2022: 1840, 2023: 1860},
        "cash_from_operations": {2021: 2640, 2022: 2910, 2023: 3180},
        "capital_expenditures": {2021: 1420, 2022: 1530, 2023: 1650},
        "depreciation": {2021: 1180, 2022: 1240, 2023: 1310},
        "credit_facility": {2021: 1000, 2022: 1000, 2023: 1000},
    },
}

COMPANY_NAMES = {
    "meridian": "Meridian Technologies, Inc.",
    "cobalt": "Cobalt Energy Corp",
}

FILING_YEARS = [2022, 2023]

# Page order inside every generated filing. Gold labels in evaluation.jsonl
# refer to these zero-based indices. Nine pages per filing keep the candidate
# pool above the grader's `minimum_candidate_pages` threshold.
PAGE_ORDER = [
    "cover",
    "business",
    "risk_factors",
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "mda",
    "liquidity",
    "notes",
]
GOLD_PAGE = {name: index for index, name in enumerate(PAGE_ORDER)}


def gross_profit(slug: str, year: int) -> int:
    m = FINANCIALS[slug]
    return m["revenue"][year] - m["cost_of_revenue"][year]


def operating_income(slug: str, year: int) -> int:
    m = FINANCIALS[slug]
    return gross_profit(slug, year) - m["research_and_development"][year] - m["selling_general_administrative"][year]


def total_current_assets(slug: str, year: int) -> int:
    m = FINANCIALS[slug]
    return (
        m["cash_and_equivalents"][year]
        + m["receivables"][year]
        + m["inventory"][year]
        + m["other_current_assets"][year]
    )


def total_current_liabilities(slug: str, year: int) -> int:
    m = FINANCIALS[slug]
    return m["accounts_payable"][year] + m["accrued_expenses"][year] + m["short_term_debt"][year]


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=WRAP_WIDTH)


def cover_lines(slug: str, year: int) -> list[str]:
    return [
        COMPANY_NAMES[slug],
        "",
        f"Annual Report on Form {FORM}",
        "",
        f"Fiscal Year Ended December 31, {year}",
    ]


def business_lines(slug: str, year: int) -> list[str]:
    if slug == "meridian":        text = (
            "Meridian Technologies, Inc. is a global provider of cloud infrastructure, "
            "enterprise software, and collaboration devices. The company serves more than "
            "forty thousand business customers from offices in North America, Europe, and "
            "the Asia Pacific region, and operates data centers on three continents. Its "
            "reportable segments are Cloud Services, Enterprise Software, and Devices. The "
            "Cloud Services segment hosts managed compute, storage, and analytics platforms "
            "under multi-year subscriptions. The Enterprise Software segment licenses "
            "workflow and compliance applications used by finance, legal, and procurement "
            "teams. The Devices segment sells displays, conferencing hardware, and "
            "accessories that connect to the company's software platforms. Recurring "
            "revenue from subscriptions and support represented the majority of total "
            "revenue in fiscal years."
        )
    else:
        text = (
            "Cobalt Energy Corp explores for, develops, and produces natural gas and "
            "petroleum liquids from properties in the mountain west and the gulf region of "
            "North America. The company sells its production to marketers, refiners, and "
            "trading houses under contracts tied to published index prices. Operations are "
            "organized into two reportable segments: Natural Gas, and Petroleum Liquids. "
            "The Natural Gas segment gathers, processes, and markets dry gas from "
            "appalachian and rocky mountain basins. The Petroleum Liquids segment "
            "transports condensate and natural gas liquids by rail and pipeline to gulf "
            "coast markets. The company hedged a portion of expected production with "
            "swaps, collars, and basis contracts, and maintains midstream agreements that "
            "provide firm transportation capacity for its products."
        )
    return wrap(text)


def income_statement_lines(slug: str, year: int) -> list[str]:
    m = FINANCIALS[slug]
    prior = year - 1
    operating = operating_income(slug, year)
    operating_prior = operating_income(slug, prior)
    pre_tax = operating - m["interest_expense"][year]
    pre_tax_prior = operating_prior - m["interest_expense"][prior]
    return [
        "Consolidated Statements of Operations",
        "",
        "(in millions)",
        f"Year Ended December 31, {year} {prior}",
        "",
        f"Revenue ${m['revenue'][year]:,} ${m['revenue'][prior]:,}",
        f"Cost of revenue ${m['cost_of_revenue'][year]:,} ${m['cost_of_revenue'][prior]:,}",
        f"Gross profit ${gross_profit(slug, year):,} ${gross_profit(slug, prior):,}",
        f"Research and development ${m['research_and_development'][year]:,} ${m['research_and_development'][prior]:,}",
        f"Selling, general and administrative ${m['selling_general_administrative'][year]:,} ${m['selling_general_administrative'][prior]:,}",
        f"Operating income ${operating:,} ${operating_prior:,}",
        f"Interest expense ${m['interest_expense'][year]:,} ${m['interest_expense'][prior]:,}",
        f"Income before income taxes ${pre_tax:,} ${pre_tax_prior:,}",
        f"Provision for income taxes ${pre_tax - m['net_income'][year]:,} ${pre_tax_prior - m['net_income'][prior]:,}",
        f"Net income ${m['net_income'][year]:,} ${m['net_income'][prior]:,}",
    ]


def balance_sheet_lines(slug: str, year: int) -> list[str]:
    m = FINANCIALS[slug]
    prior = year - 1
    lines = [
        "Consolidated Balance Sheets",
        "",
        "(in millions)",
        f"As of December 31, {year} {prior}",
        "",
    ]
    assets = [
        ("Cash and cash equivalents", "cash_and_equivalents"),
        ("Accounts receivable, net", "receivables"),
        ("Inventory", "inventory"),
        ("Prepaid expenses and other current assets", "other_current_assets"),
    ]
    for label, key in assets:
        lines.append(f"{label} ${m[key][year]:,} ${m[key][prior]:,}")
    lines.append(f"Total current assets ${total_current_assets(slug, year):,} ${total_current_assets(slug, prior):,}")
    lines.append(f"Property and equipment, net ${m['property_equipment'][year]:,} ${m['property_equipment'][prior]:,}")
    lines.append(f"Goodwill and other intangible assets ${m['goodwill'][year]:,} ${m['goodwill'][prior]:,}")
    total_assets = total_current_assets(slug, year) + m["property_equipment"][year] + m["goodwill"][year]
    total_assets_prior = total_current_assets(slug, prior) + m["property_equipment"][prior] + m["goodwill"][prior]
    lines.append(f"Total assets ${total_assets:,} ${total_assets_prior:,}")
    for label, key in [
        ("Accounts payable", "accounts_payable"),
        ("Accrued expenses", "accrued_expenses"),
        ("Short-term debt", "short_term_debt"),
    ]:
        lines.append(f"{label} ${m[key][year]:,} ${m[key][prior]:,}")
    lines.append(
        f"Total current liabilities ${total_current_liabilities(slug, year):,} "
        f"${total_current_liabilities(slug, prior):,}"
    )
    lines.append(f"Long-term debt ${m['long_term_debt'][year]:,} ${m['long_term_debt'][prior]:,}")
    total_liabilities = total_current_liabilities(slug, year) + m["long_term_debt"][year]
    total_liabilities_prior = total_current_liabilities(slug, prior) + m["long_term_debt"][prior]
    lines.append(f"Total liabilities ${total_liabilities:,} ${total_liabilities_prior:,}")
    lines.append(
        f"Total stockholders' equity ${total_assets - total_liabilities:,} "
        f"${total_assets_prior - total_liabilities_prior:,}"
    )
    return lines


def cash_flow_lines(slug: str, year: int) -> list[str]:
    m = FINANCIALS[slug]
    prior = year - 1
    capex, capex_prior = m["capital_expenditures"][year], m["capital_expenditures"][prior]
    return [
        "Consolidated Statements of Cash Flows",
        "",
        "(in millions)",
        f"Year Ended December 31, {year} {prior}",
        "",
        f"Net income ${m['net_income'][year]:,} ${m['net_income'][prior]:,}",
        f"Depreciation and amortization ${m['depreciation'][year]:,} ${m['depreciation'][prior]:,}",
        f"Cash from operations ${m['cash_from_operations'][year]:,} ${m['cash_from_operations'][prior]:,}",
        f"Capital expenditures ${capex:,} ${capex_prior:,}",
        f"Cash used in investing activities ${capex + 180:,} ${capex_prior + 170:,}",
        f"Cash used in financing activities ${int(m['net_income'][year] * 0.4) + 270:,} "
        f"${int(m['net_income'][prior] * 0.4) + 260:,}",
    ]


def mda_lines(slug: str, year: int) -> list[str]:
    m = FINANCIALS[slug]
    prior = year - 1
    revenue, revenue_prior = m["revenue"][year], m["revenue"][prior]
    growth_pct = (revenue / revenue_prior - 1) * 100
    margin = operating_income(slug, year) / revenue * 100
    margin_prior = operating_income(slug, prior) / revenue_prior * 100
    free_cash = m["cash_from_operations"][year] - m["capital_expenditures"][year]
    paragraphs = [
        f"Revenue increased {growth_pct:.0f} percent to ${revenue:,} million in fiscal {year}, "
        f"compared with ${revenue_prior:,} million in fiscal {prior}.",
        f"Operating margin was {margin:.1f} percent in fiscal {year}, compared with "
        f"{margin_prior:.1f} percent in fiscal {prior}.",
    ]
    if slug == "meridian":
        if year == 2023:
            paragraphs.append(
                "In the third quarter of 2023, we increased the capacity of our revolving "
                "credit facility from $500 million to $750 million to fund the Northvale "
                "data-center expansion and to preserve liquidity for seasonal working "
                "capital needs."
            )
        else:
            paragraphs.append(
                f"Our revolving credit facility provided ${m['credit_facility'][year]:,} million "
                "of borrowing capacity throughout the year, and no borrowings were "
                "outstanding at year end."
            )
        paragraphs.append(
            f"Free cash flow reached ${free_cash:,} million after capital expenditures of "
            f"${m['capital_expenditures'][year]:,} million, reflecting disciplined spending "
            "on data-center construction."
        )
    else:
        paragraphs.append(
            f"During {year} we expanded our natural gas hedging program to cover "
            "approximately 70 percent of expected next-year production, reducing earnings "
            "volatility from index price swings."
        )
        paragraphs.append(
            f"Our current ratio stood at {total_current_assets(slug, year) / total_current_liabilities(slug, year):.1f} "
            "at year end, reflecting disciplined working capital management, and free cash "
            f"flow of ${free_cash:,} million funded dividend payments and debt reduction."
        )
    return [line for paragraph in paragraphs for line in wrap(paragraph)]


def risk_factors_lines(slug: str, year: int) -> list[str]:
    if slug == "meridian":
        paragraphs = [
            "Competition in the markets for cloud infrastructure and enterprise software "
            "is intense, and pricing pressure or slower information technology spending "
            "could harm margins and revenue growth in future periods.",
            "A failure to prevent security incidents, to protect customer data, or to "
            "maintain the availability of our platforms could damage our reputation, "
            "expose the company to liability, and reduce renewals of multi-year "
            "subscriptions.",
            "Changes in privacy, data residency, and artificial intelligence regulation "
            "across the jurisdictions where we operate could increase compliance costs "
            "and require changes to our products.",
        ]
    else:
        paragraphs = [
            "Prices for natural gas and petroleum liquids are volatile, and declines in "
            "commodity prices could reduce revenues, cause impairments of properties, "
            "and reduce the quantity of reserves that are economic to develop.",
            "Hedging activities may not fully protect expected revenue, and counterparties "
            "to derivative contracts could fail to perform their obligations.",
            "Weather, pipeline capacity constraints, and regulatory permitting decisions "
            "affect production levels and the timing of capital projects.",
        ]
    return [line for paragraph in paragraphs for line in wrap(paragraph)]


def liquidity_lines(slug: str, year: int) -> list[str]:
    m = FINANCIALS[slug]
    paragraphs = [
        f"As of December 31, {year}, we held ${m['cash_and_equivalents'][year]:,} million "
        "of cash and cash equivalents, which together with cash generated from operations "
        "is expected to fund planned capital spending and working capital needs.",
        f"Our revolving credit facility provided ${m['credit_facility'][year]:,} million of "
        "borrowing capacity, and long-term debt of "
        f"${m['long_term_debt'][year]:,} million carries maturities beyond the next "
        "several years.",
        f"Capital expenditures of ${int(m['capital_expenditures'][year] * 1.1):,} million "
        "are planned for the coming year, subject to market conditions.",
    ]
    return [line for paragraph in paragraphs for line in wrap(paragraph)]


def notes_lines(slug: str, year: int) -> list[str]:
    if slug == "meridian":
        table = [
            "Revenue by reportable segment:",
            f"Cloud Services ${int(FINANCIALS[slug]['revenue'][year] * 0.57):,}",
            f"Enterprise Software ${int(FINANCIALS[slug]['revenue'][year] * 0.31):,}",
            f"Devices ${int(FINANCIALS[slug]['revenue'][year] * 0.12):,}",
        ]
        narrative = (
            "No single customer accounted for more than ten percent of total revenue in "
            "any period presented. Subscription contracts are billed in advance and "
            "recognized ratably over the service period. Deferred revenue primarily "
            "represents unbilled amounts under multi-year cloud services agreements."
        )
    else:
        table = [
            "Production and realized prices:",
            f"Natural gas production (bcfe) {1_980 + (year - 2021) * 90:,}",
            f"Petroleum liquids production (mbbls) {86 + (year - 2021) * 6:,}",
            f"Average realized natural gas price ${'2.94' if year == 2023 else '6.41'} per mcf",
        ]
        narrative = (
            "Commodity derivative instruments are designated as cash flow hedges and "
            "recorded in accumulated other comprehensive income until settled. The company "
            "maintains firm transportation agreements covering the majority of expected "
            "volumes, and no single counterparty represented more than ten percent of "
            "revenue."
        )
    return table + [""] + wrap(narrative)


PAGE_BUILDERS = {
    "cover": cover_lines,
    "business": business_lines,
    "risk_factors": risk_factors_lines,
    "income_statement": income_statement_lines,
    "balance_sheet": balance_sheet_lines,
    "cash_flow": cash_flow_lines,
    "mda": mda_lines,
    "liquidity": liquidity_lines,
    "notes": notes_lines,
}

# Item headings give each page the `section` metadata that routing and
# statement-type inference rely on (see preprocessing.SECTION_RE).
PAGE_HEADINGS = {
    "business": "Item 1. Business",
    "risk_factors": "Item 1A. Risk Factors",
    "mda": "Item 7. Management's Discussion and Analysis",
    "notes": "Notes to Consolidated Financial Statements",
}


def filing_pages(slug: str, year: int) -> list[list[str]]:
    pages = []
    for name in PAGE_ORDER:
        lines = []
        if name in PAGE_HEADINGS:
            lines += [PAGE_HEADINGS[name], ""]
        lines += PAGE_BUILDERS[name](slug, year)
        lines += ["", f"{COMPANY_NAMES[slug]} | Form {FORM} | Fiscal Year {year}"]
        pages.append(lines)
    return pages


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    document = pymupdf.open()
    for lines in pages:
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        y = MARGIN
        for line in lines:
            if line:
                page.insert_text((MARGIN, y), line, fontname="helv", fontsize=FONT_SIZE)
            y += LINE_HEIGHT
    document.save(path)
    document.close()


def generate(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    pdf_dir = data_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for slug in FINANCIALS:
        for year in FILING_YEARS:
            doc_name = f"{slug}_{year}_10k"
            file_name = f"{doc_name}.pdf"
            write_pdf(pdf_dir / file_name, filing_pages(slug, year))
            records.append(
                {
                    "doc_name": doc_name,
                    "file_name": file_name,
                    "company": COMPANY_NAMES[slug],
                    "year": year,
                    "form": FORM,
                }
            )
    pd.DataFrame(records).to_csv(data_dir / "documents.csv", index=False)

    gold_rows = [
        {
            "question_id": "example-001",
            "question": "What was the operating margin of Meridian Technologies in 2023?",
            "doc_name": "meridian_2023_10k",
            "evidence": [
                {"evidence_page_num": GOLD_PAGE["income_statement"]},
                {"evidence_page_num": GOLD_PAGE["mda"]},
            ],
        },
        {
            "question_id": "example-002",
            "question": "What was the current ratio of Cobalt Energy Corp in 2023?",
            "doc_name": "cobalt_2023_10k",
            "evidence": [{"evidence_page_num": GOLD_PAGE["balance_sheet"]}],
        },
        {
            "question_id": "example-003",
            "question": "Why did Meridian Technologies increase its revolving credit facility in 2023?",
            "doc_name": "meridian_2023_10k",
            "evidence": [{"evidence_page_num": GOLD_PAGE["mda"]}],
        },
        {
            "question_id": "example-004",
            "question": "How much free cash flow did Meridian Technologies generate in 2023?",
            "doc_name": "meridian_2023_10k",
            "evidence": [{"evidence_page_num": GOLD_PAGE["cash_flow"]}],
        },
    ]
    with (data_dir / "evaluation.jsonl").open("w", encoding="utf-8") as handle:
        for row in gold_rows:
            handle.write(json.dumps(row) + "\n")
    return data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the synthetic FinRoute example dataset")
    parser.add_argument("--data-dir", default="data", help="output directory (default: data)")
    args = parser.parse_args()
    generate(args.data_dir)
    print(f"Wrote synthetic filings, documents.csv, and evaluation.jsonl under {args.data_dir}/")
