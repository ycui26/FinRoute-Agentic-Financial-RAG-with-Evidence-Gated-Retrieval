from __future__ import annotations


OPERAND_ALIASES = {
    "revenue": ["revenue", "net sales", "sales"],
    "net income": ["net income", "net earnings", "profit attributable"],
    "operating income": ["operating income", "income from operations", "operating profit"],
    "current assets": ["current assets", "total current assets"],
    "current liabilities": ["current liabilities", "total current liabilities"],
    "cash from operations": [
        "cash from operations",
        "operating cash flow",
        "net cash provided by operating activities",
    ],
    "capital expenditures": [
        "capital expenditures",
        "capex",
        "purchases of property plant and equipment",
    ],
}


def aliases_for(operand: str) -> list[str]:
    normalized = operand.lower().replace(" current period", "").replace(" prior period", "")
    return OPERAND_ALIASES.get(normalized, [normalized])
