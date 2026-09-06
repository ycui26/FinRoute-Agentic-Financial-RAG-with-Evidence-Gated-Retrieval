from __future__ import annotations

import re

import pandas as pd

from finroute.indexing import HybridIndex


DEFAULT_ALIASES = {
    "alphabet": ["alphabet", "google", "googl", "goog"],
    "johnson & johnson": ["johnson & johnson", "jnj"],
    "microsoft": ["microsoft", "msft"],
    "amazon": ["amazon", "amzn"],
    "apple": ["apple", "aapl"],
    "meta": ["meta", "facebook", "fb"],
}


class DocumentRouter:
    def __init__(
        self,
        index: HybridIndex,
        aliases: dict[str, list[str]] | None = None,
        exact_company_bonus: float = 0.02,
        exact_year_bonus: float = 0.01,
    ):
        self.index = index
        self.aliases = aliases or DEFAULT_ALIASES
        self.exact_company_bonus = exact_company_bonus
        self.exact_year_bonus = exact_year_bonus

    def route(self, question: str, top_k: int = 3) -> pd.DataFrame:
        ranked = self.index.search(question, top_n=max(20, top_k * 5)).copy()
        if ranked.empty:
            return ranked
        lower = question.lower()
        years = set(re.findall(r"(?<!\d)20\d{2}(?!\d)", question))
        company_matches = {
            company
            for company, aliases in self.aliases.items()
            if any(re.search(rf"\b{re.escape(alias.lower())}\b", lower) for alias in aliases)
        }
        companies = (
            ranked["company"].astype(str)
            if "company" in ranked.columns
            else pd.Series("", index=ranked.index)
        )
        filing_years = (
            ranked["year"].astype(str)
            if "year" in ranked.columns
            else pd.Series("", index=ranked.index)
        )
        ranked["exact_company"] = [
            int(str(company).lower() in company_matches)
            for company in companies
        ]
        ranked["exact_year"] = [
            int(str(year) in years) for year in filing_years
        ]
        ranked["route_score"] = (
            ranked["rrf_score"]
            + self.exact_company_bonus * ranked["exact_company"]
            + self.exact_year_bonus * ranked["exact_year"]
        )
        return (
            ranked.sort_values("route_score", ascending=False)
            .drop_duplicates("doc_name")
            .head(top_k)
            .reset_index(drop=True)
        )
