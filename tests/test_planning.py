from finroute.planning import RulePlanner, extract_years, strip_preamble


def test_metric_question_uses_calculation_route():
    plan = RulePlanner().plan("What was the operating margin in 2023?")
    assert plan.route == "calculation"
    assert plan.metric_name == "operating_margin"
    assert set(plan.operands) == {"operating income", "revenue"}
    assert plan.requires_numeric


def test_qualitative_question_uses_narrative_route():
    plan = RulePlanner().plan("Why did the company increase its borrowing capacity?")
    assert plan.route == "narrative"
    assert not plan.requires_numeric


def test_preamble_and_year_extraction():
    question = "Assume that you are a public equities analyst. What was revenue in 2022?"
    assert strip_preamble(question).startswith("What was revenue")
    assert extract_years(question) == ["2022"]
