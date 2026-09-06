import pytest

from finroute.generation import safe_calculate


def test_safe_calculator():
    assert safe_calculate("(100 - 40) / 100") == pytest.approx(0.6)


def test_safe_calculator_rejects_calls():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('echo unsafe')")
