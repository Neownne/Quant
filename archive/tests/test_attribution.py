import pytest
import numpy as np
from scripts.attribution import compute_factor_contribution


class TestFactorContribution:
    def test_positive_contribution_for_winning_signal(self):
        factor_values = {"momentum": 0.05, "volatility": -0.02, "liquidity": 0.01}
        result = compute_factor_contribution(factor_values, pnl_pct=3.0)
        assert result["momentum"] > 0
        assert isinstance(result["volatility"], float)

    def test_negative_pnl_gives_negative_sign(self):
        factor_values = {"momentum": 0.05, "volatility": -0.02}
        result = compute_factor_contribution(factor_values, pnl_pct=-2.0)
        for v in result.values():
            assert v <= 0

    def test_empty_factors_returns_empty(self):
        result = compute_factor_contribution({}, pnl_pct=5.0)
        assert result == {}
