import pytest
import numpy as np
from scripts.overfit_check import OverfitChecker


def make_metrics(**overrides):
    defaults = {
        "train_sharpe": 1.5, "val_sharpe": 1.2, "test_sharpe": 0.8,
        "annual_return": 0.35, "max_drawdown": 0.18,
        "n_trades": 80, "n_params": 10,
    }
    defaults.update(overrides)
    return defaults


class TestOverfitCheck:
    def test_valid_strategy_passes(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(), regime_count=2, sensitivity_stable=True)
        assert result["quality"] == "valid"
        assert len(result["flags"]) == 0

    def test_low_sample_ratio_fails(self):
        checker = OverfitChecker()
        m = make_metrics(train_sharpe=2.5, val_sharpe=0.3)
        result = checker.check(m, regime_count=2, sensitivity_stable=True)
        assert result["quality"] == "suspect"
        assert any("样本外一致性" in f for f in result["flags"])

    def test_few_trades_triggers_warning(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(n_trades=15), regime_count=2, sensitivity_stable=True)
        assert result["quality"] == "suspect"
        assert any("交易次数" in f for f in result["flags"])

    def test_single_regime_flagged(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(), regime_count=1, sensitivity_stable=True)
        assert result["quality"] == "suspect"
        assert any("时段" in f for f in result["flags"])

    def test_param_sensitivity_flagged(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(), regime_count=2, sensitivity_stable=False)
        assert result["quality"] == "suspect"
        assert any("参数敏感" in f for f in result["flags"])

    def test_adjusted_sharpe_in_metrics(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(), regime_count=2, sensitivity_stable=True)
        assert "adjusted_sharpe" in result
        assert result["adjusted_sharpe"] < 10  # should be less than raw
