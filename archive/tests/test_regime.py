"""市场状态识别测试。"""
import pytest
import pandas as pd
import numpy as np
from models.regime import detect_regime


class TestRegime:
    def test_detect_regime_returns_labels(self):
        """应返回每个日期的市场状态标签。"""
        np.random.seed(42)
        dates = pd.date_range("2018-01-02", periods=500, freq="B")
        close = 3000 + np.cumsum(np.random.randn(500) * 30)
        df = pd.DataFrame({"trade_date": dates, "close": close})

        regimes = detect_regime(df)
        assert "trade_date" in regimes.columns
        assert "regime" in regimes.columns
        assert regimes["regime"].nunique() >= 2

    def test_regime_labels_are_in_set(self):
        """标签应为 bull/bear/sideways。"""
        np.random.seed(42)
        dates = pd.date_range("2018-01-02", periods=500, freq="B")
        close = 3000 + np.cumsum(np.random.randn(500) * 30)
        df = pd.DataFrame({"trade_date": dates, "close": close})

        regimes = detect_regime(df)
        valid_labels = {"bull", "bear", "sideways"}
        assert regimes["regime"].isin(valid_labels).all()

    def test_bull_when_above_ma250(self):
        """价格在 MA250 上方 + 正收益 → bull。"""
        np.random.seed(42)
        dates = pd.date_range("2018-01-02", periods=500, freq="B")
        close = 3000 + np.arange(500) * 2 + np.random.randn(500) * 10
        df = pd.DataFrame({"trade_date": dates, "close": close})

        regimes = detect_regime(df)
        late_regimes = regimes.tail(100)["regime"]
        assert (late_regimes == "bull").sum() > 50
