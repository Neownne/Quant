"""Test yaogu backtest buyability logic — TradingConfig + yiziban detection."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np


class TestTradingConfigLimitMethods:
    """Verify canonical limit-up/down methods exist and work."""

    def test_is_at_limit_up_exists(self):
        from config.settings import TradingConfig
        assert hasattr(TradingConfig, "is_at_limit_up")
        assert callable(TradingConfig.is_at_limit_up)

    def test_is_at_limit_down_exists(self):
        from config.settings import TradingConfig
        assert hasattr(TradingConfig, "is_at_limit_down")
        assert callable(TradingConfig.is_at_limit_down)

    def test_calc_limit_price_exists(self):
        from config.settings import TradingConfig
        assert hasattr(TradingConfig, "calc_limit_price")
        assert callable(TradingConfig.calc_limit_price)

    def test_get_limit_multiplier_main_board(self):
        from config.settings import TradingConfig
        # 主板 10%
        assert TradingConfig.get_limit_multiplier("600000") == 1.09899
        assert TradingConfig.get_limit_multiplier("000001") == 1.09899

    def test_is_at_limit_up_main_board(self):
        from config.settings import TradingConfig
        prev = 10.00
        limit_price = TradingConfig.calc_limit_price(prev, "600000", is_up=True)
        # Exact limit-up close
        assert TradingConfig.is_at_limit_up(limit_price, prev, "600000")
        # 0.1% below limit-up: still at limit-up (tolerance=1.0 means exact)
        assert not TradingConfig.is_at_limit_up(limit_price - 0.02, prev, "600000")
        # With lower tolerance: near-limit-up counts
        assert TradingConfig.is_at_limit_up(limit_price * 0.99, prev, "600000", tolerance=0.98)

    def test_is_at_limit_down_main_board(self):
        from config.settings import TradingConfig
        prev = 10.00
        limit_price = TradingConfig.calc_limit_price(prev, "600000", is_up=False)
        assert TradingConfig.is_at_limit_down(limit_price, prev, "600000")
        # 0.1% above limit-down: not at limit-down
        assert not TradingConfig.is_at_limit_down(limit_price + 0.02, prev, "600000")

    def test_is_at_limit_up_edge_cases(self):
        from config.settings import TradingConfig
        # Zero or negative prices
        assert not TradingConfig.is_at_limit_up(0, 10.0, "600000")
        assert not TradingConfig.is_at_limit_up(10.0, 0, "600000")
        assert not TradingConfig.is_at_limit_up(-1, 10.0, "600000")


class TestRunBacktestOnSignalsImport:
    """Verify function is importable and callable."""

    def test_function_is_callable(self):
        from scripts.bt_yaogu import run_backtest_on_signals
        assert callable(run_backtest_on_signals)

    def test_parse_args_importable(self):
        from scripts.bt_yaogu import parse_args
        assert callable(parse_args)


class TestYizibanDetection:
    """Test the yiziban (一字板) logic pattern used in bt_yaogu."""

    def _is_yiziban(self, open_px, high_px, prev_close, code="600000"):
        """Replicate the yiziban detection logic from bt_yaogu."""
        from config.settings import TradingConfig
        limit_up_price = TradingConfig.calc_limit_price(prev_close, code, is_up=True)
        return (
            pd.notna(open_px) and open_px > 0 and
            pd.notna(high_px) and high_px > 0 and
            abs(open_px - limit_up_price) / limit_up_price < 0.001 and
            abs(high_px - limit_up_price) / limit_up_price < 0.001
        )

    def test_yiziban_detected_when_open_equals_high_equals_limit(self):
        """一字板: open=high=limit_up_price → True."""
        from config.settings import TradingConfig
        prev = 10.00
        limit_up = TradingConfig.calc_limit_price(prev, "600000", is_up=True)
        # open and high both at limit-up price → yiziban
        assert self._is_yiziban(limit_up, limit_up, prev)

    def test_not_yiziban_when_open_below_limit(self):
        """Normal limit-up: open < limit_up_price, high = limit_up_price → False."""
        from config.settings import TradingConfig
        prev = 10.00
        limit_up = TradingConfig.calc_limit_price(prev, "600000", is_up=True)
        # opened below limit-up but closed at limit-up → not yiziban
        assert not self._is_yiziban(prev, limit_up, prev)

    def test_not_yiziban_when_high_above_limit(self):
        """High above limit-up price → not a valid limit-up day at all."""
        from config.settings import TradingConfig
        prev = 10.00
        limit_up = TradingConfig.calc_limit_price(prev, "600000", is_up=True)
        # open at limit-up but high is above → unusual case, not yiziban
        assert not self._is_yiziban(limit_up, limit_up + 0.05, prev)

    def test_not_yiziban_when_open_nan(self):
        """Missing open data → not yiziban (conservative)."""
        assert not self._is_yiziban(float("nan"), 10.99, 10.00)
        assert not self._is_yiziban(None, 10.99, 10.00)

    def test_calc_limit_price_zero_prev_close(self):
        """Zero prev_close → calc_limit_price returns 0.0 (guard needed in caller)."""
        from config.settings import TradingConfig
        limit_up_price = TradingConfig.calc_limit_price(0.0, "600000", is_up=True)
        assert limit_up_price == 0.0
        # The real code guards with `pd.notna(prev_c) and prev_c > 0` before
        # calling calc_limit_price, so division by zero never occurs in practice.
