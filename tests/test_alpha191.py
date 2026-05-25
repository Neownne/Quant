"""Alpha191 因子测试。"""
import pytest
import pandas as pd
import numpy as np
from factors.engine import FactorEngine


def _make_ohlcv(n_days: int = 200) -> pd.DataFrame:
    """构造单只股票的 OHLCV + turnover 模拟数据。"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    close = 10 + np.cumsum(np.random.randn(n_days) * 0.3)
    volume = np.random.randint(5000000, 50000000, n_days).astype(float)
    return pd.DataFrame({
        "code": "000001",
        "trade_date": dates,
        "open": close * (1 + np.random.randn(n_days) * 0.005),
        "high": close * (1 + abs(np.random.randn(n_days)) * 0.015),
        "low": close * (1 - abs(np.random.randn(n_days)) * 0.015),
        "close": close,
        "volume": volume,
        "amount": volume * close,
        "turnover": np.random.uniform(0.005, 0.05, n_days),
        "float_share_ratio": np.random.uniform(0.3, 0.9, n_days),
    })


class TestAlpha191Turnover:
    def test_all_registered(self):
        """所有换手率因子应在 ALL_FACTORS 中注册。"""
        from factors import ALL_FACTORS
        expected = [
            "turnover_skew", "turnover_cv", "turnover_ma_dev",
            "turnover_ret_corr", "free_turnover_ratio",
        ]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        """因子输出应为有限值 Series。"""
        from factors import ALL_FACTORS
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "turnover_skew", "turnover_cv", "turnover_ma_dev",
            "turnover_ret_corr", "free_turnover_ratio",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.5, f"{col} 有效值仅 {valid_pct:.1%}"


class TestAlpha191Intraday:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = ["upper_shadow", "lower_shadow", "body_ratio", "intra_day_rev"]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors import ALL_FACTORS
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "upper_shadow", "lower_shadow", "body_ratio", "intra_day_rev",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.5, f"{col} 有效值仅 {valid_pct:.1%}"

    def test_upper_shadow_negative_means_selling_pressure(self):
        """上影线长 = 卖压大，应为负值信号。"""
        from factors.alpha191_intraday import upper_shadow
        df = _make_ohlcv(200)
        df["high"] = df[["open", "close"]].max(axis=1) + 1.0
        df["low"] = df[["open", "close"]].min(axis=1) - 0.1
        result = upper_shadow(df)
        assert result.dropna().mean() > 0.5


class TestAlpha191Flow:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = [
            "money_flow", "obv_roc", "force_index",
            "cwt", "volume_climax", "vwap_momentum",
        ]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "money_flow", "obv_roc", "force_index",
            "cwt", "volume_climax", "vwap_momentum",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.5, f"{col} 有效值仅 {valid_pct:.1%}"


class TestAlpha191Gap:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = ["overnight_ret", "overnight_ret_std", "open_auction_jump", "gap_ma_dev"]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "overnight_ret", "overnight_ret_std", "open_auction_jump", "gap_ma_dev",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.5, f"{col} 有效值仅 {valid_pct:.1%}"


class TestAlpha191Vol:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = [
            "vol_of_vol", "down_vol_ratio", "tail_risk",
            "beta_20", "ret_asymmetry",
        ]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "vol_of_vol", "down_vol_ratio", "tail_risk",
            "beta_20", "ret_asymmetry",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.3, f"{col} 有效值仅 {valid_pct:.1%}"


class TestAlpha191Liquidity:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = [
            "amihud_5", "dollar_volume", "turnover_breakout", "bid_ask_proxy",
        ]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "amihud_5", "dollar_volume", "turnover_breakout", "bid_ask_proxy",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.3, f"{col} 有效值仅 {valid_pct:.1%}"
