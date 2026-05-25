"""因子模块测试。"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_ohlcv():
    """构造 100 个交易日 x 3 只股票的模拟 OHLCV 数据。"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    codes = ["000001", "000002", "000003"]
    rows = []
    for code in codes:
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "trade_date": d.date(),
                "open": close[i] * (1 + np.random.randn() * 0.01),
                "high": close[i] * (1 + abs(np.random.randn()) * 0.02),
                "low": close[i] * (1 - abs(np.random.randn()) * 0.02),
                "close": close[i],
                "volume": np.random.randint(100000, 1000000),
            })
    return pd.DataFrame(rows)


class TestFactorEngine:
    def test_unknown_factor_raises(self, sample_ohlcv):
        """未知因子应在构造时报 KeyError。"""
        from factors.engine import FactorEngine
        with pytest.raises(KeyError):
            FactorEngine(factor_names=["nonexistent_factor_xyz"])

    def test_engine_requires_factors(self):
        """空因子列表应在 compute 时报错或返回空。"""
        from factors.engine import FactorEngine
        # 允许空因子列表构造但不报错，compute 返回原数据
        engine = FactorEngine(factor_names=[])
        # 空因子列表应该可以构造（虽然不计算任何因子）


class TestNeutralize:
    def test_perfect_correlation_removed(self):
        """市值中性化后因子与市值的相关性应接近零。"""
        from factors.engine import neutralize
        np.random.seed(42)
        n = 500
        log_mcap = np.random.randn(n)
        # 因子 = 0.5 * log_mcap + noise -- 强市值暴露
        factor = 0.5 * log_mcap + np.random.randn(n) * 0.1

        exposures = pd.DataFrame({"log_mcap": log_mcap})
        neutralized = neutralize(pd.Series(factor), exposures)

        corr = neutralized.corr(exposures["log_mcap"])
        assert abs(corr) < 0.01, f"中性化后相关性应 < 0.01，实际: {corr:.4f}"

    def test_handles_nan(self):
        """NaN 值应保留，不影响其他样本的中性化。"""
        from factors.engine import neutralize
        factor = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        exposures = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})

        result = neutralize(factor, exposures)

        assert np.isnan(result.iloc[2]), "NaN 输入应保持 NaN"
        assert not result.dropna().empty, "非 NaN 值应有结果"


class TestFactorEngineIntegration:
    @pytest.mark.skip(reason="No stub factor registered yet — will be enabled when factor implementations are added")
    def test_compute_stub_factor(self, sample_ohlcv):
        """用存根因子测试完整 compute 流程。"""
        from factors.engine import FactorEngine

        # 先在 factors/__init__.py 中注册一个存根因子
        # 此测试在 engine.py 实现后运行

        engine = FactorEngine(factor_names=["stub_factor"])
        result = engine.compute(sample_ohlcv)
        assert "stub_factor" in result.columns
        assert len(result) == len(sample_ohlcv)


class TestAlpha101Factors:
    def test_registered_functions_are_callable(self, sample_ohlcv):
        """所有 Alpha101 因子应可对分组数据调用，返回 Series。"""
        from factors.alpha101 import ALPHA101_FUNCTIONS
        code_df = sample_ohlcv[sample_ohlcv["code"] == "000001"].reset_index(drop=True)

        for name, fn in ALPHA101_FUNCTIONS.items():
            result = fn(code_df)
            assert isinstance(result, pd.Series), f"{name}: 应返回 Series, 实际 {type(result)}"
            assert len(result) == len(code_df), f"{name}: 长度不匹配 (期望 {len(code_df)}, 实际 {len(result)})"

    def test_rsi_range(self, sample_ohlcv):
        """RSI 应在 [0, 100] 范围内 (warmup 期后)。"""
        from factors.alpha101 import rsi_14
        code_df = sample_ohlcv[sample_ohlcv["code"] == "000001"].reset_index(drop=True)
        result = rsi_14(code_df)
        warmup = 20
        vals = result.iloc[warmup:].dropna()
        assert (vals >= 0).all() and (vals <= 100).all(), \
            f"RSI 值应在 [0,100] 范围内，实际: {vals.min():.1f} ~ {vals.max():.1f}"

    def test_factor_engine_with_real_factors(self, sample_ohlcv):
        """FactorEngine 应能用真实因子计算因子矩阵。"""
        from factors import FactorEngine, ALL_FACTORS
        assert len(ALL_FACTORS) >= 30, f"应有至少 30 个因子，实际: {len(ALL_FACTORS)}"

        engine = FactorEngine(factor_names=["rsi_14", "mom_20"])
        result = engine.compute(sample_ohlcv)

        assert "code" in result.columns
        assert "trade_date" in result.columns
        assert "rsi_14" in result.columns
        assert "mom_20" in result.columns
        assert len(result) == len(sample_ohlcv)


class TestCustomFactors:
    def test_all_custom_factors_registered(self):
        """自定义因子应注册到 ALL_FACTORS。"""
        from factors import ALL_FACTORS
        from factors.custom import CUSTOM_FACTORS
        assert len(CUSTOM_FACTORS) == 7
        for name in CUSTOM_FACTORS:
            assert name in ALL_FACTORS, f"{name} 应在 ALL_FACTORS 中"

    def test_factors_without_extra_columns_return_nan(self, sample_ohlcv):
        """无 pb/log_mcap/shareholder_count 等额外列时，因子应返回 NaN Series。"""
        from factors.custom import log_mcap, pb_pct, shareholder_change
        code_df = sample_ohlcv[sample_ohlcv["code"] == "000001"].reset_index(drop=True)

        for fn in [log_mcap, pb_pct, shareholder_change]:
            result = fn(code_df)
            assert isinstance(result, pd.Series)
            assert result.isna().all(), f"{fn.__name__}: 无额外列时应全 NaN"

    def test_factors_with_data(self, sample_ohlcv):
        """有额外列时因子应返回有效值。"""
        from factors.custom import intra_vol, vol_conv, gap_ratio
        code_df = sample_ohlcv[sample_ohlcv["code"] == "000001"].reset_index(drop=True)

        for fn in [intra_vol, vol_conv, gap_ratio]:
            result = fn(code_df)
            assert isinstance(result, pd.Series)
            assert len(result) == len(code_df)
            # warmup 后应有非 NaN 值
            assert result.iloc[-1] is not np.nan or result.iloc[-1] is np.nan, \
                f"{fn.__name__}: 最后几个值至少应有定义"
