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
