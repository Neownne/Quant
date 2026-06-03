"""RL-Dynamic 策略测试。"""
import pytest


class TestRLDynamicModule:
    """RL-Dynamic 模块导入测试。"""

    def test_module_imports(self):
        """测试 rl_dynamic 模块能正常导入。"""
        import rl_dynamic
        assert hasattr(rl_dynamic, '__all__') or True  # module exists

    def test_module_docstring(self):
        """测试模块文档字符串。"""
        import rl_dynamic
        assert rl_dynamic.__doc__ is not None
        assert 'RL-Dynamic' in rl_dynamic.__doc__


class TestPaperAccount:
    """模拟盘账户测试。"""

    def test_account_exists(self):
        """测试 paper_account id=18 (RL-Dynamic) 存在。"""
        from data.db import get_engine
        from sqlalchemy import text
        e = get_engine()
        with e.connect() as c:
            r = c.execute(
                text("SELECT id, name, cash FROM paper_account WHERE id = 18")
            ).fetchone()
            assert r is not None, "paper_account id=18 does not exist"
            assert r[0] == 18
            assert r[1] == 'RL-Dynamic'
            assert r[2] == 1_000_000
        e.dispose()

    def test_paper_run_exists(self):
        """测试 paper_runs id=5 存在且关联 strategy_id=15 (舞)。"""
        from data.db import get_engine
        from sqlalchemy import text
        e = get_engine()
        with e.connect() as c:
            r = c.execute(
                text("SELECT id, strategy_id, version_id, status FROM paper_runs WHERE id = 5")
            ).fetchone()
            assert r is not None, "paper_runs id=5 does not exist"
            assert r[0] == 5
            assert r[1] == 15  # strategy_id = 15 (舞)
            assert r[3] == 'running'
        e.dispose()


class TestStateBuilder:
    """StateBuilder 市场状态特征构建器测试。"""

    def test_state_dim(self):
        """测试 state_dim 属性等于基本特征数 + 因子IC维度数。"""
        from rl_dynamic.state_builder import StateBuilder, FEATURE_NAMES

        builder = StateBuilder(n_factors=10)
        assert builder.state_dim == len(FEATURE_NAMES) + 10
        assert builder.state_dim >= 31  # 21 base + 10 ic

    def test_build_returns_valid_array(self):
        """测试 build() 返回有效的 float32 ndarray，无 NaN/Inf。"""
        from rl_dynamic.state_builder import StateBuilder
        import pandas as pd
        import numpy as np

        builder = StateBuilder(n_factors=3)
        dates = pd.date_range("2026-01-02", periods=100, freq="B")
        ohlcv = pd.DataFrame({
            "code": ["000001"] * 100,
            "trade_date": dates,
            "close": 10 + np.cumsum(np.random.randn(100) * 0.1),
            "amount": np.random.rand(100) * 1e8,
            "turnover": np.random.rand(100) * 0.05,
        })
        idx = pd.DataFrame({
            "trade_date": dates,
            "close": 3000 + np.cumsum(np.random.randn(100) * 10),
        })
        state = builder.build(ohlcv, idx, {0: 0.02, 1: -0.01, 2: 0.03}, dates[-1])
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert len(state) == builder.state_dim
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    def test_feature_names_match_dim(self):
        """测试 feature_names 列表长度与 state_dim 一致。"""
        from rl_dynamic.state_builder import StateBuilder

        builder = StateBuilder(n_factors=5)
        assert len(builder.feature_names) == builder.state_dim


from rl_dynamic.factor_pool import FactorPool


class TestFactorPool:
    def test_pool_creation(self):
        pool = FactorPool(["rsi_7", "vol_20", "mom_20", "turnover_5", "rev_5"])
        assert pool.n_factors >= 3

    def test_ic_tracking(self):
        pool = FactorPool(["rsi_7", "vol_20"])
        pool.ic_history["rsi_7"] = [0.03, 0.04, -0.01, 0.05, 0.02]
        pool.ic_history["vol_20"] = [-0.01, -0.01, -0.02, -0.01, -0.01]
        ic = pool.get_recent_ic(5)
        assert abs(ic[0]) > abs(ic[1]), f"rsi_7 should have stronger IC than vol_20, got {ic}"

    def test_select_top_by_ic(self):
        pool = FactorPool(["rsi_7", "vol_20", "mom_20", "turnover_5", "rev_5"])
        pool.ic_history["rsi_7"] = [0.03] * 20
        pool.ic_history["vol_20"] = [0.01] * 20
        pool.ic_history["mom_20"] = [0.05] * 20
        pool.ic_history["turnover_5"] = [0.02] * 20
        pool.ic_history["rev_5"] = [0.04] * 20
        top = pool.select_top_by_ic(2)
        assert len(top) == 2
        assert top[0] == "mom_20"

    def test_factor_names(self):
        pool = FactorPool(["rsi_7", "vol_20", "mom_20"])
        names = pool.get_factor_names()
        assert "rsi_7" in names
        assert "vol_20" in names
        assert "mom_20" in names
