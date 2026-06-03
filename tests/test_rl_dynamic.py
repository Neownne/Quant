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


import torch
from rl_dynamic.policy_net import FactorWeightNet


class TestFactorWeightNet:
    def test_output_shape(self):
        net = FactorWeightNet(state_dim=50, n_factors=10)
        x = torch.randn(4, 50)
        out = net(x)
        assert out.shape == (4, 10)

    def test_weights_sum_to_one(self):
        net = FactorWeightNet(state_dim=30, n_factors=8)
        x = torch.randn(3, 30)
        out = net(x)
        for i in range(3):
            assert abs(out[i].sum().item() - 1.0) < 0.001

    def test_output_range(self):
        net = FactorWeightNet(state_dim=20, n_factors=5)
        x = torch.randn(10, 20)
        out = net(x)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_gradient_flow(self):
        net = FactorWeightNet(state_dim=10, n_factors=3)
        net.train()
        x = torch.randn(4, 10, requires_grad=True)
        out = net(x)
        loss = out.mean()
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"


from rl_dynamic.env import WeightLearningEnv
from rl_dynamic.trainer import _build_daily_data, walk_forward_train_rl_weights
from rl_dynamic.state_builder import StateBuilder
import numpy as np
import pandas as pd


from rl_dynamic.predictor import RLDynamicPredictor
import numpy as np


class TestPredictor:
    """RLDynamicPredictor 测试。"""

    def test_predict_returns_dataframe(self):
        from rl_dynamic.policy_net import FactorWeightNet
        from rl_dynamic.state_builder import StateBuilder
        net = FactorWeightNet(state_dim=10, n_factors=3)
        builder = StateBuilder(n_factors=3)
        predictor = RLDynamicPredictor(net, ["f0", "f1", "f2"], builder)

        df = pd.DataFrame({
            "code": ["000001", "000002", "000003"],
            "f0": [1.0, 2.0, 0.5],
            "f1": [-0.5, 0.3, 1.2],
            "f2": [0.2, 0.8, -0.3],
        })
        result = predictor.predict(df)
        assert "code" in result.columns
        assert "score" in result.columns
        assert "rank" in result.columns
        assert len(result) == 3
        assert result["rank"].min() == 1

    def test_predict_sorted_descending(self):
        from rl_dynamic.policy_net import FactorWeightNet
        from rl_dynamic.state_builder import StateBuilder
        net = FactorWeightNet(state_dim=5, n_factors=2)
        builder = StateBuilder(n_factors=2)
        predictor = RLDynamicPredictor(net, ["f0", "f1"], builder)

        df = pd.DataFrame({
            "code": ["A", "B", "C"],
            "f0": [10.0, 1.0, 5.0],
            "f1": [0.1, 10.0, 0.1],
        })
        result = predictor.predict(df)
        scores = result["score"].values
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_predict_with_market_state(self):
        from rl_dynamic.policy_net import FactorWeightNet
        from rl_dynamic.state_builder import StateBuilder
        net = FactorWeightNet(state_dim=10, n_factors=3)
        builder = StateBuilder(n_factors=3)
        predictor = RLDynamicPredictor(net, ["f0", "f1", "f2"], builder)

        df = pd.DataFrame({
            "code": ["000001", "000002"],
            "f0": [1.0, -1.0],
            "f1": [0.5, 0.0],
            "f2": [0.3, 0.3],
        })
        state = np.zeros(10, dtype=np.float32)
        result = predictor.predict(df, market_state=state)
        assert len(result) == 2
        assert "score" in result.columns

    def test_empty_dataframe(self):
        from rl_dynamic.policy_net import FactorWeightNet
        from rl_dynamic.state_builder import StateBuilder
        net = FactorWeightNet(state_dim=5, n_factors=2)
        builder = StateBuilder(n_factors=2)
        predictor = RLDynamicPredictor(net, ["f0", "f1"], builder)
        df = pd.DataFrame(columns=["code", "f0", "f1"])
        result = predictor.predict(df)
        assert result.empty


class TestEnv:
    """WeightLearningEnv 测试。"""

    def test_env_creation(self):
        """测试环境创建与 reset/step 基本流程。"""
        builder = StateBuilder(n_factors=3)
        pool = FactorPool(["rsi_7", "vol_20", "mom_20"])
        state = np.zeros(builder.state_dim, dtype=np.float32)
        matrix = np.random.randn(20, 3).astype(np.float32)
        rets = np.random.randn(20).astype(np.float32)
        data = {"2026-01-01": {"state": state, "factor_matrix": matrix, "returns": rets}}
        env = WeightLearningEnv(builder, pool, data, n_factors=3)
        obs, _ = env.reset()
        assert obs.shape == (builder.state_dim,)
        action = np.array([0.3, 0.5, 0.2], dtype=np.float32)
        obs2, reward, term, trunc, info = env.step(action)
        assert term is True
        assert isinstance(reward, float)

    def test_env_reward_with_uniform_weights(self):
        """测试正相关因子高权重获得更高奖励。"""
        builder = StateBuilder(n_factors=2)
        pool = FactorPool(["rsi_7", "vol_20"])
        state = np.zeros(builder.state_dim, dtype=np.float32)
        # 构造15只股票: factor0与收益正相关，factor1与收益负相关
        np.random.seed(42)
        n_stocks = 15
        factor0 = np.linspace(-3, 3, n_stocks, dtype=np.float32)
        factor1 = -factor0
        matrix = np.column_stack([factor0, factor1])
        # 收益与factor0正相关
        rets = (factor0 * 0.02 + np.random.randn(n_stocks).astype(np.float32) * 0.005)
        data = {"d": {"state": state, "factor_matrix": matrix, "returns": rets}}
        env = WeightLearningEnv(builder, pool, data, n_factors=2)
        env.reset()
        # 给factor0高权重（正相关）
        _, r1, _, _, _ = env.step(np.array([0.9, 0.1], dtype=np.float32))
        # 给factor1高权重（负相关）
        env.reset()
        _, r2, _, _, _ = env.step(np.array([0.1, 0.9], dtype=np.float32))
        assert r1 > r2, f"正相关权重应得更高奖励, r1={r1}, r2={r2}"


class TestTrainer:
    """Trainer 测试。"""

    def test_build_daily_data(self):
        """测试 _build_daily_data 构建每日数据字典。"""
        dates = pd.date_range("2026-01-02", periods=30, freq="B")
        n = len(dates) * 10
        ohlcv = pd.DataFrame({
            "code": [f"{600000+i:06d}" for i in range(10)] * len(dates),
            "trade_date": np.repeat(dates, 10),
            "close": np.random.randn(n).cumsum() + 500,
            "open": np.random.randn(n).cumsum() + 500,
            "high": np.random.randn(n).cumsum() + 505,
            "low": np.random.randn(n).cumsum() + 495,
            "volume": np.random.rand(n) * 1e8,
            "amount": np.random.rand(n) * 1e9,
            "turnover": np.random.rand(n) * 0.05,
        })
        idx = pd.DataFrame({
            "trade_date": dates,
            "close": 3000 + np.cumsum(np.random.randn(len(dates)) * 10),
        })
        pool = FactorPool(["rsi_7", "vol_20", "mom_20"])
        builder = StateBuilder(n_factors=pool.n_factors)
        ds = pool.compute_factors(ohlcv, None)
        ds["trade_date"] = pd.to_datetime(ds["trade_date"])
        daily = _build_daily_data(ds, builder, pool, ohlcv, idx)
        assert len(daily) > 0
        first_key = list(daily.keys())[0]
        assert "state" in daily[first_key]
        assert "factor_matrix" in daily[first_key]
        assert "returns" in daily[first_key]
