"""RL 环境测试。"""
import pytest
import numpy as np
import pandas as pd
from gymnasium import spaces


class TestStockScoringEnv:
    """StockScoringEnv 测试。"""

    @pytest.fixture
    def sample_data(self):
        """构造测试用因子数据。"""
        np.random.seed(42)
        n_stocks = 50
        n_factors = 10
        factor_names = [f"factor_{i}" for i in range(n_factors)]

        # 模拟因子值
        data = {}
        for f in factor_names:
            data[f] = np.random.randn(n_stocks) * 0.5
        data["code"] = [f"{i:06d}.SH" for i in range(n_stocks)]
        data["label"] = np.random.randint(0, 2, n_stocks).astype(float)
        data["ret_1d"] = np.random.randn(n_stocks) * 0.02

        df = pd.DataFrame(data)
        return df, factor_names

    def test_env_creation(self, sample_data):
        """环境应能成功创建。"""
        from rl.environment import StockScoringEnv

        df, factor_names = sample_data
        env = StockScoringEnv(df, factor_names)
        assert env is not None
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)

    def test_reset_returns_valid_observation(self, sample_data):
        """reset() 应返回符合观测空间的有效观测。"""
        from rl.environment import StockScoringEnv

        df, factor_names = sample_data
        env = StockScoringEnv(df, factor_names)
        obs, info = env.reset()

        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_step_returns_valid_transition(self, sample_data):
        """step() 应返回 (obs, reward, terminated, truncated, info)。"""
        from rl.environment import StockScoringEnv

        df, factor_names = sample_data
        env = StockScoringEnv(df, factor_names)
        obs, _ = env.reset()

        action = np.array([0.7], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape

    def test_episode_terminates_after_all_stocks(self, sample_data):
        """遍历完所有股票后 episode 应终止。"""
        from rl.environment import StockScoringEnv

        df, factor_names = sample_data
        env = StockScoringEnv(df, factor_names)
        obs, _ = env.reset()

        steps = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = np.array([0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if steps > 1000:
                break

        # 应该在处理完所有股票后终止
        assert steps == len(df)
        assert terminated

    def test_action_clipped_to_valid_range(self, sample_data):
        """动作应被限制在 [0, 1] 范围内。"""
        from rl.environment import StockScoringEnv

        df, factor_names = sample_data
        env = StockScoringEnv(df, factor_names)
        assert env.action_space.low[0] == 0.0
        assert env.action_space.high[0] == 1.0

    def test_observation_contains_factors_and_context(self, sample_data):
        """观测应包含因子值 + 市场环境。"""
        from rl.environment import StockScoringEnv

        df, factor_names = sample_data
        env = StockScoringEnv(df, factor_names)
        obs, _ = env.reset()

        # 观测维度 = 因子数 + 环境特征数
        n_factors = len(factor_names)
        assert len(obs) >= n_factors, f"期望至少 {n_factors} 维，实际 {len(obs)} 维"

    def test_reward_positive_for_correct_prediction(self, sample_data):
        """正确预测应获得正奖励。"""
        from rl.environment import StockScoringEnv

        # 构造简单数据：所有 label=1（上涨）
        n_stocks = 5
        df = pd.DataFrame({
            "factor_0": np.random.randn(n_stocks),
            "factor_1": np.random.randn(n_stocks),
            "code": [f"{i:06d}.SH" for i in range(n_stocks)],
            "label": [1.0] * n_stocks,  # 全部上涨
            "ret_1d": np.random.randn(n_stocks) * 0.01,
        })

        env = StockScoringEnv(df, ["factor_0", "factor_1"])
        obs, _ = env.reset()

        # 给高分（>0.5），期望正奖励（因为 label=1）
        action = np.array([0.9], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward > 0, f"高分应对应正奖励，实际 reward={reward}"

    def test_reward_negative_for_wrong_prediction(self, sample_data):
        """错误预测应获得负奖励。"""
        from rl.environment import StockScoringEnv

        n_stocks = 5
        df = pd.DataFrame({
            "factor_0": np.random.randn(n_stocks),
            "factor_1": np.random.randn(n_stocks),
            "code": [f"{i:06d}.SH" for i in range(n_stocks)],
            "label": [0.0] * n_stocks,  # 全部下跌
            "ret_1d": np.random.randn(n_stocks) * 0.01,
        })

        env = StockScoringEnv(df, ["factor_0", "factor_1"])
        obs, _ = env.reset()

        # 给高分但 label=0，期望负奖励
        action = np.array([0.9], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward < 0, f"高分+下跌应对应负奖励，实际 reward={reward}"
