"""RL 因子权重学习环境。"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WeightLearningEnv(gym.Env):
    """每个episode学习一天的最优因子权重。

    State: StateBuilder输出的市场状态向量
    Action: N维因子权重(连续, softmax归一化)
    Reward: 加权选股后的组合日收益
    """

    def __init__(self, builder, pool, daily_data: dict, n_factors: int = 10):
        super().__init__()
        self.builder = builder
        self.pool = pool
        self.daily_data = daily_data  # {date_str: {"state": np.array, "factor_matrix": np.array, "returns": np.array}}
        self.dates = sorted(daily_data.keys())
        self.n_factors = n_factors
        self.state_dim = self.builder.state_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_factors,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, len(self.dates))
        date = self.dates[idx]
        data = self.daily_data[date]
        self._current_data = data
        return data["state"].astype(np.float32), {}

    def step(self, action):
        weights = np.clip(action, 0.0, 1.0)
        weights = weights / max(weights.sum(), 1e-10)

        factor_matrix = self._current_data.get("factor_matrix")
        returns = self._current_data.get("returns", np.array([0]))

        if factor_matrix is not None and len(factor_matrix) > 0:
            # 加权打分: scores = factor_matrix @ weights
            scores = factor_matrix @ weights
            # 选Top-10
            n_select = min(10, len(scores))
            top_idx = np.argsort(scores)[-n_select:]
            reward = float(returns[top_idx].mean()) if len(top_idx) > 0 else 0.0
        else:
            reward = 0.0

        terminated = True
        truncated = False
        info = {"weights": weights, "reward": reward}

        return (np.zeros(self.state_dim, dtype=np.float32),
                float(reward), terminated, truncated, info)
