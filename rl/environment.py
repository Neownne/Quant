"""RL 环境：点式股票评分 Gymnasium 环境。

StockScoringEnv — 逐股打分，模仿现有 predict() 接口的行为。
每个 episode = 一个交易日的所有候选股票。
状态 = 因子值 + 市场环境特征
动作 = [0, 1] 连续评分
奖励 = 方向正确性 × 收益幅度
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StockScoringEnv(gym.Env):
    """点式股票评分环境。

    逐股处理：每步对一只股票打分，episode 结束得到完整排序。
    与现有 EnsemblePredictor.predict() 的行为等价 —— 输出连续评分。

    观测空间：因子值 + 市场均值/标准差 + 板块编码
    动作空间：[0, 1] 连续值
    """

    def __init__(
        self,
        factor_df,
        factor_names: list[str],
        context_features: list[str] | None = None,
    ):
        super().__init__()

        self.factor_df = factor_df.reset_index(drop=True)
        self.factor_names = [f for f in factor_names if f in self.factor_df.columns]

        # 市场环境特征：从因子数据中提取截面统计量
        self.n_factors = len(self.factor_names)
        self.n_context = 4  # mean, std, skew, kurt of key factors + market return

        # 观测 = 因子值 + 截面统计量
        obs_dim = self.n_factors + self.n_context

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        self._current_idx = 0
        self._n_stocks = len(self.factor_df)
        self._context = self._compute_context()

    def _compute_context(self) -> np.ndarray:
        """计算截面市场环境特征。"""
        ctx = np.zeros(self.n_context, dtype=np.float32)
        if self.n_factors > 0:
            vals = self.factor_df[self.factor_names].values
            ctx[0] = float(np.nanmean(vals))
            ctx[1] = float(np.nanstd(vals))
            if self.n_factors >= 3:
                ctx[2] = float(np.nanmean(np.sign(vals) * np.abs(vals) ** 2))
            else:
                ctx[2] = 0.0
            ctx[3] = float(len(self.factor_df))  # 股票数量
        return ctx

    def _get_obs(self) -> np.ndarray:
        """获取当前股票的观测。"""
        row = self.factor_df.iloc[self._current_idx]
        factor_vals = row[self.factor_names].fillna(0).values.astype(np.float32)
        return np.concatenate([factor_vals, self._context])

    def _get_label(self) -> float:
        """获取当前股票的标签（1=上涨，0=下跌）。"""
        if "label" in self.factor_df.columns:
            return float(self.factor_df.iloc[self._current_idx]["label"])
        return 0.5  # 默认

    def reset(self, seed=None, options=None):
        """重置环境，开始新的 episode。"""
        super().reset(seed=seed)
        self._current_idx = 0
        self._context = self._compute_context()
        obs = self._get_obs()
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray):
        """对当前股票评分并前进到下一只。

        参数
        ----
        action : [score] 评分 ∈ [0, 1]

        返回
        ----
        obs, reward, terminated, truncated, info
        """
        score = float(np.clip(action[0], 0.0, 1.0))
        label = self._get_label()
        ret = 0.01  # 默认收益幅度

        if "ret_1d" in self.factor_df.columns:
            ret = abs(float(self.factor_df.iloc[self._current_idx]["ret_1d"]))
            ret = max(ret, 0.001)

        # 奖励：方向正确性 × 收益幅度
        # score > 0.5 表示预测上涨，label=1 表示实际上涨
        direction_signal = (score - 0.5) * (2.0 * label - 1.0)
        reward = float(direction_signal * ret)

        self._current_idx += 1
        terminated = self._current_idx >= self._n_stocks
        truncated = False

        if terminated:
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        else:
            obs = self._get_obs().astype(np.float32)

        info = {
            "score": score,
            "label": label,
            "idx": self._current_idx - 1,
        }
        return obs, reward, terminated, truncated, info
