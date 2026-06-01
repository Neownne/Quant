"""RL 板块打分环境 — 学习恐贪指标到板块收益的映射。

SectorScoringEnv: 每步给一个板块打分，episode遍历所有板块。
状态 = 5项恐贪子指标 + 市场环境
动作 = [0,1] 连续评分
奖励 = (score - 0.5) × direction × |return|
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SectorScoringEnv(gym.Env):
    """板块恐贪打分环境。

    观测空间：5维恐贪指标 + 4维市场环境 = 9维
    动作空间：[0, 1] 连续评分
    """

    FG_FEATURES = [
        "fg_volatility", "fg_money_flow", "fg_momentum",
        "fg_new_high_ratio", "fg_advance_decline",
    ]
    N_CONTEXT = 4  # market mean ret, mean vol, n_sectors, n_stocks

    def __init__(self, daily_data: dict, device: str = "cpu"):
        """
        daily_data: {date: {sector: {fg_*, ret_nd}}}
        """
        super().__init__()
        self.daily_data = daily_data
        self.dates = sorted(daily_data.keys())
        self.n_features = len(self.FG_FEATURES)

        obs_dim = self.n_features + self.N_CONTEXT
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        self._current_date_idx = 0
        self._current_sector_idx = 0
        self._sectors = []
        self._current_sectors = {}

    def _get_obs(self) -> np.ndarray:
        sector = self._sectors[self._current_sector_idx]
        day_data = self._current_sectors[sector]
        feats = np.array([day_data.get(f, 0.0) for f in self.FG_FEATURES], dtype=np.float32)

        # 市场环境：截面统计量
        all_fg = []
        for s in self._sectors:
            all_fg.append([self._current_sectors[s].get(f, 0.0) for f in self.FG_FEATURES])
        all_fg = np.array(all_fg)
        ctx = np.array([
            float(np.mean(all_fg[:, 2])),   # mean momentum
            float(np.std(all_fg[:, 2])),    # momentum dispersion
            float(len(self._sectors)),       # n_sectors
            float(np.mean(all_fg)),          # mean FG
        ], dtype=np.float32)
        return np.concatenate([feats, ctx])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, len(self.dates))
        # idx = self._current_date_idx  # sequential mode
        date = self.dates[idx]
        self._current_sectors = self.daily_data[date]
        self._sectors = sorted(self._current_sectors.keys())
        self._current_sector_idx = 0
        return self._get_obs(), {}

    def step(self, action):
        score = float(np.clip(action[0], 0.0, 1.0))
        sector = self._sectors[self._current_sector_idx]
        ret = self._current_sectors[sector].get("ret_nd", 0.01)
        ret = max(abs(ret), 0.001) * (1 if ret >= 0 else -1)

        # 奖励：高分 → 买高收益板块
        reward = float((score - 0.5) * np.sign(ret) * abs(ret) * 10)

        self._current_sector_idx += 1
        terminated = self._current_sector_idx >= len(self._sectors)

        if terminated:
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, reward, terminated, False, {"sector": sector, "score": score}
