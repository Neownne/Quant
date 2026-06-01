"""RL 预测器：实现标准 predict() 接口。

RLPredictor 是 EnsemblePredictor 的直接替代品。
它使用训练好的 StockScoreNet 策略网络进行批量推理，
返回 [code, score, rank] 格式的 DataFrame。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


class RLPredictor:
    """基于 StockScoreNet 的股票评分预测器。

    实现与 EnsemblePredictor 完全兼容的接口：
      predict(factor_df) -> DataFrame[code, score, rank]

    可直接替换 EnsemblePredictor，回测循环无需任何改动。
    """

    def __init__(
        self,
        policy_net: torch.nn.Module,
        factor_names: list[str],
        device: str = "cpu",
    ):
        self.policy_net = policy_net
        self.factor_names = list(factor_names)
        self.device = device

        # 确保网络在正确的设备和模式下
        self.policy_net.to(self.device)
        self.policy_net.eval()

    def predict(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """对一批股票打分并排序。

        参数
        ----
        factor_df : 至少含 code 列和因子列

        返回
        ----
        DataFrame: [code, score, rank]，按 score 降序排列
        """
        if factor_df.empty:
            return pd.DataFrame(columns=["code", "score", "rank"])

        # 构建状态矩阵
        n_stocks = len(factor_df)
        n_factors = len(self.factor_names)

        # 因子部分
        factor_array = np.zeros((n_stocks, n_factors), dtype=np.float32)
        for i, f in enumerate(self.factor_names):
            if f in factor_df.columns:
                col_vals = factor_df[f].fillna(0).values.astype(np.float32)
                factor_array[:, i] = col_vals

        # 环境特征：截面统计量
        factor_mean = factor_array.mean()
        factor_std = factor_array.std()
        factor_skew = np.mean(np.sign(factor_array) * np.abs(factor_array) ** 2) if n_factors >= 3 else 0.0
        n_context = np.array([float(n_stocks)], dtype=np.float32)
        context = np.array([factor_mean, factor_std, factor_skew, n_context[0]], dtype=np.float32)
        context_tiled = np.tile(context, (n_stocks, 1))

        # 合并观测
        obs = np.concatenate([factor_array, context_tiled], axis=1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # 批量推理
        with torch.no_grad():
            scores = self.policy_net(obs_tensor).squeeze(-1).cpu().numpy()

        result = pd.DataFrame({
            "code": factor_df["code"].values,
            "score": scores,
        })
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result
