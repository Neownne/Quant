"""RL 板块打分预测器 — 替代原来的 ML/动量板块打分。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from rl.models import StockScoreNet  # 复用同一网络架构


class RLSectorPredictor:
    """用训练好的 RL 策略网络给板块打分。

    实现与 SectorScoringModel 相同的接口。
    """

    FG_FEATURES = [
        "fg_volatility", "fg_money_flow", "fg_momentum",
        "fg_new_high_ratio", "fg_advance_decline",
    ]
    N_CONTEXT = 4

    def __init__(self, policy_net, device: str = "cpu"):
        self.policy_net = policy_net
        self.device = device
        self.policy_net.to(device)
        self.policy_net.eval()

    def predict(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """对板块特征 DataFrame 打分。

        参数: sector_df 须含 sector + fg_* 列
        返回: DataFrame[sector, score, rank] 按 score 降序
        """
        n_sectors = len(sector_df)
        if n_sectors == 0:
            return pd.DataFrame(columns=["sector", "score", "rank"])

        # 构建观测矩阵
        obs_list = []
        for _, row in sector_df.iterrows():
            feats = np.array([row.get(f, 0.0) for f in self.FG_FEATURES], dtype=np.float32)
            obs_list.append(feats)

        all_feats = np.array(obs_list)
        ctx = np.array([
            float(np.mean(all_feats[:, 2])),
            float(np.std(all_feats[:, 2])) if n_sectors > 1 else 0.0,
            float(n_sectors),
            float(np.mean(all_feats)),
        ], dtype=np.float32)
        ctx_tiled = np.tile(ctx, (n_sectors, 1))

        obs = np.concatenate([all_feats, ctx_tiled], axis=1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            scores = self.policy_net(obs_tensor).squeeze(-1).cpu().numpy()

        result = pd.DataFrame({
            "sector": sector_df["sector"].values,
            "score": scores,
        })
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result
