"""RL动态因子权重预测器。"""
import numpy as np
import pandas as pd
import torch


class RLDynamicPredictor:
    """用RL学习到的因子权重给股票打分。

    实现标准 predict() 接口，与 EnsemblePredictor 兼容。
    """

    def __init__(self, policy_net, factor_names: list[str],
                 builder, device: str = "cpu"):
        self.net = policy_net
        self.factor_names = list(factor_names)
        self.builder = builder
        self.device = device
        self.net.to(device)
        self.net.eval()

    def predict(self, factor_df: pd.DataFrame,
                market_state: np.ndarray = None) -> pd.DataFrame:
        """对股票打分排序。

        Args:
            factor_df: 含 code + 因子列
            market_state: 市场状态向量（可选，None则等权）

        Returns:
            DataFrame[code, score, rank] 按score降序
        """
        if factor_df.empty:
            return pd.DataFrame(columns=["code", "score", "rank"])

        # 获取因子权重
        if market_state is not None:
            state_t = torch.tensor(market_state, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
            with torch.no_grad():
                weights = self.net(state_t).squeeze(0).cpu().numpy()
        else:
            weights = np.ones(len(self.factor_names)) / len(self.factor_names)

        # 构建因子矩阵
        cols = [f for f in self.factor_names if f in factor_df.columns]
        col_indices = [i for i, f in enumerate(self.factor_names) if f in cols]
        X = factor_df[cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        w = np.array([weights[i] for i in col_indices], dtype=np.float32)
        w = w / max(w.sum(), 1e-10)

        # 加权打分
        scores = X @ w

        result = pd.DataFrame({"code": factor_df["code"].values, "score": scores})
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result
