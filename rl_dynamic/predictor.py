"""RL 动态因子权重预测器 — 用训练好的 PPO 直接推理。"""
import numpy as np
import pandas as pd


class RLDynamicPredictor:
    """PPO 模型 → 因子权重 → 加权打分。

    实现标准 predict() 接口。
    """

    def __init__(self, ppo_model, factor_names: list[str], builder, device: str = "cpu"):
        self.ppo_model = ppo_model
        self.factor_names = list(factor_names)
        self.builder = builder
        self.device = device

    def predict(self, factor_df: pd.DataFrame,
                market_state: np.ndarray = None) -> pd.DataFrame:
        if factor_df.empty:
            return pd.DataFrame(columns=["code", "score", "rank"])

        if market_state is not None:
            action, _ = self.ppo_model.predict(market_state, deterministic=True)
            weights = np.clip(action, 0, None)
            weights = weights / max(weights.sum(), 1e-10)
        else:
            weights = np.ones(len(self.factor_names)) / len(self.factor_names)

        cols = [f for f in self.factor_names if f in factor_df.columns]
        col_idx = [i for i, f in enumerate(self.factor_names) if f in cols]
        X = factor_df[cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        w = np.array([weights[i] for i in col_idx], dtype=np.float32)
        w = w / max(w.sum(), 1e-10)
        scores = X @ w

        result = pd.DataFrame({"code": factor_df["code"].values, "score": scores})
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result
