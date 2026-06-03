"""因子池管理：维护因子IC追踪、筛选最优因子。"""
import numpy as np
import pandas as pd
from factors import ALL_FACTORS


class FactorPool:
    """管理因子列表并追踪IC表现。"""

    def __init__(self, factor_names: list[str] = None):
        all_names = factor_names or list(ALL_FACTORS.keys())
        self.all_factors = [f for f in all_names if f in ALL_FACTORS]
        self.n_factors = len(self.all_factors)
        self.ic_history: dict[str, list[float]] = {f: [] for f in self.all_factors}

    def compute_factors(self, ohlcv: pd.DataFrame, extra_data=None) -> pd.DataFrame:
        """计算因子矩阵。复用现有 build_factor_dataset。"""
        from models.dataset import build_factor_dataset
        return build_factor_dataset(
            ohlcv, self.all_factors, label_mode="binary",
            forward_days=5, extra_data=extra_data,
        )

    def update_ic(self, dataset: pd.DataFrame):
        """从数据集更新每个因子的IC追踪。"""
        for f in self.all_factors:
            if f not in dataset.columns:
                continue
            valid = dataset[[f, "ret_1d"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 10:
                continue
            ic = valid[f].corr(valid["ret_1d"], method="spearman")
            self.ic_history[f].append(float(ic) if not np.isnan(ic) else 0.0)

    def get_recent_ic(self, n: int = 20) -> dict[int, float]:
        """获取每个因子最近N日平均IC，按因子索引返回。"""
        result = {}
        for i, f in enumerate(self.all_factors):
            hist = self.ic_history.get(f, [])
            if hist:
                result[i] = float(np.mean(hist[-n:]))
            else:
                result[i] = 0.0
        return result

    def select_top_by_ic(self, n: int = 10) -> list[str]:
        """选IC绝对值最强的N个因子。"""
        scores = {f: abs(np.mean(h[-20:])) if h[-20:] else 0
                  for f, h in self.ic_history.items() if h}
        sorted_f = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in sorted_f[:n]]

    def get_factor_names(self) -> list[str]:
        return list(self.all_factors)
