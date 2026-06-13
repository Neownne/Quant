"""每日预测：加载模型，对最新截面做预测排序。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class DailyPredictor:
    """每日预测器。

    用法:
        predictor = DailyPredictor(model, factor_names=["rsi_14", "mom_20"])
        scores = predictor.predict(today_factors)  # → [code, score, rank]
    """

    def __init__(self, model, factor_names: list[str]):
        self.model = model
        self.factor_names = factor_names

    def predict(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """对横截面数据打分排序。

        参数
        ----
        factor_df : DataFrame, 至少含 code 和所有 factor_names 列

        返回
        ----
        DataFrame: [code, score, rank], 按 score 降序排列
        """
        missing = set(self.factor_names) - set(factor_df.columns)
        if missing:
            raise KeyError(f"缺少因子列: {missing}")

        X = factor_df[self.factor_names].copy()

        # 填充 NaN（用列均值）
        X = X.fillna(X.mean())

        try:
            prob = self.model.predict_proba(X)[:, 1]
        except Exception:
            # 回归模式回退
            prob = self.model.predict(X)

        result = factor_df[["code"]].copy()
        result["score"] = prob
        result["rank"] = result["score"].rank(ascending=False, method="first").astype(int)
        result = result.sort_values("rank")

        logger.info(f"预测完成: {len(result)} 只股票, top-5: {result['code'].head().tolist()}")
        return result.reset_index(drop=True)
