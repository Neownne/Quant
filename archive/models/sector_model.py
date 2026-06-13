"""板块打分模型：XGBoost 板块评分 + Walk-Forward 训练。

SectorScoringModel 实现与 EnsemblePredictor 兼容的接口：
  predict(sector_df) -> DataFrame[sector, score, rank]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from models.dataset import walk_forward_split
from models.trainer import train_xgboost


class SectorScoringModel:
    """XGBoost 板块评分预测器。

    实现标准 predict 接口：
      predict(sector_features_df) -> DataFrame[sector, score, rank]
    """

    def __init__(
        self,
        model,
        feature_names: list[str],
        sector_col: str = "sector",
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.sector_col = sector_col

    def predict(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """对板块特征 DataFrame 打分。

        参数
        ----
        sector_df : 至少含 self.sector_col 和 self.feature_names 列

        返回
        ----
        DataFrame: [sector, score, rank]，按 score 降序排列
        """
        # 准备特征矩阵
        available_features = [f for f in self.feature_names if f in sector_df.columns]
        X = sector_df[available_features].copy()
        X = X.fillna(X.mean())  # 用列均值填 NaN

        # 对于模型需要但数据中缺失的特征，填0
        for f in self.feature_names:
            if f not in X.columns:
                X[f] = 0.0
        X = X[self.feature_names]

        # 预测概率
        prob = self.model.predict_proba(X)[:, 1]

        result = pd.DataFrame({
            self.sector_col: sector_df[self.sector_col].values,
            "score": prob,
        })
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result


def walk_forward_train_sectors(
    sector_dataset: pd.DataFrame,
    feature_cols: list[str],
    train_years: int = 3,
    val_years: int = 1,
) -> list[dict]:
    """Walk-forward 训练板块打分模型。

    每个窗口：
    1. 用 XGBoost 在训练集上训练二分类模型
    2. 在验证集上计算准确率/precision/recall
    3. 将模型包装为 SectorScoringModel

    参数
    ----
    sector_dataset : build_sector_dataset 输出，含 trade_date, sector, 特征列, label
    feature_cols : 用于训练的特征列名
    train_years : 训练窗口年数
    val_years : 验证窗口年数

    返回
    ----
    list[dict]: 每窗口 {model: SectorScoringModel, metrics: dict, active_cols: list, ...}
    """
    if sector_dataset.empty:
        return []

    df = sector_dataset.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 确保所需列存在
    available_features = [f for f in feature_cols if f in df.columns]
    if not available_features:
        logger.warning("板块数据中无可用特征列")
        return []

    results = []
    for train_df, val_df in walk_forward_split(df, train_years=train_years, val_years=val_years):
        # 准备数据
        active_cols = [c for c in available_features if c in train_df.columns and c in val_df.columns]

        X_train = train_df[active_cols].copy()
        y_train = train_df["label"].copy()
        X_val = val_df[active_cols].copy()
        y_val = val_df["label"].copy()

        # 清理 NaN
        valid_train = X_train.notna().all(axis=1) & y_train.notna()
        valid_val = X_val.notna().all(axis=1) & y_val.notna()
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        X_val = X_val[valid_val]
        y_val = y_val[valid_val]

        if len(X_train) < 50 or len(X_val) < 10:
            logger.warning(f"窗口数据不足: train={len(X_train)}, val={len(X_val)}, 跳过")
            continue

        # 训练 XGBoost
        xgb_model, metrics = train_xgboost(
            X_train, y_train.values,
            X_val, y_val.values,
        )

        # 包装为 SectorScoringModel
        sector_model = SectorScoringModel(
            model=xgb_model,
            feature_names=active_cols,
        )

        train_end = train_df["trade_date"].max()
        val_end = val_df["trade_date"].max()

        results.append({
            "model": sector_model,
            "metrics": metrics,
            "active_cols": active_cols,
            "train_end": train_end,
            "val_end": val_end,
        })

    logger.info(f"板块模型训练完成: {len(results)} 个窗口")
    return results
