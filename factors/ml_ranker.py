"""LightGBM LambdaRank 排序器 — 将因子映射为排序得分。

用法:
    from factors.ml_ranker import train_lambdarank, predict_rank, compute_ndcg

    result = train_lambdarank(factor_df, forward_ret)
    scores = predict_rank(result, factor_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import lightgbm as lgb


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """计算 NDCG@k。

    Args:
        y_true: 真实相关性标签（整数值，越高越好）
        y_pred: 预测得分（越高越好）
        k: 截断位置

    Returns:
        NDCG@k 值，范围通常 [0, 1]，理想排序时为 1.0
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0

    k_actual = min(k, len(y_true))
    order = np.argsort(y_pred)[::-1][:k_actual]
    gains = np.take(y_true, order)
    if gains.sum() == 0:
        return 0.0

    denom = np.log2(np.arange(2, k_actual + 2))
    dcg = np.sum(gains / denom)
    ideal_order = np.argsort(y_true)[::-1][:k_actual]
    ideal_gains = np.take(y_true, ideal_order)
    idcg = np.sum(ideal_gains / denom)
    return float(dcg / idcg) if idcg > 0 else 0.0


def train_lambdarank(
    factor_df: pd.DataFrame,
    forward_ret: pd.Series | np.ndarray,
    market_context: pd.DataFrame | None = None,
) -> dict:
    """训练 LightGBM LambdaRank 模型。

    在每个交易日内将 forward_ret 分为 5 个分位数作为相关性标签，
    用 80%/20% 时间序列切分训练/验证集，训练后返回模型及评估指标。

    Args:
        factor_df: 因子 DataFrame，至少包含 [code, trade_date] 列及因子列
        forward_ret: 前向收益率，与 factor_df 相同索引
        market_context: 可选，市场环境特征（当前未使用，为未来扩展保留）

    Returns:
        dict:
            - model: LGBMRanker 实例，或 None（训练失败时）
            - feature_importances: {factor_name: importance} 字典
            - ndcg_score: 验证集日均 NDCG@5
    """
    factor_cols = [c for c in factor_df.columns if c not in ("code", "trade_date")]
    if not factor_cols:
        return {"model": None, "feature_importances": {}, "ndcg_score": 0.0}

    df = factor_df.copy()
    df["_ret"] = forward_ret.values if isinstance(forward_ret, pd.Series) else forward_ret

    # 删除因子或收益率缺失的行
    df = df.dropna(subset=factor_cols + ["_ret"])
    if len(df) == 0:
        return {"model": None, "feature_importances": {}, "ndcg_score": 0.0}

    # 相关性标签：每个交易日内将 forward_ret 分为 5 个分位数 (0-4)
    df = df.sort_values("trade_date")
    df["_q"] = df.groupby("trade_date")["_ret"].transform(
        lambda g: pd.qcut(g, 5, labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["_q"])
    if len(df) == 0:
        return {"model": None, "feature_importances": {}, "ndcg_score": 0.0}

    # 时间序列切分：前 80% 日期训练，后 20% 验证
    dates = sorted(df["trade_date"].unique())
    split_idx = int(len(dates) * 0.8)
    if split_idx == 0 or len(dates) < 2:
        return {"model": None, "feature_importances": {}, "ndcg_score": 0.0}
    train_dates = set(dates[:split_idx])
    train_mask = df["trade_date"].isin(train_dates)

    train_df = df[train_mask]
    val_df = df[~train_mask]

    if len(train_df) == 0 or len(val_df) == 0:
        return {"model": None, "feature_importances": {}, "ndcg_score": 0.0}

    X_train = train_df[factor_cols].values.astype(np.float32)
    y_train = train_df["_q"].values.astype(int)
    groups_train = train_df.groupby("trade_date").size().values

    X_val = val_df[factor_cols].values.astype(np.float32)
    y_val = val_df["_q"].values.astype(int)
    groups_val = val_df.groupby("trade_date").size().values

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        num_leaves=31,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        group=groups_train,
        eval_set=[(X_val, y_val)],
        eval_group=[groups_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # 特征重要性
    importances = dict(zip(factor_cols, model.feature_importances_))

    # 验证集 NDCG@5
    y_pred = model.predict(X_val)
    daily_ndcg = []
    offset = 0
    for g_size in groups_val:
        if g_size >= 5:
            daily_ndcg.append(
                compute_ndcg(
                    y_val[offset : offset + g_size],
                    y_pred[offset : offset + g_size],
                    k=5,
                )
            )
        offset += g_size
    ndcg_score = np.mean(daily_ndcg) if daily_ndcg else 0.0

    return {
        "model": model,
        "feature_importances": importances,
        "ndcg_score": float(ndcg_score),
    }


def predict_rank(
    model_dict: dict,
    factor_df: pd.DataFrame,
    market_context: pd.DataFrame | None = None,
) -> pd.Series:
    """用训练好的模型预测排序得分。

    Args:
        model_dict: train_lambdarank 返回的字典
        factor_df: 因子 DataFrame，列须与训练时一致
        market_context: 可选，市场环境特征（当前未使用）

    Returns:
        pd.Series: 排序得分，索引与 factor_df 对齐
    """
    model = model_dict.get("model")
    if model is None:
        return pd.Series(0.0, index=factor_df.index)

    factor_cols = [c for c in factor_df.columns if c not in ("code", "trade_date")]
    X = factor_df[factor_cols].copy()
    X = X.fillna(X.median())
    scores = model.predict(X.values.astype(np.float32))
    return pd.Series(scores, index=factor_df.index)
