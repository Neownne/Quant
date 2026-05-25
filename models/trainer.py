"""模型训练：XGBoost / LightGBM 训练与评估。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, feature_names: list[str], model) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "feature_importance": _feature_importance(model, feature_names),
    }


def _feature_importance(model, names: list[str]) -> pd.Series:
    """提取特征重要性。"""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    elif hasattr(model, "get_score"):
        scores = model.get_score(importance_type="gain")
        return pd.Series({k: scores.get(f"f{i}", 0) for i, k in enumerate(names)}).sort_values(ascending=False)
    return pd.Series(dtype=float)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict | None = None,
) -> tuple:
    """训练 XGBoost 二分类器。

    返回: (model, metrics_dict)
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = _compute_metrics(
        y_val.values if hasattr(y_val, "values") else y_val,
        y_pred, y_prob,
        list(X_train.columns), model,
    )
    logger.info(f"XGBoost: acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, rec={metrics['recall']:.3f}")
    return model, metrics


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict | None = None,
) -> tuple:
    """训练 LightGBM 二分类器。"""
    default_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = _compute_metrics(
        y_val.values if hasattr(y_val, "values") else y_val,
        y_pred, y_prob,
        list(X_train.columns), model,
    )
    logger.info(f"LightGBM: acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, rec={metrics['recall']:.3f}")
    return model, metrics


def walk_forward_train(
    df: pd.DataFrame,
    factor_cols: list[str],
    model_type: str = "xgboost",
    train_years: int = 3,
    val_years: int = 1,
) -> list[dict]:
    """Walk-forward 训练循环。

    返回: [{model, metrics, train_end, val_start, val_end}, ...]
    """
    from models.dataset import walk_forward_split

    train_fn = train_xgboost if model_type == "xgboost" else train_lightgbm
    results = []

    for train_df, val_df in walk_forward_split(df, train_years, val_years):
        # 过滤全 NaN 因子列（例如需要 extra_data 才能计算的因子）
        active_cols = [c for c in factor_cols if train_df[c].notna().any()]
        if not active_cols:
            continue
        cols_to_use = active_cols + ["label"]

        train_clean = train_df[cols_to_use].dropna()
        val_clean = val_df[cols_to_use].dropna()

        if len(train_clean) < 100 or len(val_clean) < 50:
            continue

        X_tr = train_clean[active_cols]
        y_tr = train_clean["label"]
        X_v = val_clean[active_cols]
        y_v = val_clean["label"]

        model, metrics = train_fn(X_tr, y_tr, X_v, y_v)
        results.append({
            "model": model,
            "metrics": metrics,
            "train_end": train_df["trade_date"].max(),
            "val_start": val_df["trade_date"].min(),
            "val_end": val_df["trade_date"].max(),
        })

    return results
