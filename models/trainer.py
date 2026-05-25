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


class EnsemblePredictor:
    """XGBoost + LightGBM 概率平均集成预测器。"""

    def __init__(self, xgb_model, lgb_model, factor_names: list[str], threshold: float = 0.5):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.factor_names = factor_names
        self.threshold = threshold

    def predict(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.factor_names) - set(factor_df.columns)
        if missing:
            raise KeyError(f"缺少因子列: {missing}")

        X = factor_df[self.factor_names].copy()
        X = X.fillna(X.mean())

        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X)[:, 1]
        prob = (xgb_prob + lgb_prob) / 2

        result = factor_df[["code"]].copy()
        result["score"] = prob
        result["rank"] = result["score"].rank(ascending=False, method="first").astype(int)
        result = result.sort_values("rank")
        return result.reset_index(drop=True)


def walk_forward_train_ensemble(
    df: pd.DataFrame,
    factor_cols: list[str],
    train_years: int = 3,
    val_years: int = 1,
    threshold: float = 0.5,
) -> list[dict]:
    """Walk-forward 训练（XGBoost + LightGBM 集成）。

    返回: [{ensemble, xgb_model, lgb_model, metrics, ...}, ...]
    """
    from models.dataset import walk_forward_split

    results = []

    for train_df, val_df in walk_forward_split(df, train_years, val_years):
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

        xgb_model, xgb_metrics = train_xgboost(X_tr, y_tr, X_v, y_v)
        lgb_model, lgb_metrics = train_lightgbm(X_tr, y_tr, X_v, y_v)

        xgb_prob = xgb_model.predict_proba(X_v)[:, 1]
        lgb_prob = lgb_model.predict_proba(X_v)[:, 1]
        ensemble_prob = (xgb_prob + lgb_prob) / 2
        ensemble_pred = (ensemble_prob >= threshold).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score
        ensemble_metrics = {
            "accuracy": float(accuracy_score(y_v, ensemble_pred)),
            "precision": float(precision_score(y_v, ensemble_pred, zero_division=0)),
            "recall": float(recall_score(y_v, ensemble_pred, zero_division=0)),
        }
        logger.info(
            f"Ensemble: acc={ensemble_metrics['accuracy']:.3f}, "
            f"prec={ensemble_metrics['precision']:.3f}, rec={ensemble_metrics['recall']:.3f}"
        )

        results.append({
            "ensemble": EnsemblePredictor(xgb_model, lgb_model, active_cols, threshold),
            "xgb_model": xgb_model,
            "lgb_model": lgb_model,
            "metrics": ensemble_metrics,
            "active_cols": active_cols,
            "train_end": train_df["trade_date"].max(),
            "val_start": val_df["trade_date"].min(),
            "val_end": val_df["trade_date"].max(),
        })

    return results
