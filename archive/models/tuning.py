"""超参数调优：阈值搜索 + Optuna 优化。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "accuracy",
) -> tuple[float, float]:
    """搜索最优分类阈值（默认最大化准确率，避免类不平衡时 F1 偏向全预测为正）。"""
    thresholds = np.arange(0.30, 0.65, 0.01)
    best_t = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_t = t

    logger.info(f"最优阈值: {best_t:.2f}, {metric}={best_score:.4f}")
    return best_t, best_score


def optimize_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Optuna 优化 XGBoost 超参。"""
    try:
        import optuna
        import xgboost as xgb
    except ImportError:
        logger.warning("Optuna 未安装，使用默认参数")
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Optuna 最优参数: {study.best_params}, F1={study.best_value:.4f}")
    return study.best_params
