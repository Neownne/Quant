#!/usr/bin/env python
"""用 featurize_signals.py 输出的信号级特征重新训练 XGBRegressor。

比原训练快 5-10x（跳过因子计算，直接用预计算特征）。
"""

from __future__ import annotations

import argparse, os, sys, time, pickle
from datetime import date, timedelta
import numpy as np
import pandas as pd
from loguru import logger
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from factors.monitor import compute_ic_series, compute_ic_summary

TRAIN_END = "2024-06-30"
VAL_END   = "2025-06-30"
TEST_END  = "2026-06-14"

MODEL_OUT = "data/models/signal_quality_xgb_v2.pkl"


def parse_args():
    p = argparse.ArgumentParser(description="信号级特征重训练")
    p.add_argument("--features-csv", default="data/signals/bt_signals_features.csv")
    p.add_argument("--signals-csv", default="data/signals/bt_signals_full.csv")
    p.add_argument("--forward-days", type=int, default=10)
    p.add_argument("--ic-threshold", type=float, default=0.015)
    p.add_argument("--t-threshold", type=float, default=1.2)
    p.add_argument("--corr-threshold", type=float, default=0.6)
    p.add_argument("--max-factors", type=int, default=30)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--model-out", default=MODEL_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # ── 1. 加载特征 ──
    logger.info(f"加载特征: {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)

    # 因子列
    meta = {"date", "code", "name", "rank", "score", "close", "is_limit_up", "is_limit_down"}
    factor_cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    logger.info(f"{len(df)} 行, {len(factor_cols)} 因子")

    # ── 2. 加载 label ──
    # 我们需要 forward return。从日线数据计算。
    logger.info("计算标签（forward return）...")
    signals = pd.read_csv(args.signals_csv)
    signals = signals.rename(columns={"date": "trade_date"}) if "date" in signals.columns else signals
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])
    signals["code"] = signals["code"].astype(str).str.zfill(6)

    signal_codes = sorted(signals["code"].unique().tolist())
    engine = get_engine()
    pre_start = (signals["trade_date"].min() - timedelta(days=5)).strftime("%Y-%m-%d")
    post_end = (signals["trade_date"].max() + timedelta(days=args.forward_days + 5)).strftime("%Y-%m-%d")

    daily = load_daily_data(engine, signal_codes, pre_start, post_end, cols=["close"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    engine.dispose()

    daily = daily.sort_values(["code", "trade_date"])
    daily["ret_fwd"] = daily.groupby("code")["close"].transform(
        lambda x: x.shift(-args.forward_days) / x - 1
    )

    # 创建 label
    daily["key"] = daily["trade_date"].astype(str) + "_" + daily["code"].astype(str)
    df["key"] = df["date"].astype(str) + "_" + df["code"].astype(str)

    labels = daily[["key", "ret_fwd"]].dropna(subset=["ret_fwd"])
    df = df.merge(labels, on="key", how="inner")
    df["label"] = df["ret_fwd"]

    valid = df.dropna(subset=["label"])
    logger.info(f"有效样本: {len(valid)}/{len(df)}")

    # ── 3. 排除 NaN>80% 的列 ──
    bad_cols = [c for c in factor_cols if valid[c].isna().mean() > 0.8]
    if bad_cols:
        logger.info(f"排除高NaN列 ({len(bad_cols)}): {bad_cols}")
        factor_cols = [c for c in factor_cols if c not in bad_cols]

    # ── 4. 三窗口划分 ──
    valid["trade_date"] = valid["date"]
    train_df = valid[valid["trade_date"] <= TRAIN_END].copy()
    val_df = valid[(valid["trade_date"] > TRAIN_END) & (valid["trade_date"] <= VAL_END)].copy()
    test_df = valid[(valid["trade_date"] > VAL_END) & (valid["trade_date"] <= TEST_END)].copy()

    logger.info(f"窗口: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ── 5. 因子筛选 ──
    logger.info("因子筛选...")
    ic_passed = filter_factors_by_ic(
        train_df.rename(columns={"label": "ret_1d"}),
        factor_cols, ret_col="ret_1d",
        ic_threshold=args.ic_threshold, t_threshold=args.t_threshold,
    )
    logger.info(f"IC门禁: {len(factor_cols)} → {len(ic_passed)}")

    ic_series = compute_ic_series(
        train_df.rename(columns={"label": "ret_1d"}), ic_passed, ret_col="ret_1d"
    )
    ic_summary = compute_ic_summary(ic_series)
    selected = select_orthogonal_factors(
        train_df, ic_passed, threshold=args.corr_threshold, ic_summary=ic_summary
    )
    if len(selected) > args.max_factors:
        top = ic_summary.loc[ic_summary.index.isin(selected)].sort_values("ic_mean", key=abs, ascending=False)
        selected = list(top.index[:args.max_factors])

    logger.info(f"选中: {len(selected)} 因子")
    for f in selected:
        logger.info(f"  {f}: |IC|={abs(ic_summary.loc[f, 'ic_mean']):.4f}")

    # ── 6. 训练 ──
    X_train = np.nan_to_num(train_df[selected].values, nan=0.0, posinf=1e6, neginf=-1e6)
    y_train = train_df["label"].values
    X_val = np.nan_to_num(val_df[selected].values, nan=0.0, posinf=1e6, neginf=-1e6)
    y_val = val_df["label"].values
    X_test = np.nan_to_num(test_df[selected].values, nan=0.0, posinf=1e6, neginf=-1e6)
    y_test = test_df["label"].values

    model = XGBRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.lr, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0, eval_metric="rmse",
        early_stopping_rounds=30, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ── 7. 评估 ──
    def evaluate(X, y, name):
        pred = model.predict(X)
        r2 = r2_score(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        ic, _ = spearmanr(pred, y)
        # 分组单调性
        qids = pd.qcut(pred, 5, labels=False, duplicates="drop")
        q_actual = [y[qids == q].mean() for q in sorted(set(qids)) if (qids == q).sum() > 5]
        mono = all(q_actual[i] <= q_actual[i+1] for i in range(len(q_actual)-1))
        logger.info(f"  [{name}] R²={r2:.4f} RMSE={rmse:.4f} IC={ic:.3f} mono={mono}")
        return {"r2": r2, "rmse": rmse, "ic": ic, "monotonic": mono, "quintiles": q_actual}

    evals = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        evals[name] = evaluate(X, y, name)

    # ── 8. 保存 ──
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({
            "model": model,
            "selected_factors": selected,
            "evals": evals,
            "model_type": "XGBRegressor",
            "feature_source": "featurize_signals",
            "trained_at": str(date.today()),
        }, f)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  重训练完成 ({elapsed:.0f}s)")
    print(f"  特征: {args.features_csv}")
    print(f"  因子: {len(factor_cols)} → {len(selected)} 个")
    print(f"  Train R²={evals['train']['r2']:.4f}  IC={evals['train']['ic']:.3f}")
    print(f"  Val   R²={evals['val']['r2']:.4f}  IC={evals['val']['ic']:.3f}")
    print(f"  Test  R²={evals['test']['r2']:.4f}  IC={evals['test']['ic']:.3f}")
    print(f"  模型: {args.model_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
