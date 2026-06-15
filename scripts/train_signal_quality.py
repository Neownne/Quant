#!/usr/bin/env python
"""涨停信号质量 XGBoost 回归模型 — 训练脚本。

预测每条涨停信号未来 N 日实际收益率（连续值），按预测收益率排序取 top-N 入场。

用法:
    python scripts/train_signal_quality.py --start 2020-01-01
    python scripts/train_signal_quality.py --start 2020-01-01 --retrain
"""

from __future__ import annotations

import argparse, os, sys, time, pickle
from datetime import date, timedelta
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data
from scripts.validate_factors import (
    compute_factors_fast, compute_cross_sectional_factors,
)
from factors.monitor import compute_ic_series, compute_ic_summary
from factors.screening import filter_factors_by_ic, select_orthogonal_factors

# ── 三窗口 ──
TRAIN_END = "2024-06-30"
VAL_END = "2025-06-30"
TEST_END = "2026-06-14"

MODEL_DIR = "data/models"
DEFAULT_MODEL_PATH = "data/models/signal_quality_xgb_reg.pkl"


def parse_args():
    p = argparse.ArgumentParser(description="涨停信号质量 XGBoost 回归训练")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--signals", default="data/signals/bt_signals_latest.csv")
    p.add_argument("--forward-days", type=int, default=10)
    p.add_argument("--ic-threshold", type=float, default=0.015)
    p.add_argument("--t-threshold", type=float, default=1.2)
    p.add_argument("--corr-threshold", type=float, default=0.6)
    p.add_argument("--max-factors", type=int, default=25)
    p.add_argument("--xgb-n-estimators", type=int, default=200)
    p.add_argument("--xgb-max-depth", type=int, default=5)
    p.add_argument("--xgb-learning-rate", type=float, default=0.03)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.7)
    p.add_argument("--xgb-reg-alpha", type=float, default=1.0)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--model-out", default=None)
    return p.parse_args()


def load_signals(args):
    csv_path = args.signals
    if not os.path.exists(csv_path):
        logger.error(f"信号文件不存在: {csv_path}")
        sys.exit(1)
    sig = pd.read_csv(csv_path)
    sig = sig.rename(columns={"date": "trade_date"}) if "date" in sig.columns else sig
    sig["trade_date"] = pd.to_datetime(sig["trade_date"])
    sig["code"] = sig["code"].astype(str).str.zfill(6)
    sig = sig[(sig["trade_date"] >= pd.Timestamp(args.start))]
    if args.end:
        sig = sig[(sig["trade_date"] <= pd.Timestamp(args.end))]
    return sig


def make_labels(daily, signals, forward_days=10):
    """为每条信号构造连续值回归标签: 直接用前瞻 N 日收益率。"""
    daily = daily.sort_values(["code", "trade_date"]).copy()
    daily["ret_fwd"] = daily.groupby("code")["close"].transform(
        lambda x: x.shift(-forward_days) / x - 1
    )

    signals = signals.copy()
    signals["signal_key"] = (
        signals["trade_date"].astype(str) + "_" + signals["code"].astype(str)
    )
    daily["daily_key"] = (
        daily["trade_date"].astype(str) + "_" + daily["code"].astype(str)
    )

    labels = signals[["signal_key", "trade_date", "code"]].merge(
        daily[["daily_key", "ret_fwd"]].rename(columns={"daily_key": "signal_key"}),
        on="signal_key", how="inner",
    )
    labels["label"] = labels["ret_fwd"].astype(float)  # 连续值回归标签
    labels["valid"] = labels["ret_fwd"].notna()

    logger.info(
        f"标签: {len(labels)} 条, "
        f"mean={labels['label'].mean():.4f}, std={labels['label'].std():.4f}, "
        f"min={labels['label'].min():.4f}, max={labels['label'].max():.4f}, "
        f"有效={labels['valid'].sum()}"
    )
    return labels


def featurize(daily, signals, engine):
    """计算58因子并提取信号日的因子值。"""
    # 行业映射
    basic = pd.read_sql(
        "SELECT code, industry FROM stock_basic WHERE is_st = FALSE", engine
    )
    industry_map = dict(zip(basic["code"].astype(str).str.zfill(6), basic["industry"]))
    daily["industry"] = daily["code"].map(industry_map).fillna("其他")

    # 计算因子
    logger.info("计算 58 因子 ...")
    daily = compute_factors_fast(daily)
    daily = compute_cross_sectional_factors(daily)

    meta_cols = {"code", "trade_date", "open", "high", "low", "close",
                 "volume", "amount", "turnover", "market_cap", "ret", "industry"}
    factor_cols = [c for c in daily.columns
                   if c not in meta_cols
                   and pd.api.types.is_numeric_dtype(daily[c])
                   and not c.startswith("ret_fwd_")]

    # 提取信号日
    signals = signals.copy()
    signals["signal_key"] = (
        signals["trade_date"].astype(str) + "_" + signals["code"].astype(str)
    )
    daily["daily_key"] = (
        daily["trade_date"].astype(str) + "_" + daily["code"].astype(str)
    )

    sf = signals[["signal_key", "trade_date", "code"]].merge(
        daily[["daily_key"] + factor_cols].rename(columns={"daily_key": "signal_key"}),
        on="signal_key", how="inner",
    )
    logger.info(f"特征矩阵: {len(sf)} 行 × {len(factor_cols)} 因子")
    return sf, factor_cols


def select_features(train_df, factor_cols, label_col="label",
                    ic_threshold=0.015, t_threshold=1.2,
                    corr_threshold=0.6, max_factors=25):
    """IC 门禁 + 正交贪心筛选因子（仅用训练集）。"""
    # Stage 1: IC 过滤
    ic_passed = filter_factors_by_ic(
        train_df.rename(columns={label_col: "ret_1d"}),
        factor_cols, ret_col="ret_1d",
        ic_threshold=ic_threshold, t_threshold=t_threshold,
    )
    logger.info(f"IC 门禁: {len(factor_cols)} → {len(ic_passed)} 因子")

    if not ic_passed:
        logger.error("无因子通过 IC 门禁，降低 --ic-threshold 或 --t-threshold")
        return []

    # Stage 2: 正交贪心
    ic_series = compute_ic_series(
        train_df.rename(columns={label_col: "ret_1d"}),
        ic_passed, ret_col="ret_1d",
    )
    ic_summary = compute_ic_summary(ic_series)

    selected = select_orthogonal_factors(
        train_df, ic_passed,
        threshold=corr_threshold, ic_summary=ic_summary,
    )
    if len(selected) > max_factors:
        top = ic_summary.loc[ic_summary.index.isin(selected)]
        top = top.sort_values("ic_mean", key=abs, ascending=False)
        selected = list(top.index[:max_factors])

    logger.info(f"正交筛选: {len(ic_passed)} → {len(selected)} 因子 (corr<{corr_threshold}, max={max_factors})")
    for f in selected:
        logger.info(f"  {f}: |IC|={abs(ic_summary.loc[f, 'ic_mean']):.4f}")
    return selected


def train_xgb(X_train, y_train, X_val, y_val, args):
    """训练 XGBoost 回归模型，预测未来收益率。"""
    logger.info(
        f"y_train: mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
        f"min={y_train.min():.4f}, max={y_train.max():.4f}"
    )

    model = XGBRegressor(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        reg_alpha=args.xgb_reg_alpha,
        reg_lambda=args.xgb_reg_lambda,
        eval_metric="rmse",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    evals_result = model.evals_result()
    best_val_rmse = float(min(evals_result["validation_0"]["rmse"]))

    info = {
        "best_iteration": model.best_iteration or args.xgb_n_estimators,
        "best_val_rmse": best_val_rmse,
        "n_features": X_train.shape[1],
        "train_samples": len(y_train),
        "train_y_mean": float(y_train.mean()),
        "train_y_std": float(y_train.std()),
    }
    logger.info(f"训练完成: {info['best_iteration']} 棵树, val RMSE={best_val_rmse:.6f}")
    return model, info


def evaluate(model, X, y_true, window_name):
    """回归评估：R², MSE, MAE + 按预测分组回测。"""
    y_pred = model.predict(X)

    # 核心回归指标
    r2 = float(r2_score(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    # 基线：预测均值
    y_mean = float(y_true.mean())
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2_baseline = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    pred_mean = float(y_pred.mean())
    pred_std = float(y_pred.std())

    # 按预测收益率分组回测：分成 5 组，看各组实际均值
    n_bins = 5
    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["bin"] = pd.qcut(df["pred"], n_bins, labels=False, duplicates="drop")
    quantile_stats = {}
    for b in sorted(df["bin"].unique()):
        mask = df["bin"] == b
        quantile_stats[int(b)] = {
            "n": int(mask.sum()),
            "pred_mean": float(df.loc[mask, "pred"].mean()),
            "actual_mean": float(df.loc[mask, "actual"].mean()),
            "actual_std": float(df.loc[mask, "actual"].std()),
        }

    # 特征重要性
    imp = model.get_booster().get_score(importance_type="gain")
    top_feat = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]

    result = {
        "window": window_name,
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "n_samples": len(y_true),
        "y_true_mean": y_mean,
        "y_true_std": float(y_true.std()),
        "y_pred_mean": pred_mean,
        "y_pred_std": pred_std,
        "quantile_groups": quantile_stats,
        "top_features": top_feat,
    }

    logger.info(
        f"  [{window_name}] n={len(y_true)} "
        f"R²={r2:.4f} MSE={mse:.6f} MAE={mae:.6f}"
    )
    logger.info(
        f"    y_true: mean={y_mean:.4f} std={y_true.std():.4f} | "
        f"y_pred: mean={pred_mean:.4f} std={pred_std:.4f}"
    )
    # 打印分组 info
    q_lines = []
    for b in sorted(quantile_stats.keys()):
        qs = quantile_stats[b]
        q_lines.append(f"Q{b}: pred={qs['pred_mean']:.4f} actual={qs['actual_mean']:.4f} (n={qs['n']})")
    logger.info(f"    分组: {' | '.join(q_lines)}")
    return result


def main():
    args = parse_args()
    end_date = args.end or date.today().strftime("%Y-%m-%d")
    t0 = time.time()

    model_path = args.model_out or DEFAULT_MODEL_PATH
    if os.path.exists(model_path) and not args.retrain:
        logger.info(f"模型已存在: {model_path}（用 --retrain 重新训练）")
        return

    # ── 1. 加载数据 ──
    logger.info("=== Step 1/5: 加载信号 ===")
    signals = load_signals(args)
    signal_codes = sorted(signals["code"].unique().tolist())
    logger.info(f"信号: {len(signals)} 条, {len(signal_codes)} 只, {signals['trade_date'].nunique()} 天")

    logger.info("=== Step 2/5: 加载日线 ===")
    pre_start = (pd.Timestamp(args.start) - timedelta(days=90)).strftime("%Y-%m-%d")
    post_end = (pd.Timestamp(end_date) + timedelta(days=args.forward_days + 10)).strftime("%Y-%m-%d")

    engine = get_engine()
    daily = load_daily_data(
        engine, signal_codes, pre_start, post_end,
        cols=["open", "high", "low", "close", "volume", "amount", "turnover"],
    )
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    logger.info(f"日线: {len(daily)} 行")

    # ── 2. 标签（连续值回归）──
    labels = make_labels(daily, signals, args.forward_days)

    # ── 3. 特征 ──
    logger.info("=== Step 3/5: 计算特征 ===")
    sf, factor_cols = featurize(daily, signals, engine)
    engine.dispose()

    df = sf.merge(labels[["signal_key", "label", "valid"]], on="signal_key", how="inner")
    df = df[df["valid"]].drop(columns=["valid"])
    logger.info(f"数据集: {len(df)} 行, label mean={df['label'].mean():.4f}, std={df['label'].std():.4f}")

    # ── 4. 三窗口划分 ──
    train_mask = df["trade_date"] <= TRAIN_END
    val_mask = (df["trade_date"] > TRAIN_END) & (df["trade_date"] <= VAL_END)
    test_mask = (df["trade_date"] > VAL_END) & (df["trade_date"] <= TEST_END)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(
        f"窗口: train={len(train_df)} (mean={train_df['label'].mean():.4f}), "
        f"val={len(val_df)} (mean={val_df['label'].mean():.4f}), "
        f"test={len(test_df)} (mean={test_df['label'].mean():.4f})"
    )

    if len(train_df) < 100 or len(val_df) < 50:
        logger.error("训练/验证集太小，请扩大 --start 范围")
        return

    # ── 5. 因子筛选 + 训练 ──
    logger.info("=== Step 4/5: 因子筛选 + 训练 ===")
    selected = select_features(
        train_df, factor_cols, "label",
        ic_threshold=args.ic_threshold, t_threshold=args.t_threshold,
        corr_threshold=args.corr_threshold, max_factors=args.max_factors,
    )
    if not selected:
        return

    X_train = np.nan_to_num(train_df[selected].values, nan=0.0, posinf=1e6, neginf=-1e6)
    y_train = train_df["label"].values
    X_val = np.nan_to_num(val_df[selected].values, nan=0.0, posinf=1e6, neginf=-1e6)
    y_val = val_df["label"].values
    X_test = np.nan_to_num(
        test_df[selected].values if len(test_df) > 0 else X_val,
        nan=0.0, posinf=1e6, neginf=-1e6)
    y_test = test_df["label"].values if len(test_df) > 0 else y_val

    model, train_info = train_xgb(X_train, y_train, X_val, y_val, args)

    # ── 6. 回归评估 ──
    logger.info("=== Step 5/5: 回归评估 ===")
    evals = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        evals[name] = evaluate(model, X, y, name)

    # ── 7. 保存模型 ──
    os.makedirs(os.path.dirname(model_path) or MODEL_DIR, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "selected_factors": selected,
            "evals": evals,
            "train_info": train_info,
            "hyperparams": {
                "forward_days": args.forward_days,
                "ic_threshold": args.ic_threshold,
                "corr_threshold": args.corr_threshold,
                "max_factors": args.max_factors,
                "model_type": "XGBRegressor",
            },
            "trained_at": str(date.today()),
        }, f)
    logger.info(f"模型已保存: {model_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  训练完成 ({elapsed:.0f}s)")
    print(f"  因子: {len(factor_cols)} → {len(selected)} 个")
    print(f"  Train R²: {evals['train']['r2']:.4f}  MSE: {evals['train']['mse']:.6f}  MAE: {evals['train']['mae']:.6f}")
    print(f"  Val   R²: {evals['val']['r2']:.4f}  MSE: {evals['val']['mse']:.6f}  MAE: {evals['val']['mae']:.6f}")
    print(f"  Test  R²: {evals['test']['r2']:.4f}  MSE: {evals['test']['mse']:.6f}  MAE: {evals['test']['mae']:.6f}")
    print(f"  y_true mean: {evals['train']['y_true_mean']:.4f} | y_pred mean: {evals['train']['y_pred_mean']:.4f} "
          f"std: {evals['train']['y_pred_std']:.4f}")
    print(f"  Top-5 特征: {', '.join(f[0] for f in evals['test']['top_features'][:5])}")
    print(f"  模型: {model_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
