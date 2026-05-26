#!/usr/bin/env python
"""ML 选股端到端回测验证。

用法:
    python scripts/run_ml_backtest.py                              # 集成模式+IC门禁
    python scripts/run_ml_backtest.py --regime --optuna            # 分状态+超参优化
    python scripts/run_ml_backtest.py --model xgboost              # 单模型
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from data.db import get_engine
from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train, walk_forward_train_ensemble, walk_forward_train_by_regime
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from models.regime import detect_regime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--factors", default="all")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--val-years", type=int, default=1)
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260101")
    parser.add_argument("--codes", default="")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--no-orthogonal", action="store_true")
    parser.add_argument("--no-ic-gate", action="store_true", help="跳过 IC 门禁")
    parser.add_argument("--regime", action="store_true", help="启用分市场状态训练")
    parser.add_argument("--optuna", action="store_true", help="启用 Optuna 超参优化")
    parser.add_argument("--optuna-trials", type=int, default=30, help="Optuna 搜索轮数")
    args = parser.parse_args()

    if args.factors == "all":
        factor_names = list(ALL_FACTORS.keys())
    else:
        factor_names = [f.strip() for f in args.factors.split(",")]

    logger.info(f"{len(factor_names)} 个因子, IC门禁={'ON' if not args.no_ic_gate else 'OFF'}, "
                f"regime={'ON' if args.regime else 'OFF'}, optuna={'ON' if args.optuna else 'OFF'}")

    engine = get_engine()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    if codes is None:
        codes = pd.read_sql(
            "SELECT code FROM stock_basic WHERE is_st = FALSE "
            "AND list_date <= CURRENT_DATE - INTERVAL '60 days' "
            "ORDER BY code LIMIT 200",
            engine,
        )["code"].tolist()
        logger.info(f"测试范围: {len(codes)} 只股票")

    code_list = ",".join([f"'{c}'" for c in codes])

    # OHLCV
    sql = f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily
        WHERE code IN ({code_list})
          AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY code, trade_date
    """
    ohlcv = pd.read_sql(sql, engine)
    logger.info(f"OHLCV: {len(ohlcv)} 行")

    # 加载指数数据（regime 模式需要）
    regime_df = None
    if args.regime:
        try:
            idx_sql = f"""
                SELECT trade_date, close FROM index_daily
                WHERE code = '000001' AND trade_date BETWEEN '{args.start}' AND '{args.end}'
                ORDER BY trade_date
            """
            index_df = pd.read_sql(idx_sql, engine)
            if not index_df.empty:
                regime_df = detect_regime(index_df)
                logger.info(f"指数数据: {len(index_df)} 行, regime={regime_df['regime'].value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"指数数据加载失败: {e}, 回退到非 regime 模式")
            args.regime = False

    # extra_data
    extra_data = {}
    try:
        extra_sql = f"""
            SELECT code, trade_date, market_cap, pb
            FROM stock_daily_extra
            WHERE code IN ({code_list})
              AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        """
        extra_df = pd.read_sql(extra_sql, engine)
        if not extra_df.empty:
            extra_df["log_mcap"] = np.log(extra_df["market_cap"].replace(0, np.nan))
            extra_data["log_mcap"] = extra_df[["code", "trade_date", "log_mcap"]]
            extra_data["pb"] = extra_df[["code", "trade_date", "pb"]]
            logger.info(f"  估值数据: {len(extra_df)} 行")
    except Exception as e:
        logger.warning(f"  估值数据加载失败: {e}")

    try:
        sh_sql = f"""
            SELECT code, end_date AS trade_date, shareholder_count
            FROM stock_shareholder
            WHERE code IN ({code_list})
              AND end_date BETWEEN '{args.start}' AND '{args.end}'
        """
        sh_df = pd.read_sql(sh_sql, engine)
        if not sh_df.empty:
            extra_data["shareholder_count"] = sh_df[["code", "trade_date", "shareholder_count"]]
            logger.info(f"  股东数据: {len(sh_df)} 行")
    except Exception as e:
        logger.warning(f"  股东数据加载失败: {e}")

    engine.dispose()

    # 构建因子数据集（含 ret_1d 供 IC 计算）
    dataset = build_factor_dataset(
        ohlcv, factor_names, label_mode="binary",
        extra_data=extra_data if extra_data else None,
    )

    # 1. IC 门禁
    if not args.no_ic_gate:
        factor_names = filter_factors_by_ic(dataset, factor_names, ret_col="ret_1d")

    # 2. 正交筛选
    if not args.no_orthogonal:
        factor_cols = select_orthogonal_factors(dataset, factor_names, threshold=0.7)
    else:
        factor_cols = factor_names

    # 3. Walk-forward 训练
    use_ensemble = not args.no_ensemble

    if args.regime and use_ensemble and regime_df is not None:
        logger.info("使用分市场状态集成训练")
        results = walk_forward_train_by_regime(
            dataset, factor_cols, regime_df,
            train_years=args.train_years, val_years=args.val_years,
        )
    elif use_ensemble:
        logger.info("使用 XGBoost + LightGBM 集成模式")
        results = walk_forward_train_ensemble(
            dataset, factor_cols,
            train_years=args.train_years, val_years=args.val_years,
            use_optuna=args.optuna, optuna_trials=args.optuna_trials,
        )
    else:
        logger.info(f"使用单模型: {args.model}")
        results = walk_forward_train(
            dataset, factor_cols, model_type=args.model,
            train_years=args.train_years, val_years=args.val_years,
        )

    logger.info(f"完成 {len(results)} 个 walk-forward 窗口")

    # 汇总
    all_metrics = []
    for i, r in enumerate(results):
        m = r["metrics"]
        best_t = r.get("best_threshold", 0.5)
        regime_info = f", regimes={r.get('regimes_trained', [])}" if "regimes_trained" in r else ""
        logger.info(
            f"窗口 {i+1}: val={r['val_start'].date()}~{r['val_end'].date()}, "
            f"t={best_t:.2f}, acc={m['accuracy']:.3f}, prec={m['precision']:.3f}, rec={m['recall']:.3f}"
            f"{regime_info}"
        )
        all_metrics.append({
            "window": i + 1,
            "val_start": r["val_start"],
            "val_end": r["val_end"],
            "best_t": best_t,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
        })

    summary = pd.DataFrame(all_metrics)
    print("\n=== Walk-Forward 汇总 ===")
    if summary.empty:
        print("无有效窗口，请检查数据范围（train_years + val_years 需小于数据跨度）")
        return
    print(summary.to_string(index=False))
    print(f"\n平均准确率: {summary['accuracy'].mean():.3f}")
    print(f"平均精确率: {summary['precision'].mean():.3f}")
    print(f"平均召回率: {summary['recall'].mean():.3f}")

    if not use_ensemble and results:
        fi = results[-1]["metrics"].get("feature_importance", pd.Series(dtype=float))
        if not fi.empty:
            print("\n=== Top-10 因子 ===")
            print(fi.head(10).to_string())

    if use_ensemble and results:
        print(f"\n=== 筛选管线 ===")
        active = results[-1].get("active_cols", [])
        print(f"最终活跃因子: {len(active)} 个")
        print(f"因子列表: {active[:15]}..." if len(active) > 15 else f"因子列表: {active}")


if __name__ == "__main__":
    main()
