#!/usr/bin/env python
"""ML 选股端到端回测验证。

用法:
    python scripts/run_ml_backtest.py                   # 默认参数
    python scripts/run_ml_backtest.py --model lightgbm  # 换模型
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from data.db import get_engine
from models.dataset import build_factor_dataset, walk_forward_split
from models.trainer import walk_forward_train
from models.predictor import DailyPredictor
from factors import ALL_FACTORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--factors", default="all", help="因子列表，逗号分隔或 'all'")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--val-years", type=int, default=1)
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260101")
    parser.add_argument("--codes", default="", help="测试股票代码，逗号分隔，留空=全量")
    args = parser.parse_args()

    # 选择因子
    if args.factors == "all":
        factor_names = list(ALL_FACTORS.keys())
    else:
        factor_names = [f.strip() for f in args.factors.split(",")]

    logger.info(f"使用 {len(factor_names)} 个因子: {factor_names[:5]}...")

    # 加载数据
    engine = get_engine()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    if codes is None:
        codes = pd.read_sql("SELECT code FROM stock_basic LIMIT 200", engine)["code"].tolist()
        logger.info(f"测试范围: {len(codes)} 只股票")

    # OHLCV
    code_list = ",".join([f"'{c}'" for c in codes])
    sql = f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily
        WHERE code IN ({code_list})
          AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY code, trade_date
    """
    ohlcv = pd.read_sql(sql, engine)
    engine.dispose()
    logger.info(f"OHLCV: {len(ohlcv)} 行")

    # 构建因子数据集
    dataset = build_factor_dataset(ohlcv, factor_names, label_mode="binary")

    # Walk-forward 训练
    factor_cols = factor_names
    results = walk_forward_train(
        dataset, factor_cols, model_type=args.model,
        train_years=args.train_years, val_years=args.val_years,
    )
    logger.info(f"完成 {len(results)} 个 walk-forward 窗口")

    # 汇总
    all_metrics = []
    for i, r in enumerate(results):
        m = r["metrics"]
        logger.info(
            f"窗口 {i+1}: val={r['val_start'].date()}~{r['val_end'].date()}, "
            f"acc={m['accuracy']:.3f}, prec={m['precision']:.3f}, rec={m['recall']:.3f}"
        )
        all_metrics.append({
            "window": i + 1,
            "val_start": r["val_start"],
            "val_end": r["val_end"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
        })

    summary = pd.DataFrame(all_metrics)
    print("\n=== Walk-Forward 汇总 ===")
    print(summary.to_string(index=False))
    print(f"\n平均准确率: {summary['accuracy'].mean():.3f}")
    print(f"平均精确率: {summary['precision'].mean():.3f}")
    print(f"平均召回率: {summary['recall'].mean():.3f}")

    # 特征重要性
    if results:
        fi = results[-1]["metrics"].get("feature_importance", pd.Series(dtype=float))
        if not fi.empty:
            print("\n=== Top-10 因子 ===")
            print(fi.head(10).to_string())


if __name__ == "__main__":
    main()
