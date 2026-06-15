#!/usr/bin/env python
"""ML 信号质量策略 — 一键管线。

训练模型 → 生成ML过滤信号 → 向量化回测

用法:
    python scripts/run_ml_pipeline.py --start 2020-01-01 --top-n 5 --label ML_v1
    python scripts/run_ml_pipeline.py --start 2025-01-01 --skip-train  # 复用已有模型
"""

import sys, os, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from loguru import logger

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "data/models/signal_quality_xgb.pkl"
SIGNALS_ML = "data/signals/bt_signals_ml.csv"
TRADES_DIR = "data/backtest_trades"


def parse_args():
    p = argparse.ArgumentParser(description="ML信号质量一键管线")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--label", type=str, default="ML", help="策略标签")
    p.add_argument("--skip-train", action="store_true", help="跳过训练")
    p.add_argument("--retrain", action="store_true", help="强制重新训练")
    p.add_argument("--trailing-stop", type=float, default=0.12)
    p.add_argument("--no-trailing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    end_date = args.end or date.today().strftime("%Y-%m-%d")
    os.makedirs(TRADES_DIR, exist_ok=True)

    # ── Step 1: 训练模型 ──
    if not args.skip_train:
        train_args = [
            sys.executable, os.path.join(SCRIPTS, "train_signal_quality.py"),
            "--start", args.start,
            "--signals", "data/signals/bt_signals_latest.csv",
        ]
        if args.retrain:
            train_args.append("--retrain")

        model_exists = os.path.exists(MODEL_PATH)
        if model_exists and not args.retrain:
            logger.info(f"Step 1/3: 跳过训练（模型已存在: {MODEL_PATH}）")
        else:
            logger.info(f"Step 1/3: 训练模型 ...")
            r = subprocess.run(train_args)
            if r.returncode != 0:
                logger.error("训练失败")
                return
    else:
        logger.info("Step 1/3: 跳过训练（--skip-train）")

    if not os.path.exists(MODEL_PATH):
        logger.error(f"模型不存在: {MODEL_PATH}，请先训练")
        return

    # ── Step 2: 生成 ML 信号 ──
    logger.info(f"Step 2/3: 生成 ML 信号 {args.start} → {end_date} ...")
    gen_args = [
        sys.executable, os.path.join(SCRIPTS, "gen_signals_ml.py"),
        "--start", args.start, "--end", end_date,
        "--top-n", str(args.top_n),
        "--candidate-multiplier", "4",
        "--mcap-proxy",
        "--model", MODEL_PATH,
        "--out", SIGNALS_ML,
    ]
    r = subprocess.run(gen_args)
    if r.returncode != 0:
        logger.error("信号生成失败")
        return

    # ── Step 3: 回测 ──
    logger.info(f"Step 3/3: 回测 Top-{args.top_n} ...")
    bt_args = [
        sys.executable, os.path.join(SCRIPTS, "bt_ml_signals.py"),
        "--start", args.start, "--end", end_date,
        "--top-n", str(args.top_n),
        "--cash", str(int(args.cash)),
        "--signals", SIGNALS_ML,
        "--trailing-stop", str(args.trailing_stop),
    ]
    if args.no_trailing:
        bt_args.append("--no-trailing")

    r = subprocess.run(bt_args)
    if r.returncode != 0:
        logger.error("回测失败")
        return

    # ── 文件命名（含日期区间+策略标签）──
    date_tag = f"{args.start.replace('-','')}_{end_date.replace('-','')}"
    safe_label = args.label.replace("/", "_").replace(" ", "_")

    src_trades = os.path.join(TRADES_DIR, f"trades_ml_{args.top_n}_{date_tag}.csv")
    if os.path.exists(src_trades):
        dst_trades = os.path.join(TRADES_DIR, f"trades_ml_{args.top_n}_{date_tag}_{safe_label}.csv")
        os.rename(src_trades, dst_trades)
        logger.info(f"交割单: {dst_trades}")

    logger.info(f"完成！Web: http://localhost:8899/backtest")


if __name__ == "__main__":
    main()
