#!/usr/bin/env python
"""一键回测管线：信号生成 → 回测 → CSV 输出 → Web 展示。

用法:
    python scripts/run_backtest_pipeline.py                          # 默认 2025-01-01 ~ 今天
    python scripts/run_backtest_pipeline.py --start 2020-01-01       # 长区间
    python scripts/run_backtest_pipeline.py --start 2025-01-01 --top-n 10
"""
import sys, os, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from loguru import logger

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
SIGNALS_DIR = os.path.join(os.path.dirname(SCRIPTS), "data", "signals")
TRADES_DIR = os.path.join(os.path.dirname(SCRIPTS), "data", "backtest_trades")
SIGNALS_FILE = os.path.join(SIGNALS_DIR, "bt_signals_latest.csv")


def parse_args():
    p = argparse.ArgumentParser(description="一键回测管线")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--label", type=str, default=None, help="策略标签(用于文件名)")
    return p.parse_args()


def main():
    args = parse_args()
    end_date = args.end or date.today().strftime("%Y-%m-%d")
    os.makedirs(SIGNALS_DIR, exist_ok=True)
    os.makedirs(TRADES_DIR, exist_ok=True)

    # ── Step 1: 生成信号 ──
    logger.info(f"═══ Step 1/3: 生成信号 {args.start} → {end_date} ═══")
    gen_cmd = [
        sys.executable, os.path.join(SCRIPTS, "gen_signals.py"),
        "--start", args.start, "--end", end_date,
        "--top-n", str(max(args.top_n * 4, 20)),  # 足够顺延候选
        "--mcap-proxy",
        "--out", SIGNALS_FILE,
    ]
    subprocess.run(gen_cmd, check=True)

    # ── Step 2: 回测 ──
    logger.info(f"═══ Step 2/3: 回测 Top-{args.top_n} ═══")
    bt_cmd = [
        sys.executable, os.path.join(SCRIPTS, "bt_backtest.py"),
        "--start", args.start, "--end", end_date,
        "--top-n", str(args.top_n),
        "--cash", str(int(args.cash)),
        "--signals", SIGNALS_FILE,
        "--exec-close",
    ]
    subprocess.run(bt_cmd, check=True)

    # ── Step 3: 输出（文件名含日期区间+策略名）──
    date_tag = f"{args.start.replace('-','')}_{end_date.replace('-','')}"
    label = args.label or "E4"
    safe_label = label.replace("/", "_").replace(" ", "_")
    trades_csv = os.path.join(TRADES_DIR, f"trades_top{args.top_n}_{date_tag}_{safe_label}.csv")
    equity_json = os.path.join(TRADES_DIR, f"equity_top{args.top_n}_{date_tag}_{safe_label}.json")
    # 重命名 bt_backtest 输出的固定文件名
    src_trades = os.path.join(TRADES_DIR, f"trades_top{args.top_n}.csv")
    src_equity = os.path.join(TRADES_DIR, f"equity_top{args.top_n}.json")
    if os.path.exists(src_trades):
        os.rename(src_trades, trades_csv)
    if os.path.exists(src_equity):
        os.rename(src_equity, equity_json)
    logger.info(f"═══ 完成 ═══")
    logger.info(f"  交割单: {trades_csv}")
    logger.info(f"  权益曲线: {equity_json}")
    logger.info(f"  Web: http://localhost:8899/backtest")


if __name__ == "__main__":
    main()
