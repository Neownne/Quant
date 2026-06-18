#!/usr/bin/env python
"""一键回测管线：信号生成 → 回测 → CSV 输出。

三策略统一入口:
    python scripts/run_backtest_pipeline.py --strategy limit_up --start 2020-01-01 --top-n 5 --label v1
    python scripts/run_backtest_pipeline.py --strategy yaogu --start 2020-01-01 --top-n 5 --min-score 3
    python scripts/run_backtest_pipeline.py --strategy bull --start 2020-01-01 --top-n 5
"""

import sys, os, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from loguru import logger

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
SIGNALS_DIR = os.path.join(os.path.dirname(SCRIPTS), "data", "signals")
TRADES_DIR = os.path.join(os.path.dirname(SCRIPTS), "data", "backtest_trades")

STRATEGIES = {
    "limit_up": {
        "gen": "gen_limit_up_signals.py",
        "bt": "bt_backtest.py",
        "signals_file": "bt_signals_limit_up.csv",
        "top_n_multiplier": 4,  # 多生成候选供顺延
        "bt_extra_args": ["--exec-close"],
    },
    "yaogu": {
        "gen": "gen_yaogu_signals.py",
        "bt": "bt_yaogu.py",
        "signals_file": "bt_signals_yaogu.csv",
        "top_n_multiplier": 1,  # yaogu评分已是最终排序
        "bt_extra_args": [],  # bt_yaogu有自己的入场逻辑
    },
    "bull": {
        "gen": "gen_bull_signals.py",
        "bt": "bt_yaogu.py",  # 牛股用妖股回测引擎(MA20/止损/移动止盈)
        "signals_file": "bt_signals_bull.csv",
        "top_n_multiplier": 1,
        "bt_extra_args": [],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="一键回测管线（三策略统一）")
    p.add_argument("--strategy", type=str, default="limit_up",
                   choices=["limit_up", "yaogu", "bull"],
                   help="策略: limit_up(涨停) / yaogu(妖股) / bull(牛股)")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--label", type=str, default=None, help="策略标签(用于文件名)")
    p.add_argument("--min-score", type=int, default=3, help="妖股最低评分")
    p.add_argument("--trailing-stop", type=float, default=0.12, help="移动止盈比例")
    return p.parse_args()


def main():
    args = parse_args()
    end_date = args.end or date.today().strftime("%Y-%m-%d")
    strat = STRATEGIES[args.strategy]
    os.makedirs(SIGNALS_DIR, exist_ok=True)
    os.makedirs(TRADES_DIR, exist_ok=True)

    signals_file = os.path.join(SIGNALS_DIR, strat["signals_file"])

    # ── Step 1: 生成信号 ──
    logger.info(f"═══ Step 1/3: 生成{args.strategy}信号 {args.start} → {end_date} ═══")
    gen_top_n = max(args.top_n * strat["top_n_multiplier"], args.top_n)
    gen_cmd = [
        sys.executable, os.path.join(SCRIPTS, strat["gen"]),
        "--start", args.start, "--end", end_date,
        "--top-n", str(gen_top_n),
        "--out", signals_file,
    ]
    if args.strategy == "yaogu":
        gen_cmd += ["--min-score", str(args.min_score)]
    subprocess.run(gen_cmd, check=True)

    # ── Step 2: 回测 ──
    logger.info(f"═══ Step 2/3: 回测 Top-{args.top_n} ═══")
    bt_cmd = [
        sys.executable, os.path.join(SCRIPTS, strat["bt"]),
        "--start", args.start, "--end", end_date,
        "--top-n", str(args.top_n),
        "--cash", str(int(args.cash)),
        "--signals", signals_file,
    ]
    if args.strategy == "limit_up":
        bt_cmd += strat["bt_extra_args"]
    else:
        # yaogu/bull use bt_yaogu.py which accepts these
        bt_cmd += ["--label", args.label or args.strategy,
                   "--min-score", str(args.min_score),
                   "--trailing-stop", str(args.trailing_stop)]
    subprocess.run(bt_cmd, check=True)

    # ── Step 3: 输出 ──
    date_tag = f"{args.start.replace('-','')}_{end_date.replace('-','')}"
    label = args.label or args.strategy
    safe_label = label.replace("/", "_").replace(" ", "_")

    if args.strategy == "limit_up":
        # bt_backtest.py 输出固定文件名，需要重命名
        src_trades = os.path.join(TRADES_DIR, f"trades_top{args.top_n}.csv")
        src_equity = os.path.join(TRADES_DIR, f"equity_top{args.top_n}.json")
        trades_csv = os.path.join(TRADES_DIR, f"trades_limit_up_{args.top_n}_{date_tag}_{safe_label}.csv")
        equity_json = os.path.join(TRADES_DIR, f"equity_limit_up_{args.top_n}_{date_tag}_{safe_label}.json")
        if os.path.exists(src_trades):
            os.rename(src_trades, trades_csv)
        if os.path.exists(src_equity):
            os.rename(src_equity, equity_json)
        logger.info(f"  交割单: {trades_csv}")
        logger.info(f"  权益曲线: {equity_json}")
    else:
        # bt_yaogu.py 直接输出带标签的文件名
        trades_csv = os.path.join(TRADES_DIR,
            f"trades_{args.strategy}_{args.top_n}_{date_tag}_{safe_label}.csv")
        logger.info(f"  交割单: {trades_csv}")

    logger.info(f"═══ 完成 ═══")


if __name__ == "__main__":
    main()
