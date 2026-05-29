#!/usr/bin/env python
"""v1.12 Regime 自适应网格搜索 — 分牛/熊/震荡独立评估。

使 RegimeAwareEnsemble 自动按市场状态派发子模型，输出分状态指标，
按牛市榜（收益优先）和熊市榜（回撤优先）分别排名。

用法:
    python scripts/grid_search.py                   # 全量64组
    python scripts/grid_search.py --subset          # 精简32组
    python scripts/grid_search.py --bull-only        # 仅牛市榜（不跑新搜索，分析已有结果）
"""
from __future__ import annotations

import sys
import os
import json
import re
import time
import argparse
import itertools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 固定参数 ──
FIXED = {
    "factor_preset": "+momentum+reversal+volatility+liquidity+fundamental",
    "train_years": 3,
    "val_years": 1,
    "optuna": False,
    "dynamic": True,
    "no_multi_horizon": True,   # Regime 模式必须关闭 MH
    "regime": True,
}

# ── Regime 搜索网格 ──
FULL_GRID = {
    "forward_days": [1, 5],
    "industry_neutralize": [True, False],
    "top_n": [5, 10, 15, 20],
    "rebalance_freq": [1, 5],
    "universe_size": [300, 500],
}

SUBSET_GRID = {
    "forward_days": [1, 5],
    "industry_neutralize": [True, False],
    "top_n": [10, 15, 20],
    "rebalance_freq": [1, 5],
    "universe_size": [300, 500],
}


def expand_combinations(grid: dict) -> list[dict]:
    return [dict(zip(grid.keys(), combo))
            for combo in itertools.product(*grid.values())]


def build_cmd(combo: dict, start: str, end: str) -> list[str]:
    cmd = [
        sys.executable, "scripts/run_ml_backtest.py",
        "--factor-preset", FIXED["factor_preset"],
        "--train-years", str(FIXED["train_years"]),
        "--val-years", str(FIXED["val_years"]),
        "--start", start, "--end", end,
        "--top-n", str(combo["top_n"]),
        "--rebalance-freq", str(combo["rebalance_freq"]),
        "--universe-size", str(combo["universe_size"]),
        "--forward-days", str(combo["forward_days"]),
        "--no-multi-horizon", "--regime", "--dynamic",
    ]
    if combo["industry_neutralize"]:
        cmd.append("--industry-neutralize")
    else:
        cmd.append("--no-industry-neutralize")
    return cmd


def combo_label(combo: dict) -> str:
    ind = "Ind" if combo["industry_neutralize"] else "noInd"
    return (f"T+{combo['forward_days']}_{ind}_"
            f"top{combo['top_n']}_rb{combo['rebalance_freq']}_"
            f"u{combo['universe_size']}")


def parse_metrics_from_stdout(stdout: str) -> dict | None:
    """提取主汇总 + 分状态指标。"""
    metrics = {}

    # 主汇总: 总收益: X%, 年化: Y%, Sharpe: Z, 最大回撤: W%
    m = re.search(
        r'总收益[：:]\s*([\d.-]+)%[，,]\s*年化[：:]\s*([\d.-]+)%[，,]\s*'
        r'Sharpe[：:]\s*([\d.-]+)[，,]\s*最大回撤[：:]\s*([\d.-]+)%',
        stdout)
    if m:
        metrics["total_return"] = float(m.group(1))
        metrics["cagr"] = float(m.group(2))
        metrics["sharpe"] = float(m.group(3))
        metrics["max_dd"] = abs(float(m.group(4))) / 100

    # 分状态: [bull] 天数=N, 总收益=X%, 年化=Y%, Sharpe=Z, MaxDD=W%
    for reg in ["bull", "bear", "sideways"]:
        m = re.search(
            rf'\[{reg}\]\s*天数=(\d+),\s*总收益=([\d.-]+)%,\s*'
            rf'年化=([\d.-]+)%,\s*Sharpe=([\d.-]+),\s*MaxDD=([\d.-]+)%',
            stdout)
        if m:
            metrics[f"{reg}_days"] = int(m.group(1))
            metrics[f"{reg}_total_ret"] = float(m.group(2))
            metrics[f"{reg}_cagr"] = float(m.group(3))
            metrics[f"{reg}_sharpe"] = float(m.group(4))
            metrics[f"{reg}_dd"] = abs(float(m.group(5))) / 100

    return metrics if "cagr" in metrics else None


def run_grid(start: str, end: str, grid: dict, output: str = "output/regime_grid_results.csv"):
    combos = expand_combinations(grid)
    total = len(combos)
    logger.info(f"Regime 网格搜索: {total} 个组合, 区间 {start}-{end}")

    results = []
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(combos):
        label = combo_label(combo)
        cmd = build_cmd(combo, start, end)
        logger.info(f"[{i+1}/{total}] {label}")

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0

        merged = proc.stdout + "\n" + proc.stderr
        metrics = parse_metrics_from_stdout(merged)

        row = {**combo, "label": label, "elapsed_s": round(elapsed, 1),
               "exit_code": proc.returncode}
        if metrics:
            row.update(metrics)

        results.append(row)

        # 增量保存
        pd.DataFrame(results).to_csv(output_path, index=False)

        # 实时进度
        if metrics:
            logger.info(f"  CAGR={metrics['cagr']:.1f}%, DD={metrics['max_dd']:.1%}, "
                        f"Bull: {metrics.get('bull_cagr', '?')}%, "
                        f"Bear: {metrics.get('bear_cagr', '?')}%")

    # ── 最终排名 ──
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    valid = df.dropna(subset=["cagr"])

    # 牛市榜（按 bull_cagr 降序）
    bull_rank = valid.dropna(subset=["bull_cagr"]).sort_values("bull_cagr", ascending=False)
    # 熊市榜（按 bear_dd 升序）
    bear_rank = valid.dropna(subset=["bear_dd"]).sort_values("bear_dd")

    logger.info(f"\n{'='*80}")
    logger.info(f"Regime 网格搜索完成: {len(valid)}/{total} 个有效结果")

    logger.info(f"\n═══ 牛市榜 (按 bull_cagr 降序) ═══")
    logger.info(f"{'Rank':<5} {'Label':<40} {'BullCAGR':<10} {'BullDD':<8} {'BearCAGR':<10} {'BearDD':<8}")
    for rank, (_, row) in enumerate(bull_rank.head(10).iterrows()):
        logger.info(
            f"{rank+1:<5} {row['label']:<40} "
            f"{row.get('bull_cagr',0):<10.1f}% {row.get('bull_dd',0):<8.1%} "
            f"{row.get('bear_cagr',0):<10.1f}% {row.get('bear_dd',0):<8.1%}"
        )

    logger.info(f"\n═══ 熊市榜 (按 bear_dd 升序) ═══")
    logger.info(f"{'Rank':<5} {'Label':<40} {'BearDD':<8} {'BearCAGR':<10} {'BullCAGR':<10} {'BullDD':<8}")
    for rank, (_, row) in enumerate(bear_rank.head(10).iterrows()):
        logger.info(
            f"{rank+1:<5} {row['label']:<40} "
            f"{row.get('bear_dd',0):<8.1%} {row.get('bear_cagr',0):<10.1f}% "
            f"{row.get('bull_cagr',0):<10.1f}% {row.get('bull_dd',0):<8.1%}"
        )

    logger.info(f"\n结果已保存: {output_path.resolve()}")
    return valid


def main():
    parser = argparse.ArgumentParser(description="Regime 自适应网格搜索")
    parser.add_argument("--subset", action="store_true", help="精简网格（~48组）")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260528")
    parser.add_argument("--output", default="output/regime_grid_results.csv")
    parser.add_argument("--bull-only", action="store_true", help="仅分析已有结果不跑新搜索")
    args = parser.parse_args()

    if args.bull_only:
        output_path = Path(args.output)
        if not output_path.exists():
            logger.error(f"结果文件不存在: {output_path}")
            sys.exit(1)
        df = pd.read_csv(output_path)
        valid = df.dropna(subset=["bull_cagr"])
        logger.info(f"牛市 Top 5:")
        for _, row in valid.sort_values("bull_cagr", ascending=False).head(5).iterrows():
            logger.info(f"  {row['label']}: bull_cagr={row['bull_cagr']:.1f}%, bull_dd={row['bull_dd']:.1%}, "
                        f"bear_cagr={row['bear_cagr']:.1f}%, bear_dd={row['bear_dd']:.1%}")
        logger.info(f"\n熊市 Top 5 (低回撤):")
        for _, row in valid.sort_values("bear_dd").head(5).iterrows():
            logger.info(f"  {row['label']}: bear_dd={row['bear_dd']:.1%}, bear_cagr={row['bear_cagr']:.1f}%, "
                        f"bull_cagr={row['bull_cagr']:.1f}%")
        return

    grid = SUBSET_GRID if args.subset else FULL_GRID
    grid["start"] = args.start
    grid["end"] = args.end
    run_grid(args.start, args.end, grid, args.output)


if __name__ == "__main__":
    main()
