#!/usr/bin/env python
"""v1.12 参数网格搜索：排列组合多周期/regime/行业中性化/持仓数/调仓频率，
使用动态多因子管线，按年化收益+最大回撤综合排名。

用法:
    python scripts/grid_search.py                # 全量搜索
    python scripts/grid_search.py --subset       # 核心组合(~72组，推荐快速扫)
    python scripts/grid_search.py --subset --fast  # 最小验证（~24组）
"""
from __future__ import annotations

import sys
import os
import json
import time
import argparse
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from sqlalchemy import text

# ── 固定参数（网格搜索不扫这些） ──
FIXED = {
    "factor_preset": "+momentum+reversal+volatility+liquidity+fundamental",
    "train_years": 3,
    "val_years": 1,
    "universe_size": 300,
    "model": "xgboost",           # 集成模式固定
    "optuna": False,             # 网格搜索阶段不调参，找到最优组合后再 Optuna
    "optuna_trials": 30,
    "dynamic": True,             # 始终启用动态多因子
}

# ── 完整网格 ──
FULL_GRID = {
    "multi_horizon": [True, False],
    "regime": [True, False],
    "industry_neutralize": [True, False],
    "top_n": [5, 10, 15, 20],
    "rebalance_freq": [1, 2, 5],
    # forward_days: 仅 multi_horizon=False 时生效
    "forward_days": [1, 5],
}

# ── 精简网格（推荐） ──
SUBSET_GRID = {
    "multi_horizon": [True, False],
    "regime": [True, False],
    "industry_neutralize": [True, False],
    "top_n": [10, 15, 20],
    "rebalance_freq": [1, 5],
    "forward_days": [1, 5],
}

# ── 快速验证 ──
FAST_GRID = {
    "multi_horizon": [True, False],
    "regime": [False],
    "industry_neutralize": [True],
    "top_n": [10, 20],
    "rebalance_freq": [1, 5],
    "forward_days": [1],
}


def expand_combinations(grid: dict) -> list[dict]:
    """展开网格，处理 multi_horizon/forward_days 互斥。"""
    combinations = []
    for combo in itertools.product(*grid.values()):
        d = dict(zip(grid.keys(), combo))
        # multi_horizon=True 时 forward_days 无效，去重
        if d["multi_horizon"] and d["forward_days"] != grid["forward_days"][0]:
            continue
        combinations.append(d)
    return combinations


def build_cmd(combo: dict, start: str, end: str) -> list[str]:
    """构建 CLI 参数列表。"""
    cmd = [
        sys.executable, "scripts/run_ml_backtest.py",
        "--factor-preset", FIXED["factor_preset"],
        "--train-years", str(FIXED["train_years"]),
        "--val-years", str(FIXED["val_years"]),
        "--universe-size", str(FIXED["universe_size"]),
        "--start", start,
        "--end", end,
        "--top-n", str(combo["top_n"]),
        "--rebalance-freq", str(combo["rebalance_freq"]),
    ]

    if combo["multi_horizon"]:
        cmd.append("--multi-horizon")
    else:
        cmd.extend(["--forward-days", str(combo["forward_days"])])

    if combo["regime"]:
        cmd.append("--regime")
    if combo["industry_neutralize"]:
        cmd.append("--industry-neutralize")
    if FIXED["dynamic"]:
        cmd.append("--dynamic")

    # 集成模式：不指定 --model 则走双模型集成
    return cmd


def combo_label(combo: dict) -> str:
    """生成可读的组合标签。"""
    horizon = "MH" if combo["multi_horizon"] else f"T+{combo['forward_days']}"
    regime = "R" if combo["regime"] else "noR"
    ind = "Ind" if combo["industry_neutralize"] else "noInd"
    return f"{horizon}_{regime}_{ind}_top{combo['top_n']}_rb{combo['rebalance_freq']}"


def parse_metrics_from_stdout(stdout: str) -> dict | None:
    """从 run_ml_backtest.py 输出中提取关键指标。"""
    import re
    metrics = {}

    # 主汇总行: "总收益: 45.23%, 年化: 12.34%, Sharpe: 1.56, 最大回撤: -15.67%"
    summary_pat = re.compile(
        r'总收益[：:]\s*([\d.-]+)%[，,]\s*年化[：:]\s*([\d.-]+)%[，,]\s*Sharpe[：:]\s*([\d.-]+)[，,]\s*最大回撤[：:]\s*([\d.-]+)%'
    )
    m = summary_pat.search(stdout)
    if m:
        metrics["total_return"] = float(m.group(1))
        metrics["cagr"] = float(m.group(2))
        metrics["sharpe"] = float(m.group(3))
        metrics["max_dd"] = abs(float(m.group(4))) / 100  # 转为小数

    return metrics if metrics else None


def compute_score(cagr: float, max_dd: float, sharpe: float) -> float:
    """综合评分：年化 - 2×最大回撤 + 0.5×夏普"""
    return cagr - 2.0 * abs(max_dd) * 100 + 0.5 * sharpe * 10


def run_grid(start: str, end: str, grid: dict, output: str = "output/grid_results.csv"):
    """主流程：遍历所有组合，运行回测，保存结果。"""
    combos = expand_combinations(grid)
    total = len(combos)
    logger.info(f"网格搜索: {total} 个组合, 区间 {start}-{end}")
    logger.info(f"固定参数: {FIXED}")

    results = []
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(combos):
        label = combo_label(combo)
        cmd = build_cmd(combo, start, end)
        cmd_str = " ".join(cmd)
        logger.info(f"[{i+1}/{total}] {label}")
        logger.debug(f"  CMD: {cmd_str}")

        t0 = time.time()
        import subprocess
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0

        metrics = parse_metrics_from_stdout(proc.stdout + proc.stderr)

        if metrics and all(k in metrics for k in ["cagr", "max_dd", "sharpe"]):
            score = compute_score(metrics["cagr"], metrics["max_dd"], metrics["sharpe"])
            row = {
                **combo,
                "label": label,
                "cagr": round(metrics["cagr"], 2),
                "sharpe": round(metrics["sharpe"], 2),
                "max_dd": round(metrics["max_dd"], 4),
                "total_return": round(metrics.get("total_return", 0), 2),
                "score": round(score, 2),
                "elapsed_s": round(elapsed, 1),
                "exit_code": proc.returncode,
            }
        else:
            row = {
                **combo,
                "label": label,
                "cagr": np.nan,
                "sharpe": np.nan,
                "max_dd": np.nan,
                "total_return": np.nan,
                "score": np.nan,
                "elapsed_s": round(elapsed, 1),
                "exit_code": proc.returncode,
            }

        results.append(row)

        # 增量保存
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

        # 实时输出排名
        valid = df.dropna(subset=["score"]).sort_values("score", ascending=False)
        if not valid.empty:
            best = valid.iloc[0]
            logger.info(f"  当前最佳: {best['label']} (cagr={best['cagr']}%, sharpe={best['sharpe']}, dd={best['max_dd']:.1%}, score={best['score']})")

    # ── 最终排名 ──
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    valid = df.dropna(subset=["score"]).sort_values("score", ascending=False)

    logger.info(f"\n{'='*80}")
    logger.info(f"网格搜索完成: {len(valid)}/{total} 个有效结果")
    logger.info(f"\nTop 10 (score = CAGR - 2×|DD|×100 + 0.5×Sharpe×10):")
    logger.info(f"{'Rank':<5} {'Label':<35} {'CAGR%':<10} {'Sharpe':<8} {'MaxDD':<8} {'Score':<8} {'Time':<8}")
    for rank, (_, row) in enumerate(valid.head(10).iterrows()):
        logger.info(
            f"{rank+1:<5} {row['label']:<35} "
            f"{row['cagr']:<10.2f} {row['sharpe']:<8.2f} {row['max_dd']:<8.1%} "
            f"{row['score']:<8.2f} {row['elapsed_s']:<8.0f}s"
        )

    output_path_abs = output_path.resolve()
    logger.info(f"\n结果已保存: {output_path_abs}")

    return valid


def update_defaults(best_row: pd.Series, config_path: str = "config/settings.py"):
    """将最优参数组合写入 TradingConfig 默认值。"""
    import re

    content = Path(config_path).read_text()

    updates = {
        "REBALANCE_FREQ": int(best_row["rebalance_freq"]),
        "TOP_N": int(best_row["top_n"]),
    }

    for key, val in updates.items():
        content = re.sub(
            rf'^{key}\s*=\s*\d+',
            f'{key} = {val}',
            content,
            flags=re.MULTILINE,
        )

    Path(config_path).write_text(content)
    logger.info(f"已更新 {config_path}: {updates}")


def print_best_cli(best_row: pd.Series):
    """打印最优组合的完整 CLI 命令。"""
    cmd_parts = [
        "python scripts/run_ml_backtest.py",
        f"--factor-preset {FIXED['factor_preset']}",
        "--dynamic",
        f"--top-n {int(best_row['top_n'])}",
        f"--rebalance-freq {int(best_row['rebalance_freq'])}",
    ]
    if best_row["multi_horizon"]:
        cmd_parts.append("--multi-horizon")
    if best_row["regime"]:
        cmd_parts.append("--regime")
    if best_row["industry_neutralize"]:
        cmd_parts.append("--industry-neutralize")
    if best_row.get("forward_days", 1) != 1:
        cmd_parts.append(f"--forward-days {int(best_row['forward_days'])}")

    logger.info("\n推荐默认 CLI:")
    logger.info(" \\\n    ".join(cmd_parts))


def main():
    parser = argparse.ArgumentParser(description="v1.12 参数网格搜索")
    parser.add_argument("--subset", action="store_true", help="使用精简网格（~72组）")
    parser.add_argument("--fast", action="store_true", help="快速验证网格（~24组）")
    parser.add_argument("--start", default="20220101", help="回测起始日期")
    parser.add_argument("--end", default="20260528", help="回测结束日期")
    parser.add_argument("--output", default="output/grid_results.csv", help="结果文件")
    parser.add_argument("--update-config", action="store_true", help="自动更新 config/settings.py 默认参数")
    args = parser.parse_args()

    if args.fast:
        grid = FAST_GRID
    elif args.subset:
        grid = SUBSET_GRID
    else:
        grid = FULL_GRID

    valid = run_grid(args.start, args.end, grid, args.output)

    if valid.empty:
        logger.error("没有成功的回测结果")
        sys.exit(1)

    best = valid.iloc[0]
    print_best_cli(best)

    if args.update_config:
        update_defaults(best)


if __name__ == "__main__":
    main()
