"""批量回测编排器 —— 对每个变体：生成信号 → 跑回测 → 收集指标。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from lab.variant import StrategyVariant

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
SIGNALS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "signals")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LabRunner:
    """策略变体批量回测编排器。"""

    def __init__(self, start: str = "2020-01-01", end: str = "2026-06-14",
                 cash: float = 1_000_000):
        self.benchmark_start = start
        self.benchmark_end = end
        self.cash = cash
        os.makedirs(SIGNALS_DIR, exist_ok=True)

    def _signals_path(self, variant_name: str) -> str:
        safe = variant_name.replace("/", "_").replace(" ", "_")
        return os.path.join(SIGNALS_DIR, f"bt_signals_{safe}.csv")

    def run_one(self, variant: StrategyVariant) -> dict:
        """运行单个变体，返回包含 metrics 和 variant_name 的 dict。"""
        t0 = time.time()
        logger.info(f"── 变体: {variant.name} ──")

        # 1. 生成信号
        signals_csv = self._signals_path(variant.name)
        gen_args = [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "gen_signals.py"),
            "--start", self.benchmark_start,
            "--end", self.benchmark_end,
            "--top-n", str(max(variant.top_n * 4, 20)),
            "--out", signals_csv,
        ]
        gen_args.extend(variant.to_gen_signals_args())

        logger.debug(f"  gen_signals: {variant.name}")
        r = subprocess.run(gen_args, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if r.returncode != 0:
            logger.error(f"  gen_signals 失败: {r.stderr[-500:]}")
            return {"variant_name": variant.name, "error": "gen_signals_failed", "stderr": r.stderr[-500:]}

        # 2. 跑回测
        bt_args = [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "bt_backtest.py"),
            "--start", self.benchmark_start,
            "--end", self.benchmark_end,
            "--cash", str(int(self.cash)),
        ]
        bt_args.extend(variant.to_bt_args(signals_csv))

        logger.debug(f"  bt_backtest: {variant.name}")
        r = subprocess.run(bt_args, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if r.returncode != 0:
            logger.error(f"  bt_backtest 失败: {r.stderr[-500:]}")
            return {"variant_name": variant.name, "error": "bt_backtest_failed", "stderr": r.stderr[-500:]}

        # 3. 从 DB 读取最新指标
        metrics = self._read_latest_metrics(variant.name)
        elapsed = time.time() - t0
        if metrics:
            logger.info(f"  {variant.name}: Sharpe={metrics.get('sharpe', 'N/A')}, "
                        f"MDD={metrics.get('max_drawdown', 'N/A')}, "
                        f"Return={metrics.get('return', 'N/A')} "
                        f"({elapsed:.0f}s)")
        else:
            logger.warning(f"  {variant.name}: 未能读取回测指标 ({elapsed:.0f}s)")

        return {"variant_name": variant.name, "elapsed": elapsed, **metrics} if metrics else \
               {"variant_name": variant.name, "error": "metrics_read_failed"}

    def _read_latest_metrics(self, variant_label: str) -> dict | None:
        """从 backtest_results 读取该 variant 的最新指标。"""
        try:
            eng = get_engine()
            with eng.connect() as conn:
                row = conn.execute(text("""
                    SELECT m.metrics_json
                    FROM backtest_results m
                    JOIN strategy_versions v ON v.id = m.version_id
                    JOIN strategy_configs c ON c.id = v.strategy_id
                    WHERE c.name = 'limit_up' AND v.version = :ver
                    ORDER BY m.created_at DESC LIMIT 1
                """), {"ver": variant_label}).mappings().first()
            eng.dispose()
            if row:
                return json.loads(row["metrics_json"]) if isinstance(row["metrics_json"], str) else row["metrics_json"]
        except Exception as e:
            logger.warning(f"  读取指标失败: {e}")
        return None

    def run_batch(self, variants: list[StrategyVariant],
                  parallel: int = 1) -> list[dict]:
        """批量运行变体，可选并行。"""
        if not variants:
            logger.warning("没有变体可运行")
            return []

        logger.info(f"═══ 批量回测 {len(variants)} 个变体 "
                     f"({self.benchmark_start} → {self.benchmark_end}) ═══")

        if parallel <= 1:
            results = []
            for i, v in enumerate(variants):
                logger.info(f"[{i+1}/{len(variants)}] {v.name}")
                results.append(self.run_one(v))
            return results

        # 并行模式：每个 worker 独立进程
        logger.info(f"  并行模式: {parallel} workers")
        results = []
        with ProcessPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(_run_one_in_subprocess, v, self.benchmark_start,
                                 self.benchmark_end, self.cash): v for v in variants}
            for i, fut in enumerate(as_completed(futures)):
                v = futures[fut]
                try:
                    r = fut.result(timeout=600)
                    results.append(r)
                    logger.info(f"[{i+1}/{len(variants)}] {v.name}: "
                                f"Sharpe={r.get('sharpe', 'N/A')}")
                except Exception as e:
                    logger.error(f"[{i+1}/{len(variants)}] {v.name}: {e}")
                    results.append({"variant_name": v.name, "error": str(e)})
        return results


def _run_one_in_subprocess(variant: StrategyVariant, start: str, end: str,
                           cash: float) -> dict:
    """子进程入口（并行模式），重新创建 runner 避免 DB 连接冲突。"""
    runner = LabRunner(start=start, end=end, cash=cash)
    return runner.run_one(variant)
