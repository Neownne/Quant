#!/usr/bin/env python3
"""策略实验室持续优化循环 —— 永不停止地搜索→回测→评判→循环。

用法:
    python scripts/run_lab_forever.py --start 2020-01-01 --parallel 2
    # Ctrl-C 优雅退出（完成当前轮次后停止）
"""
from __future__ import annotations

import os, sys, time, signal, argparse
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loguru import logger

from lab.variant import StrategyVariant
from lab.runner import LabRunner
from lab.judge import judge, print_report, save_report
from lab.grid import generate_grid_variants

VARIANTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "lab", "variants")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "lab", "reports")

_stop = False


def on_signal(sig, frame):
    global _stop
    logger.info(f"收到信号 {sig}，当前轮次完成后退出...")
    _stop = True


def run_forever(args):
    global _stop
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    runner = LabRunner(start=args.start, end=args.end or date.today().strftime("%Y-%m-%d"),
                       cash=args.cash)
    round_num = 0

    while not _stop:
        round_num += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"  第 {round_num} 轮")
        logger.info(f"{'='*60}")

        # 1. 搜索新策略（每3轮搜一次，避免重复）
        if round_num % 3 == 1:
            logger.info("搜索最新策略...")
            from lab.searcher import search_multi_rounds
            search_multi_rounds(num_rounds=1, output_dir=VARIANTS_DIR)

        # 2. 生成/刷新网格变体
        logger.info("刷新变体列表...")
        existing_names = {v.name for v in StrategyVariant.load_all(VARIANTS_DIR)}
        new_variants = generate_grid_variants()
        for v in new_variants:
            if v.name not in existing_names:
                v.to_json(os.path.join(VARIANTS_DIR, f"{v.name}.json"))
                logger.info(f"  新变体: {v.name}")

        # 3. 加载所有变体，找出未测试的
        all_variants = StrategyVariant.load_all(VARIANTS_DIR)
        untested = []
        for v in all_variants:
            metrics = runner._read_latest_metrics(v.name)
            if not metrics:
                untested.append(v)
            else:
                v._result = metrics

        if untested:
            logger.info(f"运行 {len(untested)} 个未测试变体...")
            results = runner.run_batch(untested, parallel=args.parallel)
            for r, v in zip(results, untested):
                if not r.get("error"):
                    v._result = r
        else:
            logger.info("没有未测试变体")

        # 4. 评判 + 报告
        all_results = []
        for v in all_variants:
            if v._result and not v._result.get("error"):
                all_results.append({"variant_name": v.name, **v._result})

        if all_results:
            ranked = judge(all_results)
            report_path = os.path.join(REPORTS_DIR,
                                       f"report_r{round_num}_{date.today().strftime('%Y%m%d')}.json")
            print_report(ranked)
            save_report(ranked, report_path)

            # 标记拒绝的变体（在 lab_experiments 中）
            try:
                from data.db import get_engine
                from sqlalchemy import text
                eng = get_engine()
                with eng.begin() as conn:
                    for r in ranked:
                        conn.execute(text("""
                            INSERT INTO lab_experiments (variant_name, composite_score, rank, verdict, variant_params_json)
                            VALUES (:vn, :sc, :rk, :ve, '{}')
                        """), {"vn": r["variant_name"], "sc": r["_score"],
                               "rk": r["_rank"], "ve": r["_verdict"]})
                eng.dispose()
            except Exception as e:
                logger.warning(f"lab_experiments 写入失败: {e}")

        # 5. 如果有被采纳的变体，下次搜索可以围绕它们的参数方向
        if _stop:
            break
        logger.info(f"第 {round_num} 轮完成，等待下一轮...\n")
        time.sleep(10)  # 轮间短暂间隔


def main():
    p = argparse.ArgumentParser(description="策略实验室持续循环")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--parallel", type=int, default=1)
    args = p.parse_args()

    os.makedirs(VARIANTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    logger.info("策略实验室持续循环启动")
    logger.info(f"基准区间: {args.start} → {args.end or '今天'}")
    logger.info(f"并行数: {args.parallel}")
    logger.info("Ctrl-C 优雅退出")
    run_forever(args)


if __name__ == "__main__":
    main()
