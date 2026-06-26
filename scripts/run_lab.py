#!/usr/bin/env python3
"""策略实验室 CLI —— 搜索/运行/评判策略变体。

用法:
    python scripts/run_lab.py list                          # 列出所有变体
    python scripts/run_lab.py run --variant E5_wide_mcap    # 运行单个变体
    python scripts/run_lab.py run --all --parallel 2        # 运行所有变体
    python scripts/run_lab.py report                        # 生成排名报告
    python scripts/run_lab.py auto --rounds 2               # 全自动管线
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from lab.variant import StrategyVariant, E4_BASELINE
from lab.runner import LabRunner
from lab.judge import judge, print_report, save_report
from lab.grid import generate_grid_variants

VARIANTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "lab", "variants")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "lab", "reports")


def cmd_list(args):
    """列出所有变体。"""
    variants = StrategyVariant.load_all(VARIANTS_DIR)
    if not variants:
        print("暂无变体。运行 'python scripts/run_lab.py grid' 生成基线变体。")
        return
    print(f"\n{'='*70}")
    print(f"  策略变体列表 ({len(variants)} 个)")
    print(f"{'='*70}")
    for v in variants:
        mc_max = v.effective('mcap_max')
        mcap = f"{v.effective('mcap_min')}-{mc_max}亿" if mc_max != float('inf') else f"≥{v.effective('mcap_min')}亿"
        price = f"{v.effective('price_min')}-{v.effective('price_max')}元"
        print(f"  {v.name:25s} | {mcap:15s} | {price:12s} | "
              f"lookback={v.effective('lu_lookback'):2d} | lu>{v.effective('lu_count')} | "
              f"top_n={v.top_n} | {v.source}")
    print()


def cmd_run(args):
    """运行回测。"""
    runner = LabRunner(start=args.start, end=args.end or date.today().strftime("%Y-%m-%d"),
                       cash=args.cash)

    if args.all:
        variants = StrategyVariant.load_all(VARIANTS_DIR)
        if not variants:
            logger.error("没有变体可运行。先执行: python scripts/run_lab.py grid")
            return
    elif args.variant:
        variants = [StrategyVariant.from_json(
            os.path.join(VARIANTS_DIR, f"{args.variant}.json")
        )]
    else:
        logger.error("请指定 --variant <name> 或 --all")
        return

    results = runner.run_batch(variants, parallel=args.parallel)

    # 简单打印摘要
    print(f"\n{'='*70}")
    print(f"  回测结果摘要 ({args.start} → {args.end})")
    print(f"{'='*70}")
    for r in results:
        if r.get("error"):
            print(f"  {r['variant_name']:25s} ❌ {r['error']}")
        else:
            sh = r.get('sharpe')
            md = r.get('max_drawdown')
            ret = r.get('return')
            sh_str = f"{sh:>6.2f}" if isinstance(sh, (int, float)) else str(sh)[:6]
            md_str = f"{md:>7.1%}" if isinstance(md, (int, float)) else str(md)[:7]
            ret_str = f"{ret:>7.1%}" if isinstance(ret, (int, float)) else str(ret)[:7]
            print(f"  {r['variant_name']:25s} | Sharpe={sh_str} | "
                  f"MDD={md_str} | Return={ret_str}")


def cmd_report(args):
    """生成排名报告。"""
    from lab.runner import LabRunner
    runner = LabRunner(start=args.start, end=args.end or date.today().strftime("%Y-%m-%d"))
    variants = StrategyVariant.load_all(VARIANTS_DIR)

    # 从 DB 收集所有 variant 的指标
    results = []
    for v in variants:
        metrics = runner._read_latest_metrics(v.name)
        if metrics:
            results.append({"variant_name": v.name, **metrics})
        else:
            logger.warning(f"  {v.name}: 无回测数据，跳过")

    if not results:
        logger.error("没有回测数据。先执行: python scripts/run_lab.py run --all")
        return

    ranked = judge(results)
    print_report(ranked)
    report_path = os.path.join(REPORTS_DIR, f"report_{date.today().strftime('%Y%m%d')}.json")
    save_report(ranked, report_path)


def cmd_auto(args):
    """全自动管线：grid → run → report。"""
    logger.info("═══ 全自动策略实验室 ═══")

    # 1. 生成网格变体（不覆盖已有的手动变体）
    logger.info("Step 1/4: 生成网格变体...")
    existing = {v.name for v in StrategyVariant.load_all(VARIANTS_DIR)}
    grid_variants = generate_grid_variants()
    new_count = 0
    for v in grid_variants:
        if v.name not in existing:
            v.to_json(os.path.join(VARIANTS_DIR, f"{v.name}.json"))
            new_count += 1
    logger.info(f"  新增 {new_count} 个网格变体（已有 {len(existing)} 个跳过）")

    # 2. 搜索（可选）
    if args.rounds > 0:
        logger.info(f"Step 2/4: 多轮搜索 ({args.rounds} 轮)...")
        from lab.searcher import search_multi_rounds
        search_multi_rounds(num_rounds=args.rounds, output_dir=VARIANTS_DIR)

    # 3. 运行
    all_variants = StrategyVariant.load_all(VARIANTS_DIR)
    logger.info(f"Step 3/4: 运行 {len(all_variants)} 个变体...")
    runner = LabRunner(start=args.start, end=args.end or date.today().strftime("%Y-%m-%d"),
                       cash=args.cash)
    results = runner.run_batch(all_variants, parallel=args.parallel)

    # 4. 报告
    logger.info("Step 4/4: 生成报告...")
    valid = [r for r in results if not r.get("error")]
    if valid:
        ranked = judge(valid)
        print_report(ranked)
        report_path = os.path.join(REPORTS_DIR, f"report_{date.today().strftime('%Y%m%d')}.json")
        save_report(ranked, report_path)


def main():
    p = argparse.ArgumentParser(description="策略实验室 — 自动化策略搜索/测试/评判")
    sub = p.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="列出所有变体")

    # run
    run_p = sub.add_parser("run", help="运行回测")
    run_p.add_argument("--variant", type=str, default=None, help="变体名称")
    run_p.add_argument("--all", action="store_true", help="运行所有变体")
    run_p.add_argument("--start", default="2020-01-01")
    run_p.add_argument("--end", default=None)
    run_p.add_argument("--cash", type=float, default=1_000_000)
    run_p.add_argument("--parallel", type=int, default=1)

    # report
    rep_p = sub.add_parser("report", help="生成排名报告")
    rep_p.add_argument("--start", default="2020-01-01")
    rep_p.add_argument("--end", default=None)

    # auto
    auto_p = sub.add_parser("auto", help="全自动管线")
    auto_p.add_argument("--start", default="2020-01-01")
    auto_p.add_argument("--end", default=None)
    auto_p.add_argument("--cash", type=float, default=1_000_000)
    auto_p.add_argument("--parallel", type=int, default=1)
    auto_p.add_argument("--rounds", type=int, default=2, help="搜索轮数（0=跳过搜索）")

    # grid
    sub.add_parser("grid", help="生成网格搜索变体")

    # search
    search_p = sub.add_parser("search", help="多轮 Web 搜索策略")
    search_p.add_argument("--rounds", type=int, default=3)
    search_p.add_argument("--query", type=str, default=None, help="单轮搜索（覆盖默认 queries）")

    args = p.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "auto":
        cmd_auto(args)
    elif args.command == "grid":
        from lab.grid import generate_grid_variants
        variants = generate_grid_variants()
        for v in variants:
            v.to_json(os.path.join(VARIANTS_DIR, f"{v.name}.json"))
        logger.info(f"生成 {len(variants)} 个网格变体 → {VARIANTS_DIR}")
    elif args.command == "search":
        from lab.searcher import search_multi_rounds
        search_multi_rounds(num_rounds=args.rounds, output_dir=VARIANTS_DIR)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
