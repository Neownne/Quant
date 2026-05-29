#!/usr/bin/env python
"""初始化模拟盘：创建 paper_account + paper_runs 种子数据。

用法:
    python scripts/init_paper_trading.py              # 创建默认账户和 run
    python scripts/init_paper_trading.py --name "牛市策略" --capital 2000000
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger
from sqlalchemy import text
from data.db import get_engine
from config.settings import TradingConfig


def main():
    parser = argparse.ArgumentParser(description="初始化模拟盘种子数据")
    parser.add_argument("--name", default="ML-日频动态多因子", help="账户名称")
    parser.add_argument("--capital", type=float, default=TradingConfig.INITIAL_CASH)
    parser.add_argument("--strategy-name", default="ML-Regime-日频", help="策略名称")
    args = parser.parse_args()

    engine = get_engine()

    with engine.begin() as conn:
        # 1. 检查是否已存在
        existing = conn.execute(
            text("SELECT id FROM paper_account WHERE name = :n"),
            {"n": args.name}
        ).fetchone()

        if existing:
            account_id = existing[0]
            logger.info(f"账户已存在: id={account_id}")
        else:
            r = conn.execute(text("""
                INSERT INTO paper_account (name, initial_capital, cash, created_at)
                VALUES (:n, :c, :c, NOW())
                RETURNING id
            """), {"n": args.name, "c": args.capital})
            account_id = r.fetchone()[0]
            logger.info(f"创建账户: id={account_id}, 本金={args.capital:,.0f}")

        # 2. 检查 strategy_configs 是否存在（可选）
        sc = conn.execute(text(
            "SELECT id FROM strategy_configs WHERE name = :n LIMIT 1"
        ), {"n": args.strategy_name}).fetchone()
        if sc:
            strategy_id = sc[0]
            logger.info(f"策略配置: strategy_id={strategy_id}")
        else:
            conn.execute(text("""
                INSERT INTO strategy_configs (name, type, description)
                VALUES (:n, 'ml', 'ML日频动态多因子 Regime自适应策略')
                ON CONFLICT (name) DO NOTHING
            """), {"n": args.strategy_name})
            strategy_id = conn.execute(
                text("SELECT id FROM strategy_configs WHERE name = :n"),
                {"n": args.strategy_name}
            ).fetchone()[0]
            logger.info(f"创建策略配置: strategy_id={strategy_id}")

        # 3. 确保 strategy_versions 有对应的版本（PaperEngine 写入 backtest_results 需要）
        sv = conn.execute(text(
            "SELECT id FROM strategy_versions WHERE strategy_id = :sid ORDER BY created_at DESC LIMIT 1"
        ), {"sid": strategy_id}).fetchone()
        if sv:
            version_id = sv[0]
            logger.info(f"策略版本: version_id={version_id}")
        else:
            r = conn.execute(text("""
                INSERT INTO strategy_versions (strategy_id, version, algorithm_type, feature_list_version, model_file_path, created_at)
                VALUES (:sid, 'v1.0', 'xgboost+lgbm_ensemble', 'v1.12_daily', 'Regime自适应日频ML策略', NOW())
                RETURNING id
            """), {"sid": strategy_id})
            version_id = r.fetchone()[0]
            logger.info(f"创建策略版本: version_id={version_id}")

        # 4. 创建 paper_runs（当前活跃的模拟盘运行）
        active_run = conn.execute(text(
            "SELECT id FROM paper_runs WHERE status = 'running' AND strategy_id = :sid LIMIT 1"
        ), {"sid": strategy_id}).fetchone()
        if active_run:
            run_id = active_run[0]
            logger.info(f"活跃 run 已存在: run_id={run_id}")
        else:
            r = conn.execute(text("""
                INSERT INTO paper_runs (strategy_id, version_id, start_date, initial_capital, status)
                VALUES (:sid, :vid, CURRENT_DATE, :c, 'running')
                RETURNING id
            """), {"sid": strategy_id, "vid": version_id, "c": args.capital})
            run_id = r.fetchone()[0]
            logger.info(f"创建 run: run_id={run_id}")

    engine.dispose()

    print(f"\n===== 模拟盘初始化完成 =====")
    print(f"paper_account.id  = {account_id}")
    print(f"strategy_configs.id = {strategy_id}")
    print(f"strategy_versions.id = {version_id}")
    print(f"paper_runs.id    = {run_id}")
    print(f"\n运行模拟盘时使用: --account-id {account_id} --run-id {run_id}")


if __name__ == "__main__":
    main()
