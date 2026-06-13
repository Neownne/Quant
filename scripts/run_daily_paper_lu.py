#!/usr/bin/env python
"""涨停 Top-5 每日模拟盘。

每日收盘后运行：
  1. 同步最新日线数据
  2. 4条件筛选（市值/股价/均线/涨停次数，去跌停）→ 取Top-5
  3. 对比当前持仓，生成买卖信号
  4. T+1 开盘执行 → 写入 paper_* 表

用法:
    python scripts/run_daily_paper_lu.py                # 今日
    python scripts/run_daily_paper_lu.py --date 2026-06-05  # 指定日期
    python scripts/run_daily_paper_lu.py --dry-run       # 试运行
"""
import sys, os, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import date, timedelta
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from data.sync import sync_stock_daily, sync_daily_extra
from data.loader import (
    load_daily_data, load_mcap_data, get_latest_trade_date, get_stock_basic,
)
from strategies.limit_up.base import LimitUpParams, run_screening
from strategies.limit_up.execution import execute
from strategies.limit_up.pnl import update_daily_pnl
from config.settings import TradingConfig

# ── 策略参数 ──
ACCOUNT_ID = 1
RUN_ID = 1
TOP_N = 5
MIN_LISTED_DAYS = 120

LU_PARAMS = LimitUpParams(
    mcap_min=30.0, mcap_max=500.0,
    price_min=5.0, price_max=63.0,
    lu_pct=TradingConfig.LIMIT_UP_PCT,
    lu_lookback=20, lu_count=1, min_conditions=4,
)


def sync_data(engine):
    """增量同步日线 + 市值数据。"""
    today = date.today().strftime("%Y-%m-%d")
    logger.info(f"同步数据至 {today} ...")
    sync_stock_daily(engine, start_date=today, workers=8)
    sync_daily_extra(engine, start_date=today, workers=8)


def _export_signals_csv(engine, trade_date, signals):
    """导出待执行信号到 CSV 文件。"""
    if not signals:
        return
    try:
        codes = [s[0] for s in signals]
        names = pd.read_sql(
            text("SELECT code, name FROM stock_basic WHERE code = ANY(:codes)"),
            engine,
            params={"codes": codes},
        )
        name_map = dict(zip(names["code"], names["name"]))

        chg_df = pd.read_sql(
            text("""
                SELECT code,
                       (close - LAG(close) OVER (PARTITION BY code ORDER BY trade_date))
                       / NULLIF(LAG(close) OVER (PARTITION BY code ORDER BY trade_date), 0) * 100 AS chg_pct
                FROM stock_daily WHERE code = ANY(:codes) AND trade_date <= :td
                ORDER BY code, trade_date DESC
            """),
            engine,
            params={"codes": codes, "td": trade_date},
        )
        chg_df = chg_df.dropna(subset=["chg_pct"])
        chg_map = dict(zip(chg_df.groupby("code").first().index, chg_df.groupby("code").first()["chg_pct"]))

        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'paper_signals')
        os.makedirs(out_dir, exist_ok=True)
        date_str = str(trade_date)[:10]
        csv_path = os.path.join(out_dir, f'signals_{date_str}.csv')

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['数据日期', '排名', '股票代码', '股票名称', '评分', '收盘价', '近交易日涨跌幅(%)'])
            for rank, s in enumerate(signals, 1):
                code = s[0]
                writer.writerow([
                    date_str, rank, code,
                    name_map.get(code, '?'),
                    round(float(s[1]), 4),
                    round(float(s[2]), 2),
                    round(float(chg_map.get(code, 0)), 2)
                ])
        logger.info(f"  CSV导出: {csv_path} ({len(signals)}条)")
    except Exception as e:
        logger.warning(f"CSV导出失败: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="回测日期 YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()

    engine = get_engine()

    # ── 1. 同步数据 ──
    if not args.no_sync:
        sync_data(engine)

    # ── 2. 确定交易日 ──
    trade_date = pd.Timestamp(args.date) if args.date else get_latest_trade_date(engine)
    trade_date = pd.Timestamp(trade_date)
    logger.info(f"交易日: {trade_date.date()}")

    # ── 3. 候选池 ──
    basic = get_stock_basic(engine, trade_date, min_listed_days=MIN_LISTED_DAYS)
    code_set = set(basic["code"])
    name_map = dict(zip(basic["code"], basic["name"]))
    logger.info(f"候选池: {len(code_set)} 只")

    # ── 4. 加载数据 ──
    pre_start = trade_date - timedelta(days=LU_PARAMS.lu_lookback + 30)
    daily = load_daily_data(engine, code_set, pre_start, trade_date, cols=["open", "close"])
    extra = load_mcap_data(engine, code_set, pre_start, trade_date)
    logger.info(f"日线: {len(daily)} 行 | 市值: {len(extra)} 行")

    # ── 5. 筛选 ──
    signals = run_screening(trade_date, daily, extra, code_set, LU_PARAMS)
    if not signals:
        logger.warning("无股票通过筛选，检查市场状态。")
        if not args.dry_run:
            update_daily_pnl(engine, ACCOUNT_ID, RUN_ID, trade_date)
        engine.dispose()
        return

    logger.info(f"筛选结果: {len(signals)} 只")
    for i, s in enumerate(signals[:15], 1):
        logger.info(f"  {i}. {s[0]} {name_map.get(s[0], '?')} 涨停{s[1]}次 收盘{s[2]:.2f}")

    # ── 6. 估值（先记今日净值，再执行明日交易）──
    if not args.dry_run:
        update_daily_pnl(engine, ACCOUNT_ID, RUN_ID, trade_date)

    # ── 7. 写信号（最多20条；web展示用）──
    if not args.dry_run:
        with engine.begin() as conn:
            for i, s in enumerate(signals[:20], 1):
                conn.execute(text("""
                    INSERT INTO paper_signals (run_id, signal_date, stock_code, predicted_score, rank)
                    VALUES (:rid, :sd, :code, :score, :rank)
                    ON CONFLICT (run_id, signal_date, stock_code) DO UPDATE SET
                        predicted_score = :score2, rank = :rank2
                """), {
                    "rid": RUN_ID, "sd": trade_date, "code": s[0],
                    "score": float(s[1]), "rank": i,
                    "score2": float(s[1]), "rank2": i,
                })

    # ── 8. 导出待执行信号 CSV ──
    _export_signals_csv(engine, trade_date, signals)

    # ── 9. 执行（Top-N）──
    from strategies.limit_up.pnl import get_current_positions
    positions = get_current_positions(engine, RUN_ID)
    execute(
        engine, account_id=ACCOUNT_ID, run_id=RUN_ID,
        trade_date=trade_date, signals=signals, positions=positions,
        top_n=TOP_N, stop_loss_pct=TradingConfig.STOP_LOSS_PCT,
        dry_run=args.dry_run,
    )

    engine.dispose()


if __name__ == "__main__":
    main()
