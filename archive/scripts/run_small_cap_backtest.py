#!/usr/bin/env python
"""小市值事件驱动回测 (SmallCapEngine)

每日扫描"低开反转"信号 → 持仓出场检查 → 生成买卖单。

用法:
    python scripts/run_small_cap_backtest.py
    python scripts/run_small_cap_backtest.py --start 20240101 --end 20260528 --stock-num 10
    python scripts/run_small_cap_backtest.py --universe-size 300 --stock-num 15
"""
import sys
import os
import argparse
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from config.settings import TradingConfig
from portfolio.small_cap_engine import SmallCapEngine


def load_price_data(engine, codes, start, end):
    """Load OHLCV for a set of codes over a date range.

    Returns a dict {pd.Timestamp: DataFrame(indexed by code)}.
    """
    if not codes:
        return {}

    codes_str = ",".join([f"'{c}'" for c in codes])
    df = pd.read_sql(
        text(f"""
            SELECT code, trade_date, open, high, low, close, volume, amount
            FROM stock_daily
            WHERE code IN ({codes_str})
              AND trade_date >= :start AND trade_date <= :end
            ORDER BY code, trade_date
        """),
        engine,
        params={"start": start, "end": end},
    )
    if df.empty:
        return {}

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return {dt: grp.set_index("code")
            for dt, grp in df.groupby("trade_date")}


def load_industry_map(engine):
    """Load code -> industry_sw1 mapping."""
    df = pd.read_sql(
        text("SELECT code, industry_sw1 FROM stock_industry"),
        engine,
    )
    if df.empty:
        return {}
    return dict(zip(df["code"], df["industry_sw1"]))


def compute_metrics(equity, daily_rets):
    """Compute CAGR, Sharpe, MaxDD from equity curve."""
    eq_vals = list(equity.values())
    if len(eq_vals) < 2:
        return {"annual_return": 0.0, "total_return": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0}

    total_return = float(eq_vals[-1] / eq_vals[0] - 1)
    n_years = max(len(eq_vals) / 252.0, 0.2)
    cagr = float((1 + total_return) ** (1.0 / n_years) - 1)

    dr_vals = [v for v in daily_rets.values() if v != 0.0]
    if dr_vals and np.std(dr_vals) > 0:
        sharpe = float(np.mean(dr_vals) / np.std(dr_vals) * np.sqrt(252))
    else:
        sharpe = 0.0

    peak = eq_vals[0]
    max_dd = 0.0
    for v in eq_vals:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return {
        "annual_return": cagr,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": float(max_dd),
    }


def save_results(engine, equity, daily_rets, metrics, start_date, end_date):
    """Write backtest results to backtest_results table."""
    try:
        with engine.begin() as conn:
            # Upsert strategy_configs
            sc = conn.execute(
                text("SELECT id FROM strategy_configs WHERE name = '小市'")
            ).fetchone()
            if not sc:
                sc = conn.execute(
                    text("""
                        INSERT INTO strategy_configs (name, type, description)
                        VALUES ('小市', 'static', '小市值事件驱动低开反转')
                        RETURNING id
                    """)
                ).fetchone()
            sid = sc[0]

            # Upsert strategy_versions
            sv = conn.execute(
                text("SELECT id FROM strategy_versions WHERE strategy_id = :sid"),
                {"sid": sid},
            ).fetchone()
            if not sv:
                sv = conn.execute(
                    text("""
                        INSERT INTO strategy_versions
                            (strategy_id, version, algorithm_type, feature_list_version)
                        VALUES (:sid, 'v1.0', 'event_driven', 'small_cap_v1')
                        RETURNING id
                    """),
                    {"sid": sid},
                ).fetchone()
            vid = sv[0]

            conn.execute(
                text("""
                    INSERT INTO backtest_results
                        (version_id, start_date, end_date, quality,
                         metrics_json, equity_curve_json, daily_returns_json)
                    VALUES (:vid, :s, :e, 'valid',
                            CAST(:m_json AS jsonb),
                            CAST(:eq_json AS jsonb),
                            CAST(:dr_json AS jsonb))
                """),
                {
                    "vid": vid,
                    "s": start_date,
                    "e": end_date,
                    "m_json": json.dumps(metrics),
                    "eq_json": json.dumps(equity),
                    "dr_json": json.dumps(daily_rets),
                },
            )
        return True
    except Exception as e:
        logger.error(f"写入DB失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="小市值事件驱动回测 (SmallCapEngine)")
    parser.add_argument("--start", default="20200101",
                        help="回测起始日期 (YYYYMMDD)")
    parser.add_argument("--end", default="20260528",
                        help="回测结束日期 (YYYYMMDD)")
    parser.add_argument("--stock-num", type=int, default=10,
                        help="持仓股数上限")
    parser.add_argument("--universe-size", type=int, default=300,
                        help="候选池大小 (按code排序取前N)")
    args = parser.parse_args()

    engine = get_engine()
    eg = SmallCapEngine(account_id=0, run_id=0,
                        stock_num=args.stock_num,
                        universe_size=args.universe_size)

    # ── 1. 加载交易日历 ──
    dates_df = pd.read_sql(
        text("""
            SELECT DISTINCT trade_date FROM stock_daily
            WHERE trade_date >= :start AND trade_date <= :end
            ORDER BY trade_date
        """),
        engine,
        params={"start": args.start, "end": args.end},
    )
    dates_df["trade_date"] = pd.to_datetime(dates_df["trade_date"])
    trade_dates = sorted(dates_df["trade_date"].tolist())
    if len(trade_dates) < 2:
        logger.error(f"交易日不足: {len(trade_dates)} 天")
        engine.dispose()
        sys.exit(1)

    logger.info(f"回测: {trade_dates[0].date()} ~ {trade_dates[-1].date()}, "
                f"{len(trade_dates)} 个交易日")

    # ── 2. 加载行业映射 ──
    industry_map = load_industry_map(engine)
    logger.info(f"行业映射: {len(industry_map)} 只")

    # ── 3. 加载初始价格数据 ──
    first_date_str = trade_dates[0].strftime("%Y-%m-%d")
    universe = eg.build_universe(engine, first_date_str, args.universe_size)
    all_loaded_codes = set(universe)
    price_by_date = load_price_data(engine, all_loaded_codes,
                                    args.start, args.end)
    logger.info(f"初始候选池: {len(universe)} 只, "
                f"价格覆盖: {len(price_by_date)} 个交易日")

    # ── 4. 模拟状态 ──
    cash = float(TradingConfig.INITIAL_CASH)
    positions: dict = {}
    peak_value = cash
    equity: dict = {}
    daily_rets: dict = {}
    last_universe_month = -1

    # ── 5. 日度模拟循环 ──
    for i, dt in enumerate(trade_dates):
        dt_str = dt.strftime("%Y-%m-%d")

        # 5a. 月度刷新候选池 + 增量加载新代码的行情
        if dt.month != last_universe_month:
            universe = eg.build_universe(engine, dt_str, args.universe_size)
            new_codes = set(universe) - all_loaded_codes
            # 也覆盖当前持仓（可能不在候选池中）
            all_needed = set(universe) | set(positions.keys())
            new_codes |= (all_needed - all_loaded_codes)

            if new_codes:
                new_price = load_price_data(engine, new_codes,
                                            args.start, args.end)
                for d, pdf in new_price.items():
                    if d in price_by_date:
                        price_by_date[d] = pd.concat(
                            [price_by_date[d], pdf]
                        ).groupby(level=0).last()
                    else:
                        price_by_date[d] = pdf
                all_loaded_codes |= new_codes
                logger.info(f"  {dt_str}: 新增{len(new_codes)}只行情, "
                            f"总加载{len(all_loaded_codes)}只")

            last_universe_month = dt.month
            logger.info(f"  {dt_str}: 候选池{len(universe)}只, "
                        f"持仓{len(positions)}只")

        # 5b. 今日/昨日行情
        today = price_by_date.get(dt)
        prev = price_by_date.get(trade_dates[i - 1]) if i > 0 else None

        # 5c. 构建 lookback (近15日 OHLC)
        lookback: dict = {}
        for code in universe:
            lb = []
            for j in range(max(0, i - 15), i):
                ld = trade_dates[j]
                ld_data = price_by_date.get(ld)
                if ld_data is not None and code in ld_data.index:
                    r = ld_data.loc[code]
                    lb.append({
                        "close": float(r["close"]),
                        "low": float(r["low"]),
                        "high": float(r["high"]),
                    })
            if lb:
                lookback[code] = lb

        # 5d. 执行单日周期
        result = eg.run_daily(
            dt,
            trade_dates[i - 1] if i > 0 else dt,
            today,
            prev,
            positions,
            cash,
            peak_value,
            lookback,
            industry_map,
        )

        positions = result["positions"]
        cash = result["cash"]
        peak_value = result["peak_value"]
        equity[dt_str] = result["total_value"]

        # 5e. 日度收益率
        prev_total = (
            list(equity.values())[-2]
            if len(equity) > 1
            else result["total_value"]
        )
        daily_rets[dt_str] = (
            float(result["total_value"] / prev_total - 1)
            if prev_total > 0
            else 0.0
        )

        if i % 500 == 0:
            logger.info(f"  {dt_str}: {len(positions)}只持仓, "
                        f"净值 {result['total_value']:,.0f}, "
                        f"现金 {cash:,.0f}")

    # ── 6. 计算指标 ──
    metrics = compute_metrics(equity, daily_rets)
    print(f"\n=== 小市值事件驱动 回测结果 ===")
    print(f"日期: {trade_dates[0].date()} ~ {trade_dates[-1].date()}")
    print(f"总收益: {metrics['total_return']:.2%}")
    print(f"年化收益: {metrics['annual_return']:.2%}")
    print(f"Sharpe: {metrics['sharpe']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"最终净值: {list(equity.values())[-1]:,.0f}")

    # ── 7. 写入 DB ──
    engine2 = get_engine()
    ok = save_results(
        engine2, equity, daily_rets, metrics,
        str(trade_dates[0].date()), str(trade_dates[-1].date()),
    )
    if ok:
        print("已写入 backtest_results (strategy='小市')")
    engine2.dispose()
    engine.dispose()


if __name__ == "__main__":
    main()
