#!/usr/bin/env python
"""小市值模拟盘每日驱动"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from loguru import logger
from sqlalchemy import text
from data.db import get_engine
from config.settings import TradingConfig
from portfolio.small_cap_engine import SmallCapEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-id", type=int, default=16)
    parser.add_argument("--run-id", type=int, default=3)
    parser.add_argument("--stock-num", type=int, default=10)
    parser.add_argument("--universe-size", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    eg = SmallCapEngine(account_id=args.account_id, run_id=args.run_id,
                        stock_num=args.stock_num, universe_size=args.universe_size)

    # Get latest two trading dates
    latest = pd.read_sql(text("SELECT MAX(trade_date) FROM stock_daily"), engine).iloc[0,0]
    prev = pd.read_sql(text(f"SELECT MAX(trade_date) FROM stock_daily WHERE trade_date < '{latest}'"), engine).iloc[0,0]
    logger.info(f"数据日: {prev} -> {latest}")

    # Load positions and cash
    with engine.connect() as conn:
        row = conn.execute(text("SELECT cash FROM paper_account WHERE id=:aid"),{"aid":args.account_id}).fetchone()
        cash = float(row[0]) if row else TradingConfig.INITIAL_CASH
        pos_rows = conn.execute(text("""
            SELECT stock_code, SUM(quantity), SUM(entry_price*quantity)/NULLIF(SUM(quantity),0)
            FROM paper_positions WHERE run_id=:rid AND exit_date IS NULL
            GROUP BY stock_code
        """),{"rid":args.run_id}).fetchall()
        positions = {}
        for r in pos_rows:
            if int(r[1])>0:
                positions[str(r[0])]={"shares":int(r[1]),"cost":float(r[2] or 0),"today_high":float(r[2] or 0)}
    logger.info(f"持仓: {len(positions)}只, 现金: {cash:,.0f}")

    # Load universe + price data
    universe = eg.build_universe(engine, str(latest), args.universe_size)
    all_codes = list(set(universe) | set(positions.keys()))
    cl = ",".join([f"'{c}'" for c in all_codes])
    today_data = pd.read_sql(text(f"""
        SELECT code, open, high, low, close FROM stock_daily
        WHERE code IN ({cl}) AND trade_date='{latest}'
    """), engine).set_index("code")
    prev_data = pd.read_sql(text(f"""
        SELECT code, open, high, low, close FROM stock_daily
        WHERE code IN ({cl}) AND trade_date='{prev}'
    """), engine).set_index("code")

    # Industry map
    ind_df = pd.read_sql(text(f"SELECT code, industry_sw1 FROM stock_industry WHERE code IN ({cl})"), engine)
    industry_map = dict(zip(ind_df["code"], ind_df["industry_sw1"]))

    # Lookback data
    lookback = {}
    for code in all_codes:
        hist = pd.read_sql(text(f"""
            SELECT close, low, high FROM stock_daily
            WHERE code='{code}' AND trade_date<'{latest}' ORDER BY trade_date DESC LIMIT 15
        """), engine)
        if not hist.empty:
            lookback[code] = hist.to_dict("records")

    # Peak value
    peak_row = conn.execute(text("SELECT COALESCE(MAX(total_value),0) FROM paper_daily_pnl WHERE account_id=:aid"),{"aid":args.account_id}).fetchone()
    peak = float(peak_row[0]) if peak_row else cash

    # Run
    result = eg.run_daily(latest, prev, today_data, prev_data, positions, cash, peak, lookback, industry_map)

    if args.dry_run:
        logger.info(f"[DRY RUN] 买入{result['n_buys']} 卖出{result['n_sells']} 总资产{result['total_value']:,.0f}")
    else:
        logger.info(f"执行完成: 买入{result['n_buys']} 卖出{result['n_sells']} 总资产{result['total_value']:,.0f}")

    engine.dispose()

if __name__ == "__main__":
    main()
