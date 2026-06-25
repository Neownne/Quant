#!/usr/bin/env python
"""批量同步 PE TTM 数据到 DB（tushare daily_basic）。

用法:
    python scripts/sync_pe_data.py              # 全量同步（2015至今，约30分钟）
    python scripts/sync_pe_data.py --recent     # 仅同步最近1个月

注意: tushare 免费版 daily_basic 限流 1次/分钟（单股票）或 1次/小时（全市场）。
      全市场一次拉取后写入 stock_daily_extra.pe 列。
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.sync_tushare import _init_pro
from data.db import get_engine
from sqlalchemy import text
import pandas as pd
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent", action="store_true", help="仅同步最近1个月")
    args = parser.parse_args()

    pro = _init_pro()
    engine = get_engine()

    if args.recent:
        start, end = '20260501', '20260624'
    else:
        start, end = '20150101', '20260624'

    logger.info(f"拉取 PE TTM: {start} ~ {end}")
    t0 = time.time()

    # 全市场一次性拉取（1次/小时限流）
    df = pro.daily_basic(ts_code='', start_date=start, end_date=end,
                         fields='ts_code,trade_date,pe_ttm,close,total_mv')

    if df.empty:
        logger.error("无数据返回")
        return

    # 清洗
    df['code'] = df['ts_code'].str.replace('.SH','').str.replace('.SZ','').str.replace('.BJ','')
    df = df[df['code'].str.match(r'^\d{6}$')]
    df['pe_ttm'] = pd.to_numeric(df['pe_ttm'], errors='coerce')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.dropna(subset=['pe_ttm', 'trade_date', 'code'])

    logger.info(f"共 {len(df)} 行, {df['code'].nunique()} 只股票, {df['trade_date'].nunique()} 个交易日")
    logger.info(f"PE TTM: {df['pe_ttm'].min():.1f} ~ {df['pe_ttm'].max():.1f}")

    # 写入 DB（更新 stock_daily_extra.pe）
    # 使用临时表 + ON CONFLICT UPDATE 避免重复
    logger.info("写入 DB...")
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(df), 5000):
            chunk = df.iloc[i:i+5000]
            for _, row in chunk.iterrows():
                conn.execute(text("""
                    INSERT INTO stock_daily_extra (code, trade_date, pe)
                    VALUES (:code, :trade_date, :pe)
                    ON CONFLICT (code, trade_date) DO UPDATE SET pe = :pe2
                """), {
                    "code": row['code'],
                    "trade_date": row['trade_date'].strftime('%Y-%m-%d'),
                    "pe": float(row['pe_ttm']) if pd.notna(row['pe_ttm']) else None,
                    "pe2": float(row['pe_ttm']) if pd.notna(row['pe_ttm']) else None,
                })
            total += len(chunk)
            if i % 50000 == 0:
                logger.info(f"  {total}/{len(df)}")

    engine.dispose()
    logger.success(f"同步完成: {total} 条 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
