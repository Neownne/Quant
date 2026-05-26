#!/usr/bin/env python3
"""同步 60 分钟 K 线数据到 stock_minute 表。

排除: ST / 上市 <180 天 / 北交所(4开头) / 科创板(8开头，无 Sina 分钟数据)。
速率: 每只间隔 0.3s，单只失败重试 3 次。

用法:
    python3 scripts/sync_minute_data.py --limit 10   # 只同步 10 只测试
    python3 scripts/sync_minute_data.py               # 全量同步
"""
import sys
import time
import argparse
sys.path.insert(0, ".")

import pandas as pd
from sqlalchemy import text
from data.db import get_engine, upsert_df, init_db
from data.fetcher import fetch_minute_data

PERIOD = "60"
SLEEP = 0.3
MAX_RETRIES = 3


def get_eligible_codes(limit: int | None = None) -> list[str]:
    engine = get_engine()
    sql = """
        SELECT DISTINCT b.code FROM stock_basic b
        INNER JOIN stock_daily d ON b.code = d.code
        WHERE b.is_st = FALSE
          AND b.list_date <= CURRENT_DATE - INTERVAL '180 days'
          AND b.code NOT LIKE '4%'
          AND b.code NOT LIKE '8%'
        ORDER BY b.code
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    codes = pd.read_sql(text(sql), engine)["code"].tolist()
    engine.dispose()
    return codes


def main():
    parser = argparse.ArgumentParser(description="同步 60 分钟 K 线")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量，用于测试")
    args = parser.parse_args()

    init_db()
    codes = get_eligible_codes(limit=args.limit)
    print(f"共 {len(codes)} 只股票待同步")

    success = 0
    fail = 0
    total_bars = 0

    for i, code in enumerate(codes):
        ok = False
        for attempt in range(MAX_RETRIES):
            try:
                df = fetch_minute_data(code, period=PERIOD, adjust="qfq")
                if df.empty:
                    ok = True
                    break
                upsert_df(df, "stock_minute")
                n = len(df)
                total_bars += n
                success += 1
                ok = True
                print(f"[{i+1}/{len(codes)}] {code} OK ({n} bars)")
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    fail += 1
                    print(f"[{i+1}/{len(codes)}] {code} FAIL: {e}")
        time.sleep(SLEEP)

    print(f"\n完成: 成功 {success}, 失败 {fail}, 总 bar {total_bars}")


if __name__ == "__main__":
    main()
