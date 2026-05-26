#!/usr/bin/env python3
"""同步 60 分钟 K 线数据到 stock_minute 表。

排除: ST / 上市 <180 天 / 北交所(4开头) / 科创板(8开头，无 Sina 分钟数据)。
速率: 每只间隔 0.3s，单只超时 30s，失败重试 3 次。

用法:
    python3 scripts/sync_minute_data.py --limit 10       # 只同步 10 只测试
    python3 scripts/sync_minute_data.py --skip 432        # 跳过前 432 只续传
    python3 scripts/sync_minute_data.py                   # 全量同步
"""
import sys
import time
import signal
import argparse
sys.path.insert(0, ".")

import pandas as pd
from sqlalchemy import text
from data.db import get_engine, upsert_df, init_db
from data.fetcher import fetch_minute_data

PERIOD = "60"
SLEEP = 0.3
MAX_RETRIES = 3
API_TIMEOUT = 30  # 单只股票 API 超时秒数


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("API 调用超时")


def get_eligible_codes(limit: int | None = None, skip: int = 0) -> list[str]:
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
    if skip > 0:
        sql += f" OFFSET {int(skip)}"
    codes = pd.read_sql(text(sql), engine)["code"].tolist()
    engine.dispose()
    return codes


def main():
    parser = argparse.ArgumentParser(description="同步 60 分钟 K 线")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 只股票")
    args = parser.parse_args()

    init_db()
    codes = get_eligible_codes(limit=args.limit, skip=args.skip)
    print(f"共 {len(codes)} 只股票待同步", flush=True)

    success = 0
    fail = 0
    total_bars = 0

    for i, code in enumerate(codes):
        ok = False
        for attempt in range(MAX_RETRIES):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(API_TIMEOUT)
            try:
                df = fetch_minute_data(code, period=PERIOD, adjust="qfq")
                signal.alarm(0)
                if df.empty:
                    ok = True
                    break
                upsert_df(df, "stock_minute")
                n = len(df)
                total_bars += n
                success += 1
                ok = True
                print(f"[{i+1}/{len(codes)}] {code} OK ({n} bars)", flush=True)
                break
            except (TimeoutError, Exception) as e:
                signal.alarm(0)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    fail += 1
                    print(f"[{i+1}/{len(codes)}] {code} FAIL: {e}", flush=True)
        time.sleep(SLEEP)

    print(f"\n完成: 成功 {success}, 失败 {fail}, 总 bar {total_bars}", flush=True)


if __name__ == "__main__":
    main()
