#!/usr/bin/env python3
"""同步 60 分钟 K 线数据到 stock_minute 表。

排除: ST / 上市 <180 天 / 北交所(4开头) / 科创板(8开头，无 Sina 分钟数据)。
速率: 每只间隔 0.3s，单只超时 30s，失败重试 3 次。
自动分批: --batch-size 70 --cooldown 2100 (35 分钟) 适配新浪 ~75 次封 IP 限制。

用法:
    python3 scripts/sync_minute_data.py --limit 10          # 只同步 10 只测试
    python3 scripts/sync_minute_data.py --skip 432          # 跳过前 432 只续传
    python3 scripts/sync_minute_data.py --batch-size 70     # 每 70 只自动冷却
    python3 scripts/sync_minute_data.py                     # 全量同步（无冷却）
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
SLEEP = 0.35  # 每秒 ~3 只，给 API 留余量
MAX_RETRIES = 3
API_TIMEOUT = 30


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("API 调用超时")


def get_eligible_codes(limit: int | None = None, skip: int = 0,
                       missing_only: bool = True) -> list[str]:
    """获取待同步股票列表。missing_only=True 只返回还没有分钟数据的。"""
    engine = get_engine()
    base_sql = """
        SELECT DISTINCT b.code FROM stock_basic b
        WHERE b.is_st = FALSE
          AND b.list_date <= CURRENT_DATE - INTERVAL '180 days'
          AND b.code NOT LIKE '4%'
          AND b.code NOT LIKE '8%'
          AND EXISTS (SELECT 1 FROM stock_daily d WHERE d.code = b.code)
    """
    if missing_only:
        base_sql += " AND b.code NOT IN (SELECT DISTINCT code FROM stock_minute WHERE period='60')"

    base_sql += " ORDER BY b.code"

    if limit:
        base_sql += f" LIMIT {int(limit)}"
    if skip > 0:
        base_sql += f" OFFSET {int(skip)}"

    codes = pd.read_sql(text(base_sql), engine)["code"].tolist()
    engine.dispose()
    return codes


def main():
    parser = argparse.ArgumentParser(description="同步 60 分钟 K 线")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 只股票（在 missing-only 过滤后）")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="每 N 只后冷却（0=不冷却）")
    parser.add_argument("--cooldown", type=int, default=2100,
                        help="冷却秒数（默认 2100=35 分钟）")
    parser.add_argument("--all", action="store_true",
                        help="同步所有符合条件的股票（含已有数据的）")
    args = parser.parse_args()

    init_db()
    codes = get_eligible_codes(limit=args.limit, skip=args.skip,
                               missing_only=not args.all)
    total = len(codes)
    print(f"共 {total} 只股票待同步", flush=True)

    if args.batch_size > 0:
        print(f"分批模式: 每 {args.batch_size} 只冷却 {args.cooldown}s (~{args.cooldown/60:.0f}min)",
              flush=True)

    success = 0
    fail = 0
    total_bars = 0
    batch_count = 0

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
                print(f"[{i+1}/{total}] {code} OK ({n} bars)", flush=True)
                break
            except (TimeoutError, Exception) as e:
                signal.alarm(0)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    fail += 1
                    print(f"[{i+1}/{total}] {code} FAIL: {e}", flush=True)
        time.sleep(SLEEP)

        # 分批冷却
        if args.batch_size > 0 and (i + 1) % args.batch_size == 0 and i + 1 < total:
            batch_count += 1
            print(f"\n--- 已完成 {success} 只, 冷却 {args.cooldown}s "
                  f"({time.strftime('%H:%M:%S')}) ---\n", flush=True)
            time.sleep(args.cooldown)

    print(f"\n完成: 成功 {success}, 失败 {fail}, 总 bar {total_bars}", flush=True)


if __name__ == "__main__":
    main()
