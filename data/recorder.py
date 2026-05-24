"""
实盘数据录制。
用法：
    python -m data.recorder                    # 录制分钟K线（默认模式）
    python -m data.recorder --mode tick       # 收盘后抓取逐笔数据
    python -m data.recorder --watchlist 000001,600519,300750  # 自选股
"""
import argparse
import time
from datetime import date, datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.settings import DataConfig
from data.db import init_db, upsert_df, get_engine, get_existing_dates
from data.fetcher import fetch_minute_data, fetch_tick_data

# 默认自选股 —— 大盘代表性标的
DEFAULT_WATCHLIST = [
    "000001",  # 平安银行
    "000002",  # 万科A
    "000858",  # 五粮液
    "002415",  # 海康威视
    "300750",  # 宁德时代
    "600036",  # 招商银行
    "600519",  # 贵州茅台
    "601318",  # 中国平安
]


def is_trading_time() -> bool:
    """判断当前是否在交易时段内（9:30-11:30, 13:00-15:00）。"""
    now = datetime.now()
    morning = now.replace(hour=9, minute=30, second=0)
    morning_end = now.replace(hour=11, minute=30, second=0)
    afternoon = now.replace(hour=13, minute=0, second=0)
    afternoon_end = now.replace(hour=15, minute=0, second=0)
    return (morning <= now <= morning_end) or (afternoon <= now <= afternoon_end)


def is_trading_day() -> bool:
    """判断今天是否是交易日（简化：周一到周五）。"""
    return date.today().weekday() < 5


def record_minute_bars(watchlist: list[str], period: str = "1") -> None:
    """
    录制分钟K线。交易时段内每 60 秒轮询一次。
    按 Ctrl+C 停止。
    """
    logger.info(f"开始录制 {period} 分钟K线，标的: {len(watchlist)} 只")
    engine = get_engine()

    try:
        while True:
            if not is_trading_day():
                logger.info("今天不是交易日，退出")
                break

            if not is_trading_time():
                logger.info("非交易时段，10 分钟后重试 ...")
                time.sleep(600)
                continue

            # 用 ProcessPoolExecutor 并发拉取
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(fetch_minute_data, code, period): code
                    for code in watchlist
                }
                for future in as_completed(futures):
                    code = futures[future]
                    try:
                        df = future.result(timeout=30)
                    except FutureTimeoutError:
                        logger.error(f"{code} 分钟数据超时")
                        continue
                    except Exception as e:
                        logger.error(f"{code} 分钟数据失败: {e}")
                        continue

                    if df.empty:
                        continue
                    # 只保留今日数据
                    today = pd.Timestamp.now().normalize()
                    df = df[df["trade_time"] >= today]
                    if not df.empty:
                        n = upsert_df(df, "stock_minute", engine)
                        logger.info(f"{code} 新增 {n} 条分钟K线")

            logger.info(f"等待 60 秒后下一轮 ...")
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("收到中断信号，停止录制")
    finally:
        engine.dispose()


def record_ticks(watchlist: list[str]) -> None:
    """
    收盘后抓取当日完整逐笔成交数据。
    """
    logger.info(f"开始抓取逐笔数据，标的: {len(watchlist)} 只")
    engine = get_engine()
    trade_date = date.today()

    pbar = tqdm(watchlist, desc="逐笔成交", unit="只")
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fetch_tick_data, code, trade_date): code
            for code in watchlist
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                df = future.result(timeout=120)
            except FutureTimeoutError:
                logger.error(f"{code} 逐笔数据超时")
                pbar.update(1)
                continue
            except Exception as e:
                logger.error(f"{code} 逐笔数据失败: {e}")
                pbar.update(1)
                continue

            if not df.empty:
                n = upsert_df(df, "stock_tick", engine)
                pbar.set_postfix_str(f"{code} +{n}笔")
            pbar.update(1)

    engine.dispose()
    logger.success("逐笔数据抓取完成")


def main():
    parser = argparse.ArgumentParser(description="实盘数据录制工具")
    parser.add_argument(
        "--mode",
        choices=["minute", "tick"],
        default="minute",
        help="录制模式：minute=分钟K线, tick=逐笔成交",
    )
    parser.add_argument(
        "--watchlist",
        default="",
        help="自选股代码，逗号分隔（默认使用内置列表）",
    )
    parser.add_argument(
        "--period",
        default="1",
        choices=["1", "5", "15", "30", "60"],
        help="分钟K线周期（默认 1 分钟）",
    )
    args = parser.parse_args()

    watchlist = (
        [c.strip() for c in args.watchlist.split(",") if c.strip()]
        if args.watchlist
        else DEFAULT_WATCHLIST
    )

    logger.info("初始化数据库表结构 ...")
    init_db()

    if args.mode == "minute":
        record_minute_bars(watchlist, args.period)
    elif args.mode == "tick":
        record_ticks(watchlist)


if __name__ == "__main__":
    main()
