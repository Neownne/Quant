"""
数据同步主入口。
用法：
    python -m data.sync                        # 全量同步（首次）
    python -m data.sync --mode stock           # 只同步股票列表
    python -m data.sync --mode stock-daily     # 只同步股票日线（增量）
    python -m data.sync --mode index           # 只同步指数
    python -m data.sync --mode etf             # 只同步 ETF 列表
    python -m data.sync --mode etf-daily       # 只同步 ETF 日线
    python -m data.sync --mode fund            # 只同步基金列表
    python -m data.sync --mode fund-nav        # 只同步基金净值
    python -m data.sync --start 20240101       # 指定起始日期
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from datetime import date, timedelta

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tqdm import tqdm

from config.settings import DataConfig
from data.db import init_db, upsert_df, get_engine, get_existing_dates
from data.fetcher import (
    enrich_stock_basic,
    fetch_stock_daily,
    fetch_index_daily,
    fetch_stock_lg_indicator,
    fetch_shareholder_count,
    fetch_financial_data,
    fetch_industry_classification,
    fetch_etf_list,
    fetch_etf_daily,
    fetch_fund_list,
    fetch_fund_nav,
)


# ---------- 工具 ----------

def _latest_trading_day() -> date:
    """最近的交易日（简单版：跳过周末，不考虑节假日）。"""
    today = date.today()
    if today.weekday() == 5:       # 周六 → 周五
        return today - timedelta(days=1)
    if today.weekday() == 6:       # 周日 → 周五
        return today - timedelta(days=2)
    return today


# ============================================================
#  股票
# ============================================================

def sync_stock_basic(engine: Engine) -> None:
    logger.info("=" * 50)
    logger.info("开始同步股票基本信息 ...")
    df = enrich_stock_basic()
    n = upsert_df(df, "stock_basic", engine)
    logger.success(f"stock_basic 同步完成，影响 {n} 行")


def sync_stock_daily(engine: Engine, start_date: str, workers: int = 4) -> None:
    logger.info("=" * 50)
    logger.info(f"开始同步股票日线，起始日期: {start_date} ...")

    codes = pd.read_sql("SELECT code FROM stock_basic", engine)["code"].tolist()
    cutoff = _latest_trading_day().strftime("%Y%m%d")

    # 过滤：跳过已覆盖到最近交易日的股票
    to_fetch: list[tuple[str, str, set]] = []
    for code in codes:
        existing = get_existing_dates("stock_daily", code, engine)
        latest = max(existing).strftime("%Y%m%d") if existing else start_date
        if latest < cutoff:
            to_fetch.append((code, latest, existing))

    logger.info(f"待同步: {len(to_fetch)}/{len(codes)} 只股票")

    if not to_fetch:
        logger.success("所有股票已是最新")
        return

    pbar = tqdm(total=len(to_fetch), desc="股票日线", unit="只")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_stock_daily, code, latest): (code, existing)
            for code, latest, existing in to_fetch
        }

        for future in as_completed(futures):
            code, existing = futures[future]
            try:
                df = future.result(timeout=60)
            except FutureTimeoutError:
                logger.error(f"{code} 请求超时，跳过")
                pbar.update(1)
                continue
            except Exception as e:
                logger.error(f"{code} 失败，跳过: {e}")
                pbar.update(1)
                continue

            if not df.empty:
                new_rows = df[~df["trade_date"].isin(existing)]
                if len(new_rows) > 0:
                    upsert_df(new_rows, "stock_daily", engine)
                    pbar.set_postfix_str(f"{code} +{len(new_rows)}条")
                else:
                    pbar.set_postfix_str(f"{code} 无新数据")
            pbar.update(1)

    logger.success("股票日线同步完成")


# ============================================================
#  指数
# ============================================================

def sync_index_daily(engine: Engine, start_date: str) -> None:
    logger.info("=" * 50)
    logger.info("开始同步指数日线 ...")

    for code, name in DataConfig.INDEX_CODES.items():
        df = fetch_index_daily(code, start_date=start_date)
        if not df.empty:
            n = upsert_df(df, "index_daily", engine)
            logger.info(f"  {name}({code}) 同步 {n} 条")
        time.sleep(DataConfig.REQUEST_INTERVAL)

    logger.success("指数日线同步完成")


def sync_daily_extra(engine: Engine, start_date: str, workers: int = 4) -> None:
    """同步估值指标（市值/PE/PB/股本）到 stock_daily_extra。"""
    logger.info("=" * 50)
    logger.info(f"开始同步估值指标，起始日期: {start_date} ...")

    codes = pd.read_sql("SELECT code FROM stock_basic", engine)["code"].tolist()
    cutoff = _latest_trading_day().strftime("%Y%m%d")

    to_fetch = []
    for code in codes:
        existing = get_existing_dates("stock_daily_extra", code, engine)
        latest = max(existing).strftime("%Y%m%d") if existing else start_date
        if latest < cutoff:
            to_fetch.append((code, latest))

    logger.info(f"待同步: {len(to_fetch)}/{len(codes)} 只股票")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_do_fetch_daily_extra, t) for t in to_fetch]
        for f in tqdm(as_completed(futures), total=len(to_fetch), desc="估值指标"):
            pass

    logger.success("估值指标同步完成")


def _do_fetch_daily_extra(args: tuple[str, str]) -> tuple[str, int]:
    """模块级 worker，供 ProcessPoolExecutor 使用。"""
    code, latest = args
    try:
        df = fetch_stock_lg_indicator(code)
        if df.empty:
            return code, 0
        df = df[df["trade_date"] >= pd.to_datetime(latest).date()]
        return code, upsert_df(df, "stock_daily_extra", engine=None)
    except Exception as e:
        logger.warning(f"{code} 估值指标同步失败: {e}")
        return code, 0


def sync_shareholder(engine: Engine, workers: int = 2) -> None:
    """同步股东户数数据到 stock_shareholder。数据量小，低并发。"""
    logger.info("=" * 50)
    logger.info("开始同步股东户数 ...")

    codes = pd.read_sql("SELECT code FROM stock_basic", engine)["code"].tolist()

    # 股东户数是季度数据，跳过已有记录的股票
    existing_codes = set()
    with engine.connect() as conn:
        r = conn.execute(text("SELECT DISTINCT code FROM stock_shareholder")).fetchall()
        existing_codes = {x[0] for x in r}
    to_fetch = [c for c in codes if c not in existing_codes]
    logger.info(f"待同步: {len(to_fetch)}/{len(codes)} 只股票")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_do_fetch_shareholder, c) for c in to_fetch]
        for f in tqdm(as_completed(futures), total=len(to_fetch), desc="股东户数"):
            pass

    logger.success("股东户数同步完成")


def _do_fetch_shareholder(code: str) -> tuple[str, int]:
    """模块级 worker，供 ProcessPoolExecutor 使用。"""
    try:
        df = fetch_shareholder_count(code)
        if df.empty:
            return code, 0
        return code, upsert_df(df, "stock_shareholder", engine=None)
    except Exception as e:
        logger.warning(f"{code} 股东户数同步失败: {e}")
        return code, 0


# ============================================================
#  财务数据
# ============================================================

def _do_fetch_financial(code: str) -> tuple[str, int]:
    """模块级 worker，供 ProcessPoolExecutor 使用。返回 (code, rows_written)。"""
    try:
        df = fetch_financial_data(code)
        if df.empty:
            return code, 0
        return code, upsert_df(df, "stock_financial", engine=None)
    except Exception as e:
        logger.warning(f"{code} 财务数据同步失败: {e}")
        return code, 0


def sync_financial(engine: Engine, workers: int = 4) -> None:
    """同步财务数据到 stock_financial。跳过已有记录的股票。"""
    logger.info("=" * 50)
    logger.info("开始同步财务数据 ...")

    codes = pd.read_sql("SELECT code FROM stock_basic", engine)["code"].tolist()

    # 跳过已有记录
    existing_codes = set()
    with engine.connect() as conn:
        r = conn.execute(text("SELECT DISTINCT code FROM stock_financial")).fetchall()
        existing_codes = {x[0] for x in r}
    to_fetch = [c for c in codes if c not in existing_codes]
    logger.info(f"待同步: {len(to_fetch)}/{len(codes)} 只股票")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_do_fetch_financial, c) for c in to_fetch]
        for f in tqdm(as_completed(futures), total=len(to_fetch), desc="财务数据"):
            code, n = f.result()
            if n > 0:
                logger.debug(f"{code} +{n}条")

    logger.success("财务数据同步完成")


# ============================================================
#  行业分类
# ============================================================

def sync_industry(engine: Engine) -> None:
    """同步行业分类到 stock_industry。全量 upsert。"""
    logger.info("=" * 50)
    logger.info("开始同步行业分类 ...")

    df = fetch_industry_classification()
    if df.empty:
        logger.error("行业分类数据为空，跳过")
        return

    n = upsert_df(df, "stock_industry", engine)
    logger.success(f"行业分类同步完成，写入/更新 {n} 行")


# ============================================================
#  ETF
# ============================================================

def sync_etf_basic(engine: Engine) -> None:
    logger.info("=" * 50)
    logger.info("开始同步 ETF 列表 ...")
    df = fetch_etf_list()
    n = upsert_df(df, "etf_basic", engine)
    logger.success(f"etf_basic 同步完成，影响 {n} 行")


def sync_etf_daily(engine: Engine, start_date: str, workers: int = 4) -> None:
    logger.info("=" * 50)
    logger.info(f"开始同步 ETF 日线，起始日期: {start_date} ...")

    codes = pd.read_sql("SELECT code, market FROM etf_basic", engine)
    cutoff = _latest_trading_day().strftime("%Y%m%d")

    # 过滤：跳过已覆盖到最近交易日的 ETF
    to_fetch: list[tuple[str, str, str, set]] = []
    for _, row in codes.iterrows():
        code, market = row["code"], row["market"]
        existing = get_existing_dates("etf_daily", code, engine)
        latest = max(existing).strftime("%Y%m%d") if existing else start_date
        if latest < cutoff:
            to_fetch.append((code, market, latest, existing))

    logger.info(f"待同步: {len(to_fetch)}/{len(codes)} 只 ETF")

    if not to_fetch:
        logger.success("所有 ETF 已是最新")
        return

    pbar = tqdm(total=len(to_fetch), desc="ETF 日线", unit="只")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_etf_daily, code, market, latest): (code, existing)
            for code, market, latest, existing in to_fetch
        }

        for future in as_completed(futures):
            code, existing = futures[future]
            try:
                df = future.result(timeout=60)
            except FutureTimeoutError:
                logger.error(f"ETF {code} 请求超时，跳过")
                pbar.update(1)
                continue
            except Exception as e:
                logger.error(f"ETF {code} 失败，跳过: {e}")
                pbar.update(1)
                continue

            if not df.empty:
                new_rows = df[~df["trade_date"].isin(existing)]
                if len(new_rows) > 0:
                    upsert_df(new_rows, "etf_daily", engine)
                    pbar.set_postfix_str(f"{code} +{len(new_rows)}条")
                else:
                    pbar.set_postfix_str(f"{code} 无新数据")
            pbar.update(1)

    logger.success("ETF 日线同步完成")


# ============================================================
#  开放式基金
# ============================================================

def sync_fund_basic(engine: Engine) -> None:
    logger.info("=" * 50)
    logger.info("开始同步基金列表 ...")
    df = fetch_fund_list()
    n = upsert_df(df, "fund_basic", engine)
    logger.success(f"fund_basic 同步完成，影响 {n} 行")


def sync_fund_nav(engine: Engine, start_date: str, workers: int = 4) -> None:
    logger.info("=" * 50)
    logger.info(f"开始同步基金净值，起始日期: {start_date} ...")

    codes = pd.read_sql("SELECT code FROM fund_basic", engine)["code"].tolist()

    pbar = tqdm(total=len(codes), desc="基金净值", unit="只")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_fund_nav, code): code
            for code in codes
        }

        for future in as_completed(futures):
            code = futures[future]
            try:
                df = future.result(timeout=60)
            except FutureTimeoutError:
                logger.error(f"基金 {code} 请求超时，跳过")
                pbar.update(1)
                continue
            except Exception as e:
                logger.error(f"基金 {code} 失败，跳过: {e}")
                pbar.update(1)
                continue

            if df.empty:
                pbar.update(1)
                continue

            df = df[df["nav_date"] >= pd.to_datetime(start_date).date()]
            if not df.empty:
                n = upsert_df(df, "fund_nav", engine)
                pbar.set_postfix_str(f"{code} +{n}条")
            pbar.update(1)

    logger.success("基金净值同步完成")


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="A股 / ETF / 基金数据同步工具")
    parser.add_argument(
        "--mode",
        choices=[
            "all", "stock", "stock-daily", "index",
            "etf", "etf-daily", "fund", "fund-nav",
            "daily-extra", "shareholder", "financial", "industry",
        ],
        default="all",
        help="同步模式",
    )
    parser.add_argument(
        "--start",
        default="20150101",
        help="起始日期 YYYYMMDD（默认 20150101，10年历史）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发线程数（默认 4）",
    )
    args = parser.parse_args()

    logger.info("初始化数据库表结构 ...")
    init_db()

    engine = get_engine()
    try:
        mode = args.mode

        if mode in ("all", "stock"):
            sync_stock_basic(engine)

        if mode in ("all", "stock-daily"):
            sync_stock_daily(engine, args.start, args.workers)

        if mode in ("all", "index"):
            sync_index_daily(engine, args.start)

        if mode in ("all", "etf"):
            sync_etf_basic(engine)

        if mode in ("all", "etf-daily"):
            sync_etf_daily(engine, args.start, args.workers)

        if mode in ("all", "fund"):
            sync_fund_basic(engine)

        if mode in ("all", "fund-nav"):
            sync_fund_nav(engine, args.start, args.workers)

        if mode in ("all", "daily-extra"):
            sync_daily_extra(engine, args.start, args.workers)

        if mode in ("all", "shareholder"):
            sync_shareholder(engine, args.workers)

        if mode in ("all", "financial"):
            sync_financial(engine, workers=args.workers)

        if mode in ("all", "industry"):
            sync_industry(engine)

    finally:
        engine.dispose()

    logger.success("全部同步任务完成")


if __name__ == "__main__":
    main()
