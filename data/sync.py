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
from datetime import date, timedelta, datetime

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tqdm import tqdm

from config.settings import DataConfig
from data.db import init_db, upsert_df, get_engine, get_existing_dates
from data.quality import DataQualityChecker
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
    fetch_financial_supplement,
    fetch_pledge_data,
)


# ---------- 工具 ----------

_TRADING_CALENDAR: set[str] | None = None

def _get_trading_calendar() -> set[str]:
    """获取 A 股交易日历（缓存）。"""
    global _TRADING_CALENDAR
    if _TRADING_CALENDAR is None:
        try:
            import akshare as ak
            df = ak.tool_trade_date_hist_sina()
            _TRADING_CALENDAR = set(str(d) for d in df["trade_date"])
        except Exception:
            _TRADING_CALENDAR = set()
    return _TRADING_CALENDAR


def _latest_trading_day() -> date:
    """最近的 A 股交易日（使用真实交易日历）。

    盘前（< 09:00）或盘中（< 15:00）时，即使今天是交易日，
    日线数据也还没出，应返回上一个交易日。
    """
    now = datetime.now()
    today = date.today()
    calendar = _get_trading_calendar()

    # 如果今天是交易日但还在盘前/盘中（< 15:00），从昨天开始往回找
    start_offset = 0
    if str(today) in calendar and now.hour < 15:
        start_offset = 1

    for offset in range(start_offset, start_offset + 10):
        d = today - timedelta(days=offset)
        if str(d) in calendar:
            return d
    # fallback: 跳过周末
    if today.weekday() == 5:
        return today - timedelta(days=1)
    if today.weekday() == 6:
        return today - timedelta(days=2)
    return today


def check_data_quality(engine: Engine | None = None) -> dict:
    """检查所有表的数据质量。
    返回: {table_name: {latest_date, n_records, n_codes, status, stale_days}}
    """
    _engine = engine or get_engine()
    own_engine = engine is None

    table_checks = {
        "stock_basic": ("code", "list_date", None, 999),
        "stock_daily": ("code", "trade_date", "trade_date >= CURRENT_DATE - INTERVAL '30 days'", 3),
        "index_daily": ("code", "trade_date", "trade_date >= CURRENT_DATE - INTERVAL '30 days'", 3),
        "stock_daily_extra": ("code", "trade_date", "trade_date >= CURRENT_DATE - INTERVAL '30 days'", 3),
        "stock_shareholder": ("code", "end_date", None, 120),
        "stock_financial": ("code", "report_date", None, 120),
        "stock_industry": ("code", None, None, 999),
        "stock_pledge": ("code", "trade_date", None, 7),
        "paper_orders": ("id", "order_time", None, 999),
        "paper_daily_pnl": ("account_id", "trade_date", None, 999),
    }

    results = {}
    try:
        with _engine.connect() as conn:
            for table, (id_col, date_col, recent_filter, max_stale_days) in table_checks.items():
                entry = {"latest_date": None, "n_records": 0, "n_codes": 0, "status": "ok", "stale_days": 0}
                try:
                    exists = conn.execute(text(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=:t)"
                    ), {"t": table}).scalar()
                    if not exists:
                        entry["status"] = "missing"
                        results[table] = entry
                        continue

                    entry["n_records"] = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    entry["n_codes"] = conn.execute(text(f"SELECT COUNT(DISTINCT {id_col}) FROM {table}")).scalar()

                    if date_col:
                        latest = conn.execute(text(f"SELECT MAX({date_col}) FROM {table}")).scalar()
                        if latest:
                            entry["latest_date"] = str(latest)[:10]
                            gap = (date.today() - latest if hasattr(latest, 'date') else
                                   date.today() - date.fromisoformat(str(latest)[:10])).days if latest else 0
                            if isinstance(gap, int):
                                entry["stale_days"] = gap
                                if gap > max_stale_days:
                                    entry["status"] = "stale"
                            else:
                                entry["stale_days"] = 0

                    if recent_filter and entry["n_codes"] > 0:
                        recent_n = conn.execute(text(
                            f"SELECT COUNT(DISTINCT {id_col}) FROM {table} WHERE {recent_filter}"
                        )).scalar()
                        if recent_n < entry["n_codes"] * 0.5:
                            entry["status"] = "low_coverage" if entry["status"] == "ok" else entry["status"]

                except Exception as e:
                    entry["status"] = f"error: {e}"

                results[table] = entry
    finally:
        if own_engine:
            _engine.dispose()
    return results


def _run_quality_gate(trade_date: date):
    """同步后质量校验，不通过则记录告警"""
    checker = DataQualityChecker(expected_stock_count=5000)
    engine = get_engine()
    try:
        df = pd.read_sql(
            "SELECT code, close, volume FROM stock_daily WHERE trade_date = %(trade_date)s",
            engine,
            params={"trade_date": trade_date},
        )

        if df.empty:
            print(f"[QUALITY] 无数据，跳过校验")
            return False

        results = checker.run_all(df, trade_date)
        all_pass = True
        with engine.begin() as conn:
            for r in results:
                conn.execute(text("""
                    INSERT INTO data_quality_log (trade_date, check_name, expected_value, actual_value, passed, detail)
                    VALUES (:trade_date, :check_name, :expected, :actual, :passed, :detail)
                """), r)
                status = "PASS" if r["passed"] else "FAIL"
                if not r["passed"]:
                    all_pass = False
                print(f"[QUALITY] {r['check_name']}: {status} — {r['detail']}")

        if not all_pass:
            print("[QUALITY] 质量校验未通过，下游因子计算已阻断。请检查数据源后手动重跑。")
        return all_pass
    finally:
        engine.dispose()


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
    # 15:00 前不尝试同步今天数据（日线还没出）
    raw_cutoff = _latest_trading_day()
    now = datetime.now()
    if now.hour < 15 and raw_cutoff == date.today():
        calendar = _get_trading_calendar()
        d = date.today() - timedelta(days=1)
        for _ in range(10):
            if str(d) in calendar:
                raw_cutoff = d
                break
            d -= timedelta(days=1)
    cutoff = raw_cutoff.strftime("%Y%m%d")

    # ── 批量查已有日期（一发 SQL 替代 5000+ 次查询）──
    with engine.connect() as conn:
        existing_all = pd.read_sql(
            text("SELECT code, trade_date FROM stock_daily WHERE trade_date >= :s"),
            conn, params={"s": start_date}
        )
    existing_all['trade_date'] = existing_all['trade_date'].astype(str).str.replace('-', '')
    existing_map = existing_all.groupby('code')['trade_date'].apply(set).to_dict()

    # 过滤：跳过已覆盖到最近交易日的股票
    # 同时检测数据缺口 —— 如果 start_date 到最早 existing 之间有缺日，从 start_date 补起
    to_fetch: list[tuple[str, str, set]] = []
    for code in codes:
        existing = existing_map.get(code, set())
        if not existing:
            # 没有任何 >= start_date 的数据，需要全量拉取
            to_fetch.append((code, start_date, set()))
        else:
            latest = max(existing)  # 已是 YYYYMMDD 字符串
            earliest = min(existing)
            if earliest > start_date:
                # 前端有缺口（如 DB 有 06-23 但缺 06-19/06-22）
                # → 从 start_date 从头拉，upsert 不会重复已有数据
                to_fetch.append((code, start_date, existing))
            elif latest < cutoff:
                # 正常增量：尾部差几天
                to_fetch.append((code, latest, existing))

    # 汇总待补日期范围
    missing_dates = set()
    for _, latest, _ in to_fetch:
        missing_dates.add(str(latest))
    sorted_dates = sorted(missing_dates)
    date_range = f"{start_date} ~ {cutoff}" if len(sorted_dates) > 5 else ", ".join(sorted_dates[:8])
    logger.info(f"待同步: {len(to_fetch)}/{len(codes)} 只股票，日期: {date_range}")

    if not to_fetch:
        logger.success("所有股票已是最新")
        return

    pbar = tqdm(total=len(to_fetch), desc="股票日线", unit="只")
    done = 0
    errors = 0
    t_start = time.time()
    TOTAL_TIMEOUT = 3600  # 总超时 1 小时（逐只同步 5000+ 只至少需要 30-40 分钟）

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_stock_daily, code, latest): (code, existing)
            for code, latest, existing in to_fetch
        }

        for future in as_completed(futures):
            # 总超时保护
            if time.time() - t_start > TOTAL_TIMEOUT:
                logger.warning(f"  ⚠️ 总超时 ({TOTAL_TIMEOUT}s)，剩余任务取消")
                for f in futures:
                    f.cancel()
                break

            code, existing = futures[future]
            try:
                df = future.result(timeout=30)
            except FutureTimeoutError:
                logger.warning(f"  {code} 超时，跳过")
                errors += 1
                pbar.update(1)
                continue
            except Exception as e:
                logger.warning(f"  {code} 失败: {e}")
                errors += 1
                pbar.update(1)
                continue

            if not df.empty:
                new_rows = df[~df["trade_date"].isin(existing)]
                if len(new_rows) > 0:
                    upsert_df(new_rows, "stock_daily", engine)
                    synced_dates = sorted(new_rows["trade_date"].astype(str).unique())
                    date_str = ",".join(d[-5:] for d in synced_dates)  # MM-DD
                    pbar.set_postfix_str(f"{code} +{len(new_rows)}条 [{date_str}]")
            pbar.update(1)
            done += 1

    logger.success(f"股票日线同步完成: {done} 成功, {errors} 跳过")
    pbar.close()


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

    # 跳过已有基础财务数据的股票（以 net_profit 为准，避免补充数据造成的假阳性）
    existing_codes = set()
    with engine.connect() as conn:
        r = conn.execute(text(
            "SELECT DISTINCT code FROM stock_financial WHERE net_profit IS NOT NULL"
        )).fetchall()
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


def _do_fetch_financial_supplement(code: str) -> tuple[str, int]:
    """模块级 worker：获取资产负债表/现金流/利润表补充字段。"""
    try:
        df = fetch_financial_supplement(code)
        if df.empty:
            return code, 0
        return code, upsert_df(df, "stock_financial", engine=None)
    except Exception as e:
        logger.warning(f"{code} 财务补充数据同步失败: {e}")
        return code, 0


def sync_financial_supplement(engine: Engine, workers: int = 2) -> None:
    """同步资产负债表/现金流/扣非净利润等补充字段到 stock_financial。"""
    logger.info("=" * 50)
    logger.info("开始同步财务补充数据（资产负债表/现金流/扣非净利润）...")

    codes = pd.read_sql("SELECT code FROM stock_basic", engine)["code"].tolist()
    logger.info(f"共 {len(codes)} 只股票，使用 {workers} 并发")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_do_fetch_financial_supplement, c) for c in codes]
        for f in tqdm(as_completed(futures), total=len(codes), desc="财务补充"):
            pass

    logger.success("财务补充数据同步完成")


def sync_pledge(engine: Engine) -> None:
    """同步全市场股权质押数据到 stock_pledge。"""
    logger.info("=" * 50)
    logger.info("开始同步股权质押数据 ...")

    df = fetch_pledge_data()
    if df.empty:
        logger.warning("质押数据为空，跳过")
        return

    n = upsert_df(df, "stock_pledge", engine)
    logger.success(f"质押数据同步完成，写入/更新 {n} 行")


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
            "daily-extra", "shareholder", "financial", "financial-supplement",
            "pledge", "industry",
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

        if mode in ("all", "financial-supplement"):
            sync_financial_supplement(engine, workers=args.workers)

        if mode in ("all", "pledge"):
            sync_pledge(engine)

        if mode in ("all", "industry"):
            sync_industry(engine)

    finally:
        engine.dispose()

    logger.success("全部同步任务完成")

    # 同步完成后执行质量校验，不通过则阻断下游因子计算
    today = _latest_trading_day()
    passed = _run_quality_gate(today)
    if not passed:
        logger.warning("质量校验未通过！下游因子计算应暂停，待数据问题修复后重跑。")


if __name__ == "__main__":
    main()
