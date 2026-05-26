"""
通过 AKShare 获取 A 股/ETF/基金行情数据。
所有函数返回 pandas DataFrame，字段名与数据库表结构对齐。
"""
import socket
import time
from datetime import date
from functools import wraps
from typing import Callable

import akshare as ak
import numpy as np
import pandas as pd
import requests
from loguru import logger

from config.settings import DataConfig

# 全局 socket 超时 —— 防止网络请求在 C 级别无限期 hang 住
socket.setdefaulttimeout(30)


# ---------- 重试工具 ----------

def retry_on_network_error(max_retries: int = 3, base_delay: float = 2.0):
    """
    装饰器：遇到网络错误时自动重试（指数退避）。
    不捕获业务异常（如代码不存在），只重试连接类错误。
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_msg = str(e).lower()
                    is_network = any(kw in err_msg for kw in (
                        "connection", "remote end closed", "timeout",
                        "remote disconnected", "protocolerror",
                    ))
                    if not is_network:
                        raise

                    if attempt == max_retries:
                        logger.error(f"重试 {max_retries} 次后仍失败: {e}")
                        raise

                    wait = base_delay ** attempt
                    logger.warning(
                        f"网络错误，{wait:.0f}s 后重试 ({attempt}/{max_retries}): {e}"
                    )
                    time.sleep(wait)
        return wrapper
    return decorator


# ---------- 股票基本信息 ----------

@retry_on_network_error()
def fetch_stock_list() -> pd.DataFrame:
    """
    获取沪深 A 股列表。
    数据源：深交所 + 上交所官网。
    """
    logger.info("正在获取深市 A 股列表 ...")
    sz = ak.stock_info_sz_name_code("A股列表")

    logger.info("正在获取沪市 A 股列表 ...")
    sh = ak.stock_info_sh_name_code("主板A股")
    sh_kcb = ak.stock_info_sh_name_code("科创板")
    sh = pd.concat([sh, sh_kcb], ignore_index=True)

    logger.info("正在获取北交所 A 股列表 ...")
    bj = ak.stock_info_bj_name_code()

    # --- 深市 ---
    sz_df = pd.DataFrame()
    sz_df["code"] = sz["A股代码"]
    sz_df["name"] = sz["A股简称"]
    sz_df["market"] = "SZ"
    sz_df["list_date"] = pd.to_datetime(sz["A股上市日期"], errors="coerce").dt.date
    sz_df["industry"] = sz.get("所属行业", "")
    sz_df["is_st"] = sz_df["name"].str.contains("ST|\\*ST", regex=True)

    # --- 沪市 ---
    sh_df = pd.DataFrame()
    sh_df["code"] = sh["证券代码"]
    sh_df["name"] = sh["证券简称"]
    sh_df["market"] = "SH"
    sh_df["list_date"] = pd.to_datetime(sh["上市日期"], errors="coerce").dt.date
    sh_df["industry"] = ""  # 上交所基础列表无行业信息
    sh_df["is_st"] = sh_df["name"].str.contains("ST|\\*ST", regex=True)

    # --- 北交所 ---
    bj_df = pd.DataFrame()
    bj_df["code"] = bj["证券代码"]
    bj_df["name"] = bj["证券简称"]
    bj_df["market"] = "BJ"
    bj_df["list_date"] = pd.to_datetime(bj["上市日期"], errors="coerce").dt.date
    bj_df["industry"] = ""
    bj_df["is_st"] = False

    result = pd.concat([sz_df, sh_df, bj_df], ignore_index=True).drop_duplicates(subset=["code"])
    logger.success(
        f"获取到 {len(result)} 只股票（深市 {len(sz_df)} + 沪市 {len(sh_df)} + 北交所 {len(bj_df)}）"
    )
    return result


def enrich_stock_basic() -> pd.DataFrame:
    """
    获取股票基本信息（直接使用交易所接口，废弃东方财富）。
    """
    stocks = fetch_stock_list()
    return stocks


# ---------- 日线行情 ----------

def _to_tencent_symbol(code: str) -> str:
    """股票代码转腾讯格式：6xxxxx → sh6xxxxx，0/3xxxxx → sz0/3xxxxx"""
    if code.startswith(("0", "3")):
        return f"sz{code}"
    return f"sh{code}"


@retry_on_network_error()
def fetch_stock_daily(
    symbol: str,
    start_date: str = "20200101",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    下载单只股票的日线行情（后复权）。
    数据源：腾讯财经。

    参数
    ----
    symbol : 股票代码，如 "000001"（不带 SH/SZ 前缀）
    start_date : 起始日 YYYYMMDD
    end_date : 截止日 YYYYMMDD，None 则为今天

    返回
    ----
    DataFrame，字段对齐 stock_daily 表：
        code, trade_date, open, high, low, close, volume, amount, turnover
    """
    end = end_date or date.today().strftime("%Y%m%d")
    tx_symbol = _to_tencent_symbol(symbol)

    raw = ak.stock_zh_a_daily(
        symbol=tx_symbol,
        start_date=start_date,
        end_date=end,
        adjust="hfq",
    )

    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index(drop=True)
    df = pd.DataFrame({
        "code": symbol,
        "trade_date": pd.to_datetime(raw["date"].values, errors="coerce").date,
        "open": pd.to_numeric(raw["open"].values, errors="coerce"),
        "high": pd.to_numeric(raw["high"].values, errors="coerce"),
        "low": pd.to_numeric(raw["low"].values, errors="coerce"),
        "close": pd.to_numeric(raw["close"].values, errors="coerce"),
        "volume": pd.to_numeric(raw["volume"].values, errors="coerce"),
        "amount": pd.to_numeric(raw["amount"].values, errors="coerce"),
        "turnover": pd.to_numeric(raw["turnover"].values, errors="coerce"),
    })
    df = df.dropna(subset=["open", "close"])
    return df


# ---------- 指数日线 ----------

@retry_on_network_error()
def fetch_index_daily(
    symbol: str,
    start_date: str = "20200101",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    下载指数日线（上证/深证/创业板等）。
    数据源：腾讯财经。

    参数
    ----
    symbol : 指数代码
        "000001" = 上证指数, "399001" = 深证成指,
        "399006" = 创业板指, "000688" = 科创50
    """
    end = end_date or date.today().strftime("%Y%m%d")
    tx_symbol = f"sh{symbol}" if symbol.startswith("0") else f"sz{symbol}"

    raw = ak.stock_zh_index_daily_tx(symbol=tx_symbol, start_date=start_date, end_date=end)

    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index(drop=True)
    n = len(raw)
    df = pd.DataFrame({
        "code": [symbol] * n,
        "trade_date": pd.to_datetime(raw["date"].values, errors="coerce").date,
        "open": pd.to_numeric(raw["open"].values, errors="coerce"),
        "high": pd.to_numeric(raw["high"].values, errors="coerce"),
        "low": pd.to_numeric(raw["low"].values, errors="coerce"),
        "close": pd.to_numeric(raw["close"].values, errors="coerce"),
        "volume": 0,   # 腾讯指数源无成交量
        "amount": pd.to_numeric(raw["amount"].values, errors="coerce"),
    })
    df = df.dropna(subset=["open", "close"])
    df = df[(df["trade_date"] >= pd.to_datetime(start_date).date()) & (df["trade_date"] <= pd.to_datetime(end).date())]
    return df


# ============================================================
#  ETF 数据
# ============================================================

@retry_on_network_error()
def fetch_etf_list() -> pd.DataFrame:
    """
    获取全市场 ETF 列表。
    数据源：东方财富。
    返回 columns: code, name, category
    """
    logger.info("正在获取 ETF 列表 ...")
    raw = ak.fund_etf_category_sina("ETF基金")

    df = pd.DataFrame()
    df["code"] = raw["代码"].str.replace("sz", "").str.replace("sh", "")
    df["name"] = raw["名称"]
    df["market"] = raw["代码"].apply(lambda x: "SH" if x.startswith("sh") else "SZ")
    logger.success(f"获取到 {len(df)} 只 ETF")
    return df


@retry_on_network_error()
def fetch_etf_daily(
    symbol: str,
    market: str = "SZ",
    start_date: str = "20200101",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    下载单只 ETF 的日线行情。
    数据源：新浪财经。

    symbol : 纯数字代码，如 "159915"
    market : "SH" / "SZ"
    """
    end = end_date or date.today().strftime("%Y%m%d")
    prefix = "sh" if market == "SH" else "sz"
    tx_symbol = f"{prefix}{symbol}"

    raw = ak.fund_etf_hist_sina(symbol=tx_symbol)

    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index(drop=True)
    df = pd.DataFrame({
        "code": symbol,
        "trade_date": pd.to_datetime(raw["date"].values, errors="coerce").date,
        "open": pd.to_numeric(raw["open"].values, errors="coerce"),
        "high": pd.to_numeric(raw["high"].values, errors="coerce"),
        "low": pd.to_numeric(raw["low"].values, errors="coerce"),
        "close": pd.to_numeric(raw["close"].values, errors="coerce"),
        "volume": pd.to_numeric(raw["volume"].values, errors="coerce"),
        "amount": pd.to_numeric(raw["amount"].values, errors="coerce"),
    })
    df = df.dropna(subset=["open", "close"])
    df = df[(df["trade_date"] >= pd.to_datetime(start_date).date()) & (df["trade_date"] <= pd.to_datetime(end).date())]
    return df


# ============================================================
#  开放式基金数据（净值类，非 OHLCV）
# ============================================================

@retry_on_network_error()
def fetch_fund_list() -> pd.DataFrame:
    """
    获取开放式基金列表。
    数据源：天天基金。
    返回 columns: code, name, fund_type
    """
    logger.info("正在获取基金列表 ...")
    # 通过基金排名接口获取，取所有类型的前10000只
    all_funds = []
    for ftype in ["", "gp", "hh", "zq", "zs", "qdii"]:
        try:
            time.sleep(DataConfig.REQUEST_INTERVAL)
            raw = ak.fund_open_fund_rank_em(symbol=ftype if ftype else "全部")
            all_funds.append(raw[["基金代码", "基金简称"]])
        except Exception as e:
            logger.warning(f"基金类型 {ftype or '全部'} 获取失败: {e}")

    if not all_funds:
        return pd.DataFrame()

    raw = pd.concat(all_funds).drop_duplicates()

    df = pd.DataFrame()
    df["code"] = raw["基金代码"]
    df["name"] = raw["基金简称"]
    logger.success(f"获取到 {len(df)} 只基金")
    return df


@retry_on_network_error()
def fetch_fund_nav(code: str) -> pd.DataFrame:
    """
    获取单只基金的净值走势。

    返回 columns: code, nav_date, unit_nav, accumulated_nav, daily_return
    """
    raw = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")

    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index(drop=True)
    # AKShare 仅返回：净值日期、单位净值、日增长率 —— 无累计净值
    df = pd.DataFrame({
        "code": code,
        "nav_date": pd.to_datetime(raw["净值日期"].values, errors="coerce").date,
        "unit_nav": pd.to_numeric(raw["单位净值"].values, errors="coerce"),
        "accumulated_nav": np.nan,
        "daily_return": pd.to_numeric(raw.get("日增长率", pd.Series([0])).values, errors="coerce"),
    })

    df = df.dropna(subset=["unit_nav"])
    return df


# ============================================================
#  基本面/估值数据
# ============================================================

@retry_on_network_error()
def fetch_stock_lg_indicator(symbol: str) -> pd.DataFrame:
    """获取单只股票的估值指标（市值/PB）。
    数据源：百度财经（通过 AKShare）。
    返回 columns: code, trade_date, market_cap, float_market_cap, pe, pb, total_share, float_share"""
    frames = []
    for indicator, col_map in [
        ("总市值", {"value": "market_cap"}),
        ("市净率", {"value": "pb"}),
    ]:
        try:
            raw = ak.stock_zh_valuation_baidu(symbol=symbol, indicator=indicator)
            if raw is None or raw.empty:
                continue
            raw = raw.rename(columns=col_map)
            frames.append(raw[["date", *col_map.values()]])
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    # Merge all indicators on date
    from functools import reduce
    df = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), frames)
    df = df.rename(columns={"date": "trade_date"})
    df["code"] = symbol
    df["trade_date"] = pd.to_datetime(df["trade_date"].values, errors="coerce").date
    df["float_market_cap"] = np.nan
    df["pe"] = np.nan
    df["total_share"] = np.nan
    df["float_share"] = np.nan
    return df.dropna(subset=["trade_date"])


@retry_on_network_error()
def fetch_shareholder_count(symbol: str) -> pd.DataFrame:
    """获取单只股票的股东户数变化历史。
    数据源：东方财富。
    返回 columns: code, end_date, shareholder_count, avg_holding_value, avg_holding_amount, total_market_cap"""
    try:
        raw = ak.stock_zh_a_gdhs_detail_em(symbol=symbol)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index(drop=True)
    df = pd.DataFrame({
        "code": symbol,
        "end_date": pd.to_datetime(raw["股东户数统计截止日"].values, errors="coerce").date,
        "shareholder_count": pd.to_numeric(raw.get("股东户数-本次", 0), errors="coerce"),
        "avg_holding_value": pd.to_numeric(raw.get("户均持股市值", 0), errors="coerce"),
        "avg_holding_amount": pd.to_numeric(raw.get("户均持股数量", 0), errors="coerce"),
        "total_market_cap": pd.to_numeric(raw.get("总市值", 0), errors="coerce"),
    })
    return df.dropna(subset=["end_date"])


# ============================================================
#  财务基本面数据
# ============================================================

def _parse_financial_value(val) -> float | None:
    """Parse a Chinese financial value like '4302.00万' or '1.13亿' to float in yuan."""
    if pd.isna(val) or val in (False, True, ""):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(",", "").strip()
    if not s:
        return None
    multiplier = 1.0
    if "亿" in s:
        multiplier = 1e8
        s = s.replace("亿", "")
    elif "万" in s:
        multiplier = 1e4
        s = s.replace("万", "")
    s = s.replace("%", "")
    try:
        return float(s) * multiplier
    except ValueError:
        return None


@retry_on_network_error()
def fetch_financial_data(symbol: str) -> pd.DataFrame:
    """获取单只股票的财务基本面数据（同花顺接口）。
    数据源：同花顺（通过 AKShare）。
    返回 columns: code, report_date, revenue, net_profit, gross_margin,
                   net_margin, roe, total_assets, total_liability, bps, eps, cash_flow"""

    # 12-column schema for consistent output (even on empty/error returns)
    spec_cols = ["revenue", "net_profit", "gross_margin", "net_margin", "roe",
                 "total_assets", "total_liability", "bps", "eps", "cash_flow"]

    def _empty_result() -> pd.DataFrame:
        """Return an empty DataFrame with all required columns."""
        empty = pd.DataFrame(columns=["code", "report_date"] + spec_cols)
        return empty

    try:
        raw = ak.stock_financial_abstract_ths(symbol=symbol)
    except Exception as e:
        logger.warning(f"fetch_financial_data({symbol}) 失败: {e}")
        return _empty_result()

    if raw.empty:
        return _empty_result()

    raw = raw.reset_index(drop=True)

    col_map = {
        "报告期": "report_date",
        "营业总收入": "revenue",
        "净利润": "net_profit",
        "毛利率": "gross_margin",
        "销售净利率": "net_margin",
        "净资产收益率": "roe",
        "总资产": "total_assets",
        "总负债": "total_liability",
        "每股净资产": "bps",
        "基本每股收益": "eps",
        "每股经营现金流": "cash_flow",
    }

    # Only keep columns that exist in the raw DataFrame
    existing = {k: v for k, v in col_map.items() if k in raw.columns}
    df = raw[list(existing.keys())].rename(columns=existing)

    # Ensure all required columns exist (fill missing with NaN)
    for col in spec_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Add code column
    df["code"] = symbol

    # Convert report_date to date type
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date

    # Convert numeric columns using the Chinese-unit-aware parser
    numeric_cols = [v for v in existing.values() if v != "report_date"]
    for col in numeric_cols:
        df[col] = df[col].apply(_parse_financial_value).astype("float64")

    # Reorder: code + report_date + all spec columns
    return df[["code", "report_date"] + spec_cols]


def _pivot_ths_long(raw: pd.DataFrame, metric_map: dict[str, str]) -> pd.DataFrame:
    """将同花顺 _new_ths 长表转宽表，提取指定指标。

    参数
    ----
    raw : stock_financial_{debt,benefit,cash}_new_ths 的原始返回
    metric_map : {metric_name: output_col_name}

    返回
    ----
    DataFrame，列为 [report_date] + list(metric_map.values())，按 report_date 排序
    """
    if raw.empty:
        return pd.DataFrame()
    subset = raw[raw["metric_name"].isin(metric_map.keys())].copy()
    if subset.empty:
        return pd.DataFrame()
    # value 列是字符串，空串=NaN
    subset["value_num"] = pd.to_numeric(subset["value"].replace("", None), errors="coerce")
    pivot = subset.pivot_table(
        index="report_date", columns="metric_name", values="value_num", aggfunc="first"
    )
    pivot = pivot.rename(columns=metric_map)
    pivot = pivot.reset_index()
    pivot["report_date"] = pd.to_datetime(pivot["report_date"], errors="coerce").dt.date
    return pivot.sort_values("report_date")


def fetch_balance_sheet(symbol: str) -> pd.DataFrame:
    """获取资产负债表关键字段。

    数据源：同花顺 stock_financial_debt_new_ths。
    返回 columns: report_date, total_assets, total_liability, goodwill,
                   short_term_loans, cash_equivalents, holder_equity
    """
    try:
        raw = ak.stock_financial_debt_new_ths(symbol=symbol)
    except Exception as e:
        logger.warning(f"资产负债表获取失败 {symbol}: {e}")
        return pd.DataFrame()

    metric_map = {
        "assets_total": "total_assets",
        "total_debt": "total_liability",
        "goodwill": "goodwill",
        "short_term_loans": "short_term_loans",
        "cash": "cash_equivalents",
        "holder_equity_total": "holder_equity",
    }
    df = _pivot_ths_long(raw, metric_map)
    if not df.empty:
        df["code"] = symbol
        cols = ["code", "report_date"] + [v for v in metric_map.values() if v in df.columns]
        return df[cols]
    return df


def fetch_cash_flow_statement(symbol: str) -> pd.DataFrame:
    """获取现金流量表关键字段。

    数据源：同花顺 stock_financial_cash_new_ths。
    返回 columns: report_date, operating_cash_flow
    """
    try:
        raw = ak.stock_financial_cash_new_ths(symbol=symbol)
    except Exception as e:
        logger.warning(f"现金流量表获取失败 {symbol}: {e}")
        return pd.DataFrame()

    metric_map = {"act_cash_flow_net": "operating_cash_flow"}
    df = _pivot_ths_long(raw, metric_map)
    if not df.empty:
        df["code"] = symbol
        cols = ["code", "report_date"] + [v for v in metric_map.values() if v in df.columns]
        return df[cols]
    return df


def fetch_profit_statement(symbol: str) -> pd.DataFrame:
    """获取利润表补充字段。

    数据源：同花顺 stock_financial_benefit_new_ths。
    返回 columns: report_date, adjusted_profit, parent_net_profit
    """
    try:
        raw = ak.stock_financial_benefit_new_ths(symbol=symbol)
    except Exception as e:
        logger.warning(f"利润表获取失败 {symbol}: {e}")
        return pd.DataFrame()

    metric_map = {
        "index_deduct_holder_net_profit": "adjusted_profit",
        "parent_holder_net_profit": "parent_net_profit",
    }
    df = _pivot_ths_long(raw, metric_map)
    if not df.empty:
        df["code"] = symbol
        cols = ["code", "report_date"] + [v for v in metric_map.values() if v in df.columns]
        return df[cols]
    return df


def fetch_financial_supplement(symbol: str) -> pd.DataFrame:
    """合并资产负债表 + 现金流量表 + 利润表补充字段。

    返回宽表，按 report_date 对齐。
    """
    bs = fetch_balance_sheet(symbol)
    cf = fetch_cash_flow_statement(symbol)
    pl = fetch_profit_statement(symbol)

    frames = [df for df in [bs, cf, pl] if not df.empty]
    if not frames:
        return pd.DataFrame()

    from functools import reduce

    result = reduce(
        lambda left, right: pd.merge(left, right, on=["code", "report_date"], how="outer"),
        frames,
    )
    return result.sort_values("report_date")


@retry_on_network_error()
def fetch_pledge_data() -> pd.DataFrame:
    """获取全市场股权质押比例。

    数据源：东方财富 stock_gpzy_pledge_ratio_em。
    返回 columns: code, trade_date, pledge_ratio, pledge_shares, pledge_market_cap, pledge_count
    """
    try:
        raw = ak.stock_gpzy_pledge_ratio_em()
    except Exception as e:
        logger.warning(f"质押数据获取失败: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        "code": raw["股票代码"].astype(str).str.replace(" ", ""),
        "trade_date": pd.to_datetime(raw["交易日期"], errors="coerce").dt.date,
        "pledge_ratio": pd.to_numeric(raw["质押比例"], errors="coerce"),
        "pledge_shares": pd.to_numeric(raw["质押股数"], errors="coerce"),
        "pledge_market_cap": pd.to_numeric(raw["质押市值"], errors="coerce"),
        "pledge_count": pd.to_numeric(raw["质押笔数"], errors="coerce"),
    })
    return df.dropna(subset=["code", "trade_date"])
#  行业分类数据
# ============================================================

@retry_on_network_error()
def fetch_industry_classification() -> pd.DataFrame:
    """获取全市场股票行业分类。
    主数据源: stock_basic 表的 industry/market 字段。
    返回 columns: code, industry_sw1, industry_sw2, market
    """
    from data.db import get_engine

    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                "SELECT code, industry, market FROM stock_basic WHERE industry IS NOT NULL AND industry != ''",
                conn,
            )
    except Exception as e:
        logger.warning(f"从 stock_basic 获取行业分类失败: {e}")
        return pd.DataFrame(columns=["code", "industry_sw1", "industry_sw2", "market"])
    finally:
        engine.dispose()

    if df.empty:
        return pd.DataFrame(columns=["code", "industry_sw1", "industry_sw2", "market"])

    df = df.rename(columns={"industry": "industry_sw1"})
    df["industry_sw2"] = ""

    # 从代码前缀推断市场板块分类
    def _detect_market(code: str) -> str:
        if code.startswith("688"):
            return "科创板"
        elif code.startswith("300") or code.startswith("301"):
            return "创业板"
        elif code.startswith("8") or code.startswith("4"):
            return "北交所"
        elif code.startswith("6"):
            return "主板"
        elif code.startswith("0") or code.startswith("2"):
            return "主板"
        return "未知"

    df["market"] = df["code"].apply(_detect_market)

    return df[["code", "industry_sw1", "industry_sw2", "market"]]


# ============================================================
#  实时/日内数据（逐笔 & 分钟K线）
# ============================================================

@retry_on_network_error()
def fetch_tick_data(symbol: str, trade_date: date | None = None) -> pd.DataFrame:
    """
    获取单只股票当日逐笔成交数据。
    数据源：腾讯财经。

    参数
    ----
    symbol : 纯数字代码
    trade_date : 交易日，None 则用最近交易日。腾讯接口不返回日期，
                 只有 HH:MM:SS，需外部传入。

    返回
    ----
    DataFrame 字段：code, trade_time, price, price_change, volume, amount, direction
    """
    tx_symbol = _to_tencent_symbol(symbol)
    raw = ak.stock_zh_a_tick_tx_js(symbol=tx_symbol)

    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index(drop=True)
    td = trade_date or date.today()

    df = pd.DataFrame({
        "code": symbol,
        "trade_time": pd.to_datetime(
            td.strftime("%Y-%m-%d ") + raw["成交时间"].values, errors="coerce"
        ),
        "price": pd.to_numeric(raw["成交价格"].values, errors="coerce"),
        "price_change": pd.to_numeric(raw["价格变动"].values, errors="coerce"),
        "volume": pd.to_numeric(raw["成交量"].values, errors="coerce"),
        "amount": pd.to_numeric(raw["成交金额"].values, errors="coerce"),
        "direction": raw["性质"].values,
    })
    df = df.dropna(subset=["price", "trade_time"])
    return df


@retry_on_network_error()
def fetch_minute_data(
    symbol: str,
    period: str = "1",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    获取单只股票分钟 K 线。
    数据源：新浪财经 (money.finance.sina.com.cn)。

    参数
    ----
    symbol : 纯数字代码
    period : '1', '5', '15', '30', '60'
    adjust : 保留参数，sina 分钟线默认为前复权

    返回
    ----
    DataFrame 字段：code, trade_time, period, open, high, low, close, volume, amount
    """
    sina_symbol = _to_tencent_symbol(symbol)
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {
        "symbol": sina_symbol,
        "scale": period,
        "ma": "no",
        "datalen": "1970",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    if not data or not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "day" not in df.columns:
        return pd.DataFrame()

    n = len(df)
    result = pd.DataFrame({
        "code": [symbol] * n,
        "trade_time": pd.to_datetime(df["day"].values, errors="coerce"),
        "period": period,
        "open": pd.to_numeric(df["open"].values, errors="coerce"),
        "high": pd.to_numeric(df["high"].values, errors="coerce"),
        "low": pd.to_numeric(df["low"].values, errors="coerce"),
        "close": pd.to_numeric(df["close"].values, errors="coerce"),
        "volume": pd.to_numeric(df["volume"].values, errors="coerce"),
        "amount": np.nan,
    })
    result = result.dropna(subset=["open", "close"])
    return result
