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
import pandas as pd
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
    df = pd.DataFrame({
        "code": code,
        "nav_date": pd.to_datetime(raw["净值日期"].values, errors="coerce").date,
        "unit_nav": pd.to_numeric(raw["单位净值"].values, errors="coerce"),
        "accumulated_nav": pd.to_numeric(raw["累计净值"].values, errors="coerce"),
        "daily_return": pd.to_numeric(raw.get("日增长率", pd.Series([0])).values, errors="coerce"),
    })

    df = df.dropna(subset=["unit_nav"])
    return df


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
    数据源：新浪财经。

    参数
    ----
    symbol : 纯数字代码
    period : '1', '5', '15', '30', '60'
    adjust : 'qfq' 前复权 / '' 不复权

    返回
    ----
    DataFrame 字段：code, trade_time, period, open, high, low, close, volume, amount
    """
    sina_symbol = _to_tencent_symbol(symbol)
    raw = ak.stock_zh_a_minute(symbol=sina_symbol, period=period, adjust=adjust)

    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index(drop=True)
    n = len(raw)
    df = pd.DataFrame({
        "code": [symbol] * n,
        "trade_time": pd.to_datetime(raw["day"].values, errors="coerce"),
        "period": period,
        "open": pd.to_numeric(raw["open"].values, errors="coerce"),
        "high": pd.to_numeric(raw["high"].values, errors="coerce"),
        "low": pd.to_numeric(raw["low"].values, errors="coerce"),
        "close": pd.to_numeric(raw["close"].values, errors="coerce"),
        "volume": pd.to_numeric(raw["volume"].values, errors="coerce"),
        "amount": pd.to_numeric(raw["amount"].values, errors="coerce"),
    })
    df = df.dropna(subset=["open", "close"])
    return df
