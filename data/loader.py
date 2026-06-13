"""数据加载工具：封装涨停策略常用的日线/市值/价格查询，全部使用参数化 SQL。"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from config.settings import TradingConfig


def _to_code_list(codes: Iterable[str]) -> list[str]:
    return [str(c) for c in codes]


def load_daily_data(engine, codes, start, end, cols=None):
    """加载日线数据并计算收益率和均线。

    Parameters
    ----------
    engine: SQLAlchemy engine
    codes: 股票代码集合/列表
    start, end: 日期字符串或 Timestamp
    cols: 额外列名列表，默认加载 open, close

    Returns
    -------
    pd.DataFrame
    """
    codes = _to_code_list(codes)
    if not codes:
        return pd.DataFrame(columns=["code", "trade_date", "open", "close", "ret", "ma5", "ma10"])

    cols = cols or ["open", "close"]
    base_cols = ["code", "trade_date"] + cols
    col_str = ", ".join(base_cols)

    # 分批查询避免超长 IN 列表
    chunks = [codes[i : i + 500] for i in range(0, len(codes), 500)]
    frames = []
    for chunk in chunks:
        df = pd.read_sql(
            text(f"SELECT {col_str} FROM stock_daily "
                 "WHERE code = ANY(:codes) AND trade_date BETWEEN :start AND :end "
                 "ORDER BY code, trade_date"),
            engine,
            params={"codes": chunk, "start": start, "end": end},
        )
        frames.append(df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=base_cols)
    if df.empty:
        return df

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["code", "trade_date"]).reset_index(drop=True)
    df["ret"] = df.groupby("code")["close"].pct_change()
    df["ma5"] = df.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    df["ma10"] = df.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=10).mean())
    return df


def load_mcap_data(engine, codes, start, end):
    """加载市值数据。"""
    codes = _to_code_list(codes)
    if not codes:
        return pd.DataFrame(columns=["code", "trade_date", "market_cap"])

    chunks = [codes[i : i + 500] for i in range(0, len(codes), 500)]
    frames = []
    for chunk in chunks:
        df = pd.read_sql(
            text("SELECT code, trade_date, market_cap FROM stock_daily_extra "
                 "WHERE code = ANY(:codes) AND trade_date BETWEEN :start AND :end"),
            engine,
            params={"codes": chunk, "start": start, "end": end},
        )
        frames.append(df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["code", "trade_date", "market_cap"])
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def get_next_open(engine, code, trade_date):
    """获取某股票在 trade_date 之后的下一个交易日的开盘价（T+1执行价）。"""
    row = pd.read_sql(
        text("SELECT open FROM stock_daily WHERE code = :code AND trade_date = :date"),
        engine,
        params={"code": code, "date": trade_date},
    )
    return float(row.iloc[0]["open"]) if not row.empty else None


def get_prev_close(engine, code, trade_date):
    """获取某股票在 trade_date 之前最近一个交易日的收盘价。"""
    row = pd.read_sql(
        text("SELECT close FROM stock_daily WHERE code = :code AND trade_date < :date "
             "ORDER BY trade_date DESC LIMIT 1"),
        engine,
        params={"code": code, "date": trade_date},
    )
    return float(row.iloc[0]["close"]) if not row.empty else None


def get_stock_basic(engine, trade_date, min_listed_days=120):
    """获取候选股票池（非ST、上市天数足够）。"""
    cutoff = pd.Timestamp(trade_date) - timedelta(days=min_listed_days)
    df = pd.read_sql(
        text("SELECT code, name FROM stock_basic WHERE is_st = FALSE AND list_date <= :cutoff"),
        engine,
        params={"cutoff": cutoff},
    )
    return df


def get_latest_trade_date(engine):
    """数据库中最新交易日。"""
    row = pd.read_sql(text("SELECT MAX(trade_date) AS d FROM stock_daily"), engine)
    return row.iloc[0]["d"]


def get_next_trading_date(engine, trade_date):
    """trade_date 之后的下一个实际交易日。"""
    row = pd.read_sql(
        text("SELECT MIN(trade_date) AS d FROM stock_daily WHERE trade_date > :date"),
        engine,
        params={"date": trade_date},
    )
    d = row.iloc[0]["d"]
    return pd.Timestamp(d) if d is not None and not pd.isna(d) else None


def get_today_close(engine, code, trade_date):
    """获取某股票在 trade_date 的收盘价。"""
    row = pd.read_sql(
        text("SELECT close FROM stock_daily WHERE code = :code AND trade_date = :date"),
        engine,
        params={"code": code, "date": trade_date},
    )
    return float(row.iloc[0]["close"]) if not row.empty else None


def get_position_values(engine, codes, trade_date):
    """批量获取多只股票在某交易日的收盘价。"""
    if not codes:
        return {}
    codes = _to_code_list(codes)
    df = pd.read_sql(
        text("SELECT code, close FROM stock_daily WHERE code = ANY(:codes) AND trade_date = :date"),
        engine,
        params={"codes": codes, "date": trade_date},
    )
    return dict(zip(df["code"], df["close"]))
