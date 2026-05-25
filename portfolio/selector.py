"""选股：模型打分 top-N + ST/停牌/涨跌停/次新过滤。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date


def select_top_n(scores: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """从排序结果中选前 N 只。"""
    return scores.sort_values("rank").head(n).reset_index(drop=True)


def filter_suspended(
    stocks: pd.DataFrame,
    ohlcv_lookup: dict[str, pd.DataFrame],
    ref_date: pd.Timestamp,
    lookback_days: int = 5,
) -> pd.DataFrame:
    """剔除停牌股票：近 N 日成交量全为零或收盘价完全不变。"""
    if stocks.empty:
        return stocks
    result = stocks.copy()
    valid_mask = pd.Series(True, index=result.index)
    for i, row in result.iterrows():
        code = row["code"]
        hist = ohlcv_lookup.get(code)
        if hist is None or hist.empty:
            continue
        hist = hist[hist["trade_date"] <= ref_date].tail(lookback_days)
        if len(hist) < lookback_days:
            continue
        if (hist["volume"] == 0).all() or hist["close"].nunique() == 1:
            valid_mask.iloc[i] = False
    return result[valid_mask].reset_index(drop=True)


def filter_limit_up_down(
    stocks: pd.DataFrame,
    prev_close_map: dict[str, float],
    limit_pct: float = 0.10,
) -> pd.DataFrame:
    """剔除涨停（无法买入）和跌停（无法卖出）股票。"""
    if stocks.empty:
        return stocks
    result = stocks.copy()
    valid_mask = pd.Series(True, index=result.index)
    for i, row in result.iterrows():
        code = row["code"]
        prev = prev_close_map.get(code)
        if prev is None or prev <= 0:
            continue
        current = row.get("close", row.get("price"))
        if current is None or pd.isna(current) or current <= 0:
            continue
        limit_up = prev * (1 + limit_pct) * 0.999
        limit_down = prev * (1 - limit_pct) * 1.001
        if current >= limit_up or current <= limit_down:
            valid_mask.iloc[i] = False
    return result[valid_mask].reset_index(drop=True)


def filter_stocks(
    stocks: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
    exclude_st: bool = True,
    min_list_days: int = 60,
    ohlcv_lookup: dict[str, pd.DataFrame] | None = None,
    prev_close_map: dict[str, float] | None = None,
    filter_suspended_flag: bool = False,
    filter_limit_flag: bool = False,
) -> pd.DataFrame:
    """过滤不可交易的股票。

    参数
    ----
    stocks : 至少含 code, name 列
    ref_date : 参考日期（默认今天）
    exclude_st : 排除 ST
    min_list_days : 最小上市天数
    ohlcv_lookup : {code: OHLCV DataFrame}，停牌过滤需要
    prev_close_map : {code: 前日收盘价}，涨跌停过滤需要
    filter_suspended_flag : 启用停牌过滤
    filter_limit_flag : 启用涨跌停过滤
    """
    result = stocks.copy()
    ref = ref_date or pd.Timestamp(date.today())

    if exclude_st and "name" in result.columns:
        result = result[~result["name"].str.contains("ST", na=False)]

    if "list_date" in result.columns:
        result["days_listed"] = (ref - pd.to_datetime(result["list_date"])).dt.days
        result = result[result["days_listed"] >= min_list_days]
        result = result.drop(columns=["days_listed"])

    if filter_suspended_flag and ohlcv_lookup:
        result = filter_suspended(result, ohlcv_lookup, ref)

    if filter_limit_flag and prev_close_map:
        result = filter_limit_up_down(result, prev_close_map)

    return result.reset_index(drop=True)
