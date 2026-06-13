"""涨停策略核心选股逻辑。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from config.settings import TradingConfig


@dataclass
class LimitUpParams:
    """涨停策略 4 条件（去跌停）参数。"""
    mcap_min: float = 30.0
    mcap_max: float = 500.0
    price_min: float = 5.0
    price_max: float = 63.0
    lu_pct: float = TradingConfig.LIMIT_UP_PCT          # 0.09
    lu_lookback: int = 20
    lu_count: int = 1                                   # 通过标准为 > lu_count
    min_conditions: int = 4                             # 市值/股价/均线/涨停
    min_listed_days: int = 120


def run_screening(trade_date, daily_df, extra_df, code_set, params=None):
    """执行涨停策略 4 条件筛选。

    Parameters
    ----------
    trade_date: pd.Timestamp
    daily_df: pd.DataFrame
        含 code, trade_date, open, close, ret, ma5, ma10 的日线数据
    extra_df: pd.DataFrame
        含 code, trade_date, market_cap 的市值数据
    code_set: set[str]
        候选股票代码集合（已做 ST/上市天数过滤）
    params: LimitUpParams | None

    Returns
    -------
    list[tuple[str, int, float]]
        [(code, lu_count, close), ...]，按涨停次数降序、最近涨停距今升序排列
    """
    params = params or LimitUpParams()
    trade_date = pd.Timestamp(trade_date)

    daily_by_date = {d: g.set_index("code") for d, g in daily_df.groupby("trade_date")}
    if trade_date not in daily_by_date:
        return []

    td = daily_by_date[trade_date]
    lookback_start = trade_date - timedelta(days=params.lu_lookback + 5)
    lb = daily_df[(daily_df["trade_date"] >= lookback_start) & (daily_df["trade_date"] <= trade_date)]

    lu_counts = lb[lb["ret"] >= params.lu_pct].groupby("code").size()

    # 市值：取 trade_date 当日或最近一日的市值
    extra_by_date = {d: g.set_index("code") for d, g in extra_df.groupby("trade_date")} if not extra_df.empty else {}
    avail = sorted([d for d in extra_by_date if d <= trade_date], reverse=True)
    mcap_s = extra_by_date[avail[0]]["market_cap"] if avail else None

    passed = []
    for code in td.index:
        if code not in code_set:
            continue
        r = td.loc[code]
        close_p = r["close"]
        if pd.isna(close_p) or close_p <= 0:
            continue

        ma5, ma10 = r.get("ma5"), r.get("ma10")

        c1 = (mcap_s is not None and code in mcap_s.index and
              not pd.isna(mcap_s.loc[code]) and
              params.mcap_min <= mcap_s.loc[code] <= params.mcap_max)
        c2 = params.price_min <= close_p <= params.price_max
        c3 = (not pd.isna(ma5)) and (not pd.isna(ma10)) and (ma5 > ma10)
        lu_n = int(lu_counts.get(code, 0))
        c4 = lu_n > params.lu_count

        if sum([c1, c2, c3, c4]) >= params.min_conditions:
            passed.append((code, lu_n, float(close_p)))

    # 排序：涨停次数降序；同分按最近涨停距今（越近越好）升序
    lu_dates_map = {}
    for code, _, _ in passed:
        code_lu = lb[(lb["code"] == code) & (lb["ret"] >= params.lu_pct)]
        lu_dates_map[code] = (trade_date - code_lu["trade_date"].max()).days if not code_lu.empty else 99

    passed.sort(key=lambda x: (x[1], -lu_dates_map.get(x[0], 99)), reverse=True)
    return passed
