"""全市场宽度因子。

从个股 OHLCV 日聚合计算每日全市场宽度指标，作为所有股票共享的宏观因子。
这些因子对所有股票同一天取值相同，帮助模型感知市场整体冷热。

因子列表：
  mkt_adv_dec_ratio   — 全市场涨跌比（上涨家数/下跌家数）
  mkt_limit_up_n      — 全市场涨停家数
  mkt_limit_down_n    — 全市场跌停家数
  mkt_up_vol_ratio    — 上涨股成交额占比
  mkt_ret_mean        — 全市场等权平均收益
  mkt_ret_std         — 全市场收益标准差
  mkt_turnover_mean   — 全市场平均换手率
  mkt_active_pct      — 活跃度（涨+跌>0的占比）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── 宽度的计算函数（给 build_market_breadth_extra 用）──

_LIMIT_MAP = {
    "688": 0.20,  # 科创板
    "8": 0.30,    # 北交所
    "4": 0.30,    # 北交所
}
_DEFAULT_LIMIT = 0.10


def _get_limit(code: str) -> float:
    for prefix, limit in _LIMIT_MAP.items():
        if code.startswith(prefix):
            return limit
    return _DEFAULT_LIMIT


def build_market_breadth_extra(ohlcv: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """从个股日线计算每日全市场宽度指标，返回 extra_data 格式的 DataFrame。

    参数
    ----
    ohlcv : 须含 code, trade_date, close, volume, amount, turnover
    codes : 全市场候选代码列表

    返回
    ----
    pd.DataFrame: [code, trade_date, mkt_adv_dec_ratio, mkt_limit_up_n, ...]
    """
    df = ohlcv.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 前日收盘（计算收益用）
    prev = df[["code", "trade_date", "close"]].copy()
    prev["trade_date"] = prev["trade_date"] + pd.Timedelta(days=1)
    prev = prev.rename(columns={"close": "prev_close"})

    df = df.merge(prev, on=["code", "trade_date"], how="left")
    df["ret"] = (df["close"] - df["prev_close"]) / df["prev_close"]

    # 涨停阈值
    df["limit"] = df["code"].apply(_get_limit)

    records = []
    for trade_date, day in df.groupby("trade_date"):
        n = len(day)
        ret = day["ret"].dropna()

        advancers = int((ret > 0).sum())
        decliners = int((ret < 0).sum())

        # 对齐索引（ret 可能少于 day 因为 NaN 被 drop 了）
        aligned_limit = day.loc[ret.index, "limit"]
        n_limit_up = int((ret >= aligned_limit * 0.99).sum())
        n_limit_down = int((ret <= -aligned_limit * 0.99).sum())

        up_amount = day.loc[ret[ret > 0].index, "amount"].sum()
        total_amount = day.loc[ret.index, "amount"].sum()

        row = {
            "trade_date": trade_date,
            "mkt_adv_dec_ratio": advancers / decliners if decliners > 0 else float(advancers),
            "mkt_limit_up_n": n_limit_up,
            "mkt_limit_down_n": n_limit_down,
            "mkt_up_vol_ratio": up_amount / total_amount if total_amount > 0 else 0.0,
            "mkt_ret_mean": float(ret.mean()) if len(ret) > 0 else 0.0,
            "mkt_ret_std": float(ret.std()) if len(ret) > 1 else 0.0,
            "mkt_turnover_mean": float(day["turnover"].mean()) if "turnover" in day.columns else 0.0,
            "mkt_active_pct": (advancers + decliners) / n if n > 0 else 0.0,
        }
        records.append(row)

    mkt = pd.DataFrame(records)

    # 展开到每只股票（extra_data 格式需要 code 列）
    all_dates = sorted(df["trade_date"].unique())
    code_df = pd.DataFrame({"code": list(codes)})
    code_df = code_df.merge(pd.DataFrame({"trade_date": all_dates}), how="cross")
    result = code_df.merge(mkt, on="trade_date", how="left")

    return result


# ── 因子函数（从 extra_data 读取）──

def mkt_adv_dec_ratio(df: pd.DataFrame) -> pd.Series:
    return df.get("mkt_adv_dec_ratio", pd.Series(np.nan, index=df.index))


def mkt_limit_up_n(df: pd.DataFrame) -> pd.Series:
    return df.get("mkt_limit_up_n", pd.Series(np.nan, index=df.index))


def mkt_limit_down_n(df: pd.DataFrame) -> pd.Series:
    return df.get("mkt_limit_down_n", pd.Series(np.nan, index=df.index))


def mkt_up_vol_ratio(df: pd.DataFrame) -> pd.Series:
    return df.get("mkt_up_vol_ratio", pd.Series(np.nan, index=df.index))


def mkt_ret_mean(df: pd.DataFrame) -> pd.Series:
    return df.get("mkt_ret_mean", pd.Series(np.nan, index=df.index))


def mkt_ret_std(df: pd.DataFrame) -> pd.Series:
    return df.get("mkt_ret_std", pd.Series(np.nan, index=df.index))


def mkt_turnover_mean(df: pd.DataFrame) -> pd.Series:
    return df.get("mkt_turnover_mean", pd.Series(np.nan, index=df.index))


def mkt_active_pct(df: pd.DataFrame) -> pd.Series:
    return df.get("mkt_active_pct", pd.Series(np.nan, index=df.index))


MARKET_BREADTH_FACTORS: dict = {
    "mkt_adv_dec_ratio": mkt_adv_dec_ratio,
    "mkt_limit_up_n": mkt_limit_up_n,
    "mkt_limit_down_n": mkt_limit_down_n,
    "mkt_up_vol_ratio": mkt_up_vol_ratio,
    "mkt_ret_mean": mkt_ret_mean,
    "mkt_ret_std": mkt_ret_std,
    "mkt_turnover_mean": mkt_turnover_mean,
    "mkt_active_pct": mkt_active_pct,
}
