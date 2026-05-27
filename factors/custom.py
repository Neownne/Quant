"""自定义 A 股因子。

每函数签名: (df: pd.DataFrame) -> pd.Series
df 可能包含额外列: turnover, log_mcap, pb, shareholder_count 等。
"""

import numpy as np
import pandas as pd
from factors._scaling import w


def log_mcap(df: pd.DataFrame) -> pd.Series:
    """市值因子: 对数流通市值取负值（小市值溢价在A股显著）。"""
    if "log_mcap" in df.columns:
        return -df["log_mcap"]
    return pd.Series(np.nan, index=df.index)


def turnover_mom(df: pd.DataFrame) -> pd.Series:
    """换手率动量: 5日换手率变化 / 20日换手率标准差。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    chg = df["turnover"].diff(w(5))
    std = df["turnover"].rolling(w(20)).std()
    return chg / std.replace(0, np.nan)


def pb_pct(df: pd.DataFrame) -> pd.Series:
    """PB 历史分位（负值 = 低估值）: PB在500日中的分位数取负。"""
    if "pb" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    rank = df["pb"].rolling(w(500), min_periods=w(60)).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x) if len(x) > 0 else np.nan,
        raw=False,
    )
    return -rank


def shareholder_change(df: pd.DataFrame) -> pd.Series:
    """散户参与度变化: 股东户数季度环比变化率（增多利空，取负）。"""
    if "shareholder_count" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    sc = df["shareholder_count"].ffill()
    return -sc.pct_change(w(60))  # ~一个季度


def vol_conv(df: pd.DataFrame) -> pd.Series:
    """量能聚拢: -|vol/MA(vol,5) - vol/MA(vol,20)|（量缩等变盘）。"""
    v = df["volume"].replace(0, np.nan)
    r5 = v / v.rolling(w(5)).mean()
    r20 = v / v.rolling(w(20)).mean()
    return -(r5 - r20).abs()


def intra_vol(df: pd.DataFrame) -> pd.Series:
    """日内波动: (High - Low) / Open 的 5日 EMA。"""
    iv = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    return iv.ewm(span=w(5), adjust=False).mean()


def gap_ratio(df: pd.DataFrame) -> pd.Series:
    """跳空缺口: (Open - Close_lag1) / Close_lag1。"""
    prev_c = df["close"].shift(1).replace(0, np.nan)
    return (df["open"] - prev_c) / prev_c


CUSTOM_FACTORS: dict = {
    "log_mcap": log_mcap,
    "turnover_mom": turnover_mom,
    "pb_pct": pb_pct,
    "sh_change": shareholder_change,
    "vol_conv": vol_conv,
    "intra_vol": intra_vol,
    "gap_ratio": gap_ratio,
}
