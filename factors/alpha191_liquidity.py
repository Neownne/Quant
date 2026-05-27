"""Alpha191 流动性高阶类因子。"""
import numpy as np
import pandas as pd
from factors._scaling import w


def amihud_5(df: pd.DataFrame) -> pd.Series:
    """Amihud 非流动性(5日)：MA(|ret|/amount × 10^10, 5)。"""
    if "amount" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    ret = df["close"].pct_change()
    return (ret.abs() / df["amount"].replace(0, np.nan) * 1e10).rolling(w(5)).mean()


def dollar_volume(df: pd.DataFrame) -> pd.Series:
    """成交额对数：log(MA(amount, 20))，取负（小盘溢价）。"""
    if "amount" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return -np.log(df["amount"].rolling(w(20)).mean().replace(0, np.nan))


def turnover_breakout(df: pd.DataFrame) -> pd.Series:
    """换手率突破：(t - min_60) / (max_60 - min_60)。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    t = df["turnover"]
    t_min = t.rolling(w(60)).min()
    t_max = t.rolling(w(60)).max()
    denom = (t_max - t_min).replace(0, np.nan)
    return (t - t_min) / denom


def bid_ask_proxy(df: pd.DataFrame) -> pd.Series:
    """买卖价差代理：MA((H-L)/V, 20)。"""
    h, l, v = df["high"], df["low"], df["volume"]
    spread = (h - l) / v.replace(0, np.nan)
    return spread.rolling(w(20)).mean()


ALPHA191_LIQUIDITY: dict = {
    "amihud_5": amihud_5,
    "dollar_volume": dollar_volume,
    "turnover_breakout": turnover_breakout,
    "bid_ask_proxy": bid_ask_proxy,
}
