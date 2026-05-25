"""Alpha191 换手率类因子。"""
import numpy as np
import pandas as pd


def turnover_skew(df: pd.DataFrame) -> pd.Series:
    """换手率偏度：skew(turnover, 20)，右偏=资金进场。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["turnover"].rolling(20).skew()


def turnover_cv(df: pd.DataFrame) -> pd.Series:
    """换手率变异系数：std(turnover,20) / mean(turnover,20)，取负。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    t = df["turnover"]
    cv = t.rolling(20).std() / t.rolling(20).mean().replace(0, np.nan)
    return -cv


def turnover_ma_dev(df: pd.DataFrame) -> pd.Series:
    """换手率偏离度：turnover / MA(turnover,60) - 1。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    t = df["turnover"]
    return t / t.rolling(60).mean().replace(0, np.nan) - 1


def turnover_ret_corr(df: pd.DataFrame) -> pd.Series:
    """量价相关性：corr(turnover, ret, 20)。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    ret = df["close"].pct_change()
    return df["turnover"].rolling(20).corr(ret)


def free_turnover_ratio(df: pd.DataFrame) -> pd.Series:
    """流通换手比：turnover / float_share_ratio（无数据返回 NaN）。"""
    if "float_share_ratio" not in df.columns or "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["turnover"] / df["float_share_ratio"].replace(0, np.nan)


ALPHA191_TURNOVER: dict = {
    "turnover_skew": turnover_skew,
    "turnover_cv": turnover_cv,
    "turnover_ma_dev": turnover_ma_dev,
    "turnover_ret_corr": turnover_ret_corr,
    "free_turnover_ratio": free_turnover_ratio,
}
