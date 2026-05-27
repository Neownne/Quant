"""Alpha191 隔夜效应类因子。"""
import numpy as np
import pandas as pd
from factors._scaling import w


def overnight_ret(df: pd.DataFrame) -> pd.Series:
    """隔夜收益：(O_t - C_{t-1}) / C_{t-1}。"""
    o, c = df["open"], df["close"]
    prev_c = c.shift(1).replace(0, np.nan)
    return (o - prev_c) / prev_c


def overnight_ret_std(df: pd.DataFrame) -> pd.Series:
    """隔夜波动：std(overnight_ret, 10)，取负。"""
    o, c = df["open"], df["close"]
    prev_c = c.shift(1).replace(0, np.nan)
    on_ret = (o - prev_c) / prev_c
    return -on_ret.rolling(w(10)).std()


def open_auction_jump(df: pd.DataFrame) -> pd.Series:
    """开盘跳空偏离：(O_t - MA(O,5)) / MA(O,5)。"""
    o = df["open"]
    ma5 = o.rolling(w(5)).mean().replace(0, np.nan)
    return (o - ma5) / ma5


def gap_ma_dev(df: pd.DataFrame) -> pd.Series:
    """缺口偏离：gap_ratio - MA(gap_ratio, 20)。"""
    o, c = df["open"], df["close"]
    prev_c = c.shift(1).replace(0, np.nan)
    gap = (o - prev_c) / prev_c
    return gap - gap.rolling(w(20)).mean()


ALPHA191_GAP: dict = {
    "overnight_ret": overnight_ret,
    "overnight_ret_std": overnight_ret_std,
    "open_auction_jump": open_auction_jump,
    "gap_ma_dev": gap_ma_dev,
}
