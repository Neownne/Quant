"""Alpha191 日内形态类因子。"""
import numpy as np
import pandas as pd


def upper_shadow(df: pd.DataFrame) -> pd.Series:
    """上影线比例：(H - max(O,C)) / (H-L)，上影长=卖压。"""
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    max_oc = pd.concat([o, c], axis=1).max(axis=1)
    hl_range = (h - l).replace(0, np.nan)
    return (h - max_oc) / hl_range


def lower_shadow(df: pd.DataFrame) -> pd.Series:
    """下影线比例：(min(O,C) - L) / (H-L)，下影长=支撑。"""
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    min_oc = pd.concat([o, c], axis=1).min(axis=1)
    hl_range = (h - l).replace(0, np.nan)
    return (min_oc - l) / hl_range


def body_ratio(df: pd.DataFrame) -> pd.Series:
    """实体比例：|C-O| / (H-L)，实体大=趋势强。"""
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    hl_range = (h - l).replace(0, np.nan)
    return (c - o).abs() / hl_range


def intra_day_rev(df: pd.DataFrame) -> pd.Series:
    """盘中反转度：(C-O) / (H-L)，正值=低开高走。"""
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    hl_range = (h - l).replace(0, np.nan)
    return (c - o) / hl_range


ALPHA191_INTRADAY: dict = {
    "upper_shadow": upper_shadow,
    "lower_shadow": lower_shadow,
    "body_ratio": body_ratio,
    "intra_day_rev": intra_day_rev,
}
