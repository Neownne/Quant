"""Alpha101 核心因子实现。

参考: Kakushadze & Tulchinsky (2015), "101 Formulaic Alphas"
每个因子函数签名: (df: pd.DataFrame) -> pd.Series
df 是单只股票按 trade_date 排序的 DataFrame，至少含 open/high/low/close/volume。
"""

import numpy as np
import pandas as pd
from factors._scaling import w


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _ts_sum(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).sum()


def _ts_std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).std()


def _ts_corr(a: pd.Series, b: pd.Series, period: int) -> pd.Series:
    return a.rolling(period).corr(b)


def _ts_roc(series: pd.Series, period: int) -> pd.Series:
    """变化率: (x_t - x_{t-period}) / |x_{t-period}|"""
    lag = series.shift(period)
    # avoid division by very small numbers
    safe_lag = lag.abs().replace(0, np.nan)
    return (series - lag) / safe_lag


# ============================================================
#  均值回归型因子 (Reversal)
# ============================================================

def rev_5(df: pd.DataFrame) -> pd.Series:
    """5日反转: -(C_t - C_{t-5}) / C_{t-5}"""
    return -_ts_roc(df["close"], w(5))


def rev_10(df: pd.DataFrame) -> pd.Series:
    """10日反转"""
    return -_ts_roc(df["close"], w(10))


def rev_20(df: pd.DataFrame) -> pd.Series:
    """20日反转"""
    return -_ts_roc(df["close"], w(20))


# ============================================================
#  动量型因子 (Momentum)
# ============================================================

def mom_20(df: pd.DataFrame) -> pd.Series:
    """20日动量"""
    return _ts_roc(df["close"], w(20))


def mom_60(df: pd.DataFrame) -> pd.Series:
    """60日动量（中期趋势）"""
    return _ts_roc(df["close"], w(60))


def ema_ratio_5_20(df: pd.DataFrame) -> pd.Series:
    """EMA(5) / EMA(20) - 1"""
    c = df["close"]
    return ema(c, w(5)) / ema(c, w(20)) - 1


# ============================================================
#  波动率因子 (Volatility)
# ============================================================

def vol_20(df: pd.DataFrame) -> pd.Series:
    """20日年化波动率"""
    return df["close"].pct_change().rolling(w(20)).std() * np.sqrt(252)


def atr_14(df: pd.DataFrame) -> pd.Series:
    """ATR(14) / Close"""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / w(14), adjust=False).mean()
    return atr / c


# ============================================================
#  量价背离因子 (Volume-Price Divergence)
# ============================================================

def vol_ratio_5_20(df: pd.DataFrame) -> pd.Series:
    """5日均量 / 20日均量"""
    v = df["volume"]
    return _ts_sum(v, w(5)) / _ts_sum(v, w(20))


def vpt(df: pd.DataFrame) -> pd.Series:
    """量价趋势: (C×V - MA(C×V, 20)) / MA(C×V, 20)"""
    cv = df["close"] * df["volume"]
    ma = cv.rolling(w(20)).mean()
    return (cv - ma) / ma


def vwap_ratio(df: pd.DataFrame) -> pd.Series:
    """VWAP偏离: Close / VWAP(20) - 1"""
    typ = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typ * df["volume"]).rolling(w(20)).sum() / df["volume"].rolling(w(20)).sum()
    return df["close"] / vwap - 1


# ============================================================
#  趋势强度因子
# ============================================================

def macd_dif(df: pd.DataFrame) -> pd.Series:
    """MACD DIF"""
    return ema(df["close"], w(12)) - ema(df["close"], w(26))


def macd_signal(df: pd.DataFrame) -> pd.Series:
    """MACD Signal"""
    return ema(macd_dif(df), w(9))


def macd_hist(df: pd.DataFrame) -> pd.Series:
    """MACD 柱 / Close"""
    dif = macd_dif(df)
    dea = macd_signal(df)
    return (dif - dea) / df["close"]


def rsi_14(df: pd.DataFrame) -> pd.Series:
    """RSI(14) — Wilder smoothing"""
    c = df["close"]
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / w(14), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / w(14), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rsi_7(df: pd.DataFrame) -> pd.Series:
    """RSI(7)"""
    c = df["close"]
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / w(7), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / w(7), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ============================================================
#  通道/布林带因子
# ============================================================

def bb_position(df: pd.DataFrame) -> pd.Series:
    """布林带位置: (C - 中轨) / (上轨 - 下轨)"""
    c = df["close"]
    mid = sma(c, w(20))
    std = c.rolling(w(20)).std()
    return (c - mid) / (4 * std + 1e-10)


def bb_width(df: pd.DataFrame) -> pd.Series:
    """布林带宽度 (标准化): (4*std) / SMA(20)"""
    c = df["close"]
    std = c.rolling(w(20)).std()
    return (4 * std) / sma(c, w(20))


# ============================================================
#  流动性因子
# ============================================================

def turnover_5(df: pd.DataFrame) -> pd.Series:
    """5日均换手率"""
    if "turnover" in df.columns:
        return df["turnover"].rolling(w(5)).mean()
    return pd.Series(np.nan, index=df.index)


def illiquidity(df: pd.DataFrame) -> pd.Series:
    """非流动性: |return| / (volume × close) 的20日均值"""
    r = df["close"].pct_change()
    illiq = r.abs() / (df["volume"] * df["close"] + 1e-10)
    return illiq.rolling(w(20)).mean()


def amount_ratio(df: pd.DataFrame) -> pd.Series:
    """5日均额 / 20日均额"""
    if "amount" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["amount"].rolling(w(5)).mean() / df["amount"].rolling(w(20)).mean()


# ============================================================
#  高阶因子
# ============================================================

def skewness_20(df: pd.DataFrame) -> pd.Series:
    """20日收益偏度"""
    return df["close"].pct_change().rolling(w(20)).skew()


def kurtosis_20(df: pd.DataFrame) -> pd.Series:
    """20日收益峰度"""
    return df["close"].pct_change().rolling(w(20)).kurt()


def high_low_ratio(df: pd.DataFrame) -> pd.Series:
    """(High-Low)/Close 20日均值"""
    return ((df["high"] - df["low"]) / df["close"]).rolling(w(20)).mean()


def max_dd_20(df: pd.DataFrame) -> pd.Series:
    """20日回撤比例: (C - High_20) / High_20"""
    c = df["close"]
    rolling_max = c.rolling(w(20), min_periods=1).max()
    return (c - rolling_max) / rolling_max


def corr_c_v(df: pd.DataFrame) -> pd.Series:
    """10日收盘价-成交量相关系数"""
    return _ts_corr(df["close"], df["volume"], w(10))


def co_ratio(df: pd.DataFrame) -> pd.Series:
    """(Close - Open) / (High - Low + 1e-10)"""
    return (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)


def up_day_ratio(df: pd.DataFrame) -> pd.Series:
    """20日中上涨天数比例"""
    return (df["close"].diff() > 0).rolling(w(20), min_periods=1).mean()


def price_position(df: pd.DataFrame) -> pd.Series:
    """价格在20日区间位置: (C - Low_20) / (High_20 - Low_20)"""
    c = df["close"]
    h = df["high"].rolling(w(20)).max()
    l = df["low"].rolling(w(20)).min()
    return (c - l) / (h - l + 1e-10)


def vol_swing(df: pd.DataFrame) -> pd.Series:
    """量价异动: |volume/MA(volume,20) - 1| × sign(return)"""
    v = df["volume"]
    vol_ratio = v / v.rolling(w(20)).mean() - 1
    sign = np.sign(df["close"].pct_change())
    return vol_ratio * sign


# ============================================================
#  因子注册表
# ============================================================

ALPHA101_FUNCTIONS: dict = {
    # 反转
    "rev_5": rev_5,
    "rev_10": rev_10,
    "rev_20": rev_20,
    # 动量
    "mom_20": mom_20,
    "mom_60": mom_60,
    "ema_ratio_5_20": ema_ratio_5_20,
    # 波动率
    "vol_20": vol_20,
    "atr_14": atr_14,
    # 量价
    "vol_ratio_5_20": vol_ratio_5_20,
    "vpt": vpt,
    "vwap_ratio": vwap_ratio,
    # 趋势
    "macd_dif": macd_dif,
    "macd_signal": macd_signal,
    "macd_hist": macd_hist,
    "rsi_14": rsi_14,
    "rsi_7": rsi_7,
    # 通道
    "bb_position": bb_position,
    "bb_width": bb_width,
    # 流动性
    "turnover_5": turnover_5,
    "illiquidity": illiquidity,
    "amount_ratio": amount_ratio,
    # 高阶
    "skewness_20": skewness_20,
    "kurtosis_20": kurtosis_20,
    "high_low_ratio": high_low_ratio,
    "max_dd_20": max_dd_20,
    "corr_c_v": corr_c_v,
    "co_ratio": co_ratio,
    "up_day_ratio": up_day_ratio,
    "price_position": price_position,
    "vol_swing": vol_swing,
}
