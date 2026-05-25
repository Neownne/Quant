"""Alpha191 资金流向类因子。"""
import numpy as np
import pandas as pd


def money_flow(df: pd.DataFrame) -> pd.Series:
    """资金流：Σ((C-L)-(H-C))×V / ΣV, 10日。正=流入。"""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    mf = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    raw_mf = mf * v
    return raw_mf.rolling(10).sum() / v.rolling(10).sum().replace(0, np.nan)


def obv_roc(df: pd.DataFrame) -> pd.Series:
    """OBV 变化率：(OBV_t - OBV_{t-20}) / |OBV_{t-20}|。"""
    c, v = df["close"], df["volume"]
    direction = np.sign(c.diff())
    obv = (direction * v).fillna(0).cumsum()
    lag = obv.shift(20).abs().replace(0, np.nan)
    return (obv - obv.shift(20)) / lag


def force_index(df: pd.DataFrame) -> pd.Series:
    """强力指数：EMA(ΔC × V, 2)。"""
    c, v = df["close"], df["volume"]
    fi_raw = c.diff() * v
    return fi_raw.ewm(span=2, adjust=False).mean()


def cwt(df: pd.DataFrame) -> pd.Series:
    """CWT：C×V×turnover 的 5 日变化率。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    raw = df["close"] * df["volume"] * df["turnover"]
    lag = raw.shift(5).abs().replace(0, np.nan)
    return (raw - raw.shift(5)) / lag


def volume_climax(df: pd.DataFrame) -> pd.Series:
    """天量见顶：(V_t - max_{t-20..t-1}) / max_{t-20..t-1}，取负。"""
    v = df["volume"]
    rolling_max = v.shift(1).rolling(20).max()
    climax = (v - rolling_max) / rolling_max.replace(0, np.nan)
    return -climax


def vwap_momentum(df: pd.DataFrame) -> pd.Series:
    """VWAP 动量：VWAP(5) / VWAP(20) - 1。"""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    typ = (h + l + c) / 3
    tv = typ * v
    vwap5 = tv.rolling(5).sum() / v.rolling(5).sum().replace(0, np.nan)
    vwap20 = tv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    return vwap5 / vwap20.replace(0, np.nan) - 1


ALPHA191_FLOW: dict = {
    "money_flow": money_flow,
    "obv_roc": obv_roc,
    "force_index": force_index,
    "cwt": cwt,
    "volume_climax": volume_climax,
    "vwap_momentum": vwap_momentum,
}
