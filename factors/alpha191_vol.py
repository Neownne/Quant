"""Alpha191 波动率高阶类因子。"""
import numpy as np
import pandas as pd


def vol_of_vol(df: pd.DataFrame) -> pd.Series:
    """波动率波动：std(std(ret,5), 20)，取负。"""
    ret = df["close"].pct_change()
    vol5 = ret.rolling(5).std()
    return -vol5.rolling(20).std()


def down_vol_ratio(df: pd.DataFrame) -> pd.Series:
    """下行波动占比：std(ret_neg, 20) / std(ret, 20)，取负。"""
    ret = df["close"].pct_change()
    ret_neg = ret.clip(upper=0)
    down_std = ret_neg.rolling(20).std()
    total_std = ret.rolling(20).std().replace(0, np.nan)
    return -down_std / total_std


def tail_risk(df: pd.DataFrame) -> pd.Series:
    """尾部风险：ret 5% 分位数(60日)，取负。"""
    ret = df["close"].pct_change()
    return -ret.rolling(60).quantile(0.05)


def beta_20(df: pd.DataFrame) -> pd.Series:
    """20 日 Beta（以自身收益替代市场——实盘需 index）。"""
    ret = df["close"].pct_change()
    mkt = ret  # 占位，实盘替换为指数收益
    cov = ret.rolling(20).cov(mkt)
    var = mkt.rolling(20).var().replace(0, np.nan)
    return cov / var


def ret_asymmetry(df: pd.DataFrame) -> pd.Series:
    """收益不对称度：(mean(pos) - |mean(neg)|) / std。"""
    ret = df["close"].pct_change()
    pos = ret.clip(lower=0).rolling(20).mean()
    neg = ret.clip(upper=0).abs().rolling(20).mean()
    std = ret.rolling(20).std().replace(0, np.nan)
    return (pos - neg) / std


ALPHA191_VOL: dict = {
    "vol_of_vol": vol_of_vol,
    "down_vol_ratio": down_vol_ratio,
    "tail_risk": tail_risk,
    "beta_20": beta_20,
    "ret_asymmetry": ret_asymmetry,
}
