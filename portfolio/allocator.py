"""仓位分配。"""
import numpy as np
import pandas as pd


def equal_weight(codes: list[str], cash: float) -> pd.DataFrame:
    """等权分配。

    返回 DataFrame: [code, weight, value]
    """
    n = len(codes)
    weight = 1.0 / n
    return pd.DataFrame({
        "code": codes,
        "weight": weight,
        "value": cash * weight,
    })


def volatility_inverse_weight(
    codes: list[str],
    returns_matrix: pd.DataFrame,
    cash: float,
    lookback: int = 60,
) -> pd.DataFrame:
    """波动率倒数加权。

    参数
    ----
    codes : 股票代码列表
    returns_matrix : DataFrame, 列为 code, 行=日期, 值=日收益率
    cash : 总资金
    lookback : 波动率回看窗口

    返回
    ----
    DataFrame: [code, weight, vol, value]
    """
    vols = {}
    for c in codes:
        if c in returns_matrix.columns:
            r = returns_matrix[c].tail(lookback).dropna()
            vols[c] = r.std() if len(r) > 10 else 1.0
        else:
            vols[c] = 1.0

    inv_vols = {c: 1.0 / max(v, 0.001) for c, v in vols.items()}
    total = sum(inv_vols.values())
    weights = {c: v / total for c, v in inv_vols.items()}

    return pd.DataFrame({
        "code": list(weights.keys()),
        "weight": list(weights.values()),
        "vol": [vols[c] for c in codes],
        "value": [cash * weights[c] for c in codes],
    })
