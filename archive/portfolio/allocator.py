"""仓位分配。"""
from __future__ import annotations

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


def apply_position_limits(
    weights_df: pd.DataFrame,
    industry_map: dict[str, str] | None = None,
    max_single: float = 0.10,
    max_industry: float = 0.30,
    max_iterations: int = 20,
) -> pd.DataFrame:
    """对权重施加单只上限和行业上限约束。

    算法：迭代裁剪 + 按比例再分配，最终归一化。

    参数
    ----
    weights_df : 含 code, weight 列
    industry_map : {code: industry_label}，缺失则跳过行业限制
    max_single : 单只股票最大权重
    max_industry : 单行业最大总权重
    max_iterations : 最大迭代次数

    返回
    ----
    DataFrame: [code, weight]，权重和为 1.0
    """
    if weights_df.empty:
        return weights_df

    result = weights_df.copy()
    result["weight"] = result["weight"] / result["weight"].sum()
    industry_map = industry_map or {}

    for _ in range(max_iterations):
        clipped = False

        # 单只上限裁剪
        for i, row in result.iterrows():
            if row["weight"] > max_single:
                result.at[i, "weight"] = max_single
                clipped = True

        # 行业上限裁剪
        if industry_map:
            industry_weights: dict[str, float] = {}
            industry_codes: dict[str, list[int]] = {}
            for i, row in result.iterrows():
                ind = industry_map.get(row["code"], "")
                if not ind:
                    continue
                industry_weights[ind] = industry_weights.get(ind, 0) + row["weight"]
                industry_codes.setdefault(ind, []).append(i)

            for ind, total_w in industry_weights.items():
                if total_w > max_industry:
                    scale = max_industry / total_w
                    for idx in industry_codes.get(ind, []):
                        result.at[idx, "weight"] *= scale
                    clipped = True

        if not clipped:
            break

        # 归一化
        total = result["weight"].sum()
        if total > 0:
            result["weight"] = result["weight"] / total
        else:
            result["weight"] = 1.0 / len(result)

    return result.reset_index(drop=True)
