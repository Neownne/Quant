"""因子筛选：相关性矩阵 + 正交性贪心筛选。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_factor_correlation(
    factor_df: pd.DataFrame,
    factor_names: list[str],
) -> pd.DataFrame:
    """计算因子间 Spearman 秩相关矩阵。"""
    clean = factor_df[factor_names].dropna()
    return clean.corr(method="spearman")


def select_orthogonal_factors(
    factor_df: pd.DataFrame,
    factor_names: list[str],
    threshold: float = 0.7,
) -> list[str]:
    """贪心筛选正交因子。

    算法：
    1. 按方差降序排列候选因子
    2. 逐个检验与已选因子的最大相关性
    3. max |corr| < threshold → 入选

    返回通过筛选的因子名列表。
    """
    if not factor_names:
        return []

    corr = compute_factor_correlation(factor_df, factor_names)

    variances = factor_df[factor_names].var().sort_values(ascending=False)
    sorted_factors = variances.index.tolist()

    selected = []
    for f in sorted_factors:
        ok = True
        for s in selected:
            if abs(corr.loc[f, s]) >= threshold:
                ok = False
                break
        if ok:
            selected.append(f)

    logger.info(
        f"正交筛选: {len(factor_names)} → {len(selected)} 个因子 (threshold={threshold})"
    )
    return selected
