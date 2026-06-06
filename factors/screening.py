"""因子筛选：IC 门禁 + 相关性矩阵 + 正交性贪心筛选。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def filter_factors_by_ic(
    factor_df: pd.DataFrame,
    factor_names: list[str],
    ret_col: str = "ret_1d",
    ic_threshold: float = 0.03,
    t_threshold: float = 2.0,
) -> list[str]:
    """IC 门禁：过滤预测力不足的因子。

    对每个因子计算逐日 RankIC，保留满足以下条件的因子：
    - |mean IC| > ic_threshold (default 0.02)
    - |t-statistic| > t_threshold (default 2.0)

    返回通过门禁的因子名列表。
    """
    from factors.monitor import compute_ic_series, compute_ic_summary

    if not factor_names:
        return []

    # 只要求 ret_1d 和 trade_date 非空（因子 NaN 由 compute_rank_ic 内部处理）
    valid = factor_df[[ret_col, "trade_date"] + factor_names].dropna(subset=[ret_col, "trade_date"])
    if len(valid) < 100:
        logger.warning("IC 门禁: 有效样本不足，跳过过滤")
        return factor_names

    ic_df = compute_ic_series(valid, factor_names, ret_col=ret_col)
    summary = compute_ic_summary(ic_df)

    passed = []
    for f in factor_names:
        if f not in summary.index:
            continue
        row = summary.loc[f]
        ic_mean = row["ic_mean"]
        n_days = row["n_days"]
        if pd.isna(ic_mean) or n_days < 10:
            continue
        # t = ic_mean / (ic_std / sqrt(n_days)) = ICIR * sqrt(n_days)
        ic_std = row["ic_std"]
        t_stat = abs(ic_mean) / (ic_std / np.sqrt(n_days)) if ic_std > 0 else 0
        if abs(ic_mean) > ic_threshold and t_stat > t_threshold:
            passed.append(f)
        else:
            logger.debug(
                f"IC 门禁淘汰: {f}, |IC|={abs(ic_mean):.4f}, t={t_stat:.2f}"
            )

    logger.info(
        f"IC 门禁: {len(factor_names)} → {len(passed)} 个因子 "
        f"(|IC|>{ic_threshold}, |t|>{t_threshold})"
    )
    return passed


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
    threshold: float = 0.5,
    ic_summary: pd.DataFrame | None = None,
) -> list[str]:
    """贪心筛选正交因子。

    算法：
    1. 按 |IC mean| 降序排列（若提供 ic_summary），否则按方差
    2. 逐个检验与已选因子的最大 Spearman 相关性
    3. max |corr| < threshold → 入选

    返回通过筛选的因子名列表。
    """
    if not factor_names:
        return []

    corr = compute_factor_correlation(factor_df, factor_names)

    # 排序：优先按 IC 均值，其次按方差
    if ic_summary is not None and not ic_summary.empty:
        ic_abs = ic_summary["ic_mean"].abs()
        sorted_factors = [f for f in ic_abs.sort_values(ascending=False).index
                          if f in factor_names]
        remaining = [f for f in factor_names if f not in sorted_factors]
        sorted_factors += remaining
    else:
        variances = factor_df[factor_names].var().sort_values(ascending=False)
        sorted_factors = variances.index.tolist()

    selected = []
    for f in sorted_factors:
        if f not in corr.index:
            selected.append(f)
            continue
        ok = True
        for s in selected:
            if s in corr.index and abs(corr.loc[f, s]) >= threshold:
                ok = False
                break
        if ok:
            selected.append(f)

    logger.info(
        f"正交筛选: {len(factor_names)} → {len(selected)} 个因子 (threshold={threshold})"
    )
    return selected
