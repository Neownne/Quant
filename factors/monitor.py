"""因子 IC 监控管线。

提供:
- compute_rank_ic: 单截面 RankIC (Spearman rank correlation)
- compute_ic_series: 因子矩阵 → 逐日 IC
- compute_ic_summary: IC/ICIR 汇总
- compute_ic_decay: IC 衰减曲线
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_rank_ic(factor: np.ndarray, ret: np.ndarray) -> float:
    """计算截面 RankIC（Spearman 秩相关系数）。

    参数
    ----
    factor : 因子值数组
    ret : 对应收益率数组

    返回
    ----
    RankIC ∈ [-1, 1]，有效样本 < 5 返回 NaN
    """
    mask = np.isfinite(factor) & np.isfinite(ret)
    if mask.sum() < 5:
        return np.nan
    ic, _ = spearmanr(factor[mask], ret[mask])
    return ic if not np.isnan(ic) else np.nan


def compute_ic_series(
    df: pd.DataFrame, factor_cols: list[str], ret_col: str = "ret_1d"
) -> pd.DataFrame:
    """逐日计算每个因子的 RankIC。

    参数
    ----
    df : 须包含 trade_date, 以及 factor_cols 和 ret_col
    factor_cols : 因子列名列表
    ret_col : 收益率列名

    返回
    ----
    pd.DataFrame: 行=trade_date, 列=factor_cols, 值=RankIC
    """
    records = []
    for dt, group in df.groupby("trade_date", sort=True):
        row = {"trade_date": dt}
        for fcol in factor_cols:
            row[fcol] = compute_rank_ic(group[fcol].values, group[ret_col].values)
        records.append(row)

    return pd.DataFrame(records).sort_values("trade_date").reset_index(drop=True)


def compute_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """汇总 IC 统计量。

    返回 DataFrame: index=factor, columns=ic_mean, ic_std, icir, n_days
    """
    factor_cols = [c for c in ic_df.columns if c != "trade_date"]
    rows = []
    for fcol in factor_cols:
        ics = ic_df[fcol].dropna()
        if len(ics) == 0:
            rows.append({
                "factor": fcol,
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "icir": np.nan,
                "n_days": 0,
            })
        else:
            rows.append({
                "factor": fcol,
                "ic_mean": ics.mean(),
                "ic_std": ics.std(ddof=0),
                "icir": ics.mean() / ics.std(ddof=0) if ics.std() > 0 else np.nan,
                "n_days": len(ics),
            })
    return pd.DataFrame(rows).set_index("factor")


def compute_ic_decay(
    df: pd.DataFrame, factor_col: str, ret_cols: list[str]
) -> pd.DataFrame:
    """IC 衰减曲线: 因子对未来不同时间窗口收益的预测力。

    参数
    ----
    factor_col : 因子名
    ret_cols : 不同时间窗口的收益率列（如 ["ret_1d", "ret_3d", "ret_5d"]）

    返回
    ----
    pd.DataFrame: columns=horizon, mean_ic
    """
    # 逐日计算该因子对各 ret 窗口的 IC
    records = []
    for dt, group in df.groupby("trade_date", sort=True):
        row = {"trade_date": dt}
        for rcol in ret_cols:
            row[rcol] = compute_rank_ic(group[factor_col].values, group[rcol].values)
        records.append(row)

    ic_df = pd.DataFrame(records)

    # 汇总每个 horizon 的均值
    summary = {}
    for rcol in ret_cols:
        ics = ic_df[rcol].dropna()
        summary[rcol] = float(ics.mean()) if len(ics) > 0 else np.nan

    return pd.DataFrame(
        {"horizon": list(summary.keys()), "mean_ic": list(summary.values())}
    )
