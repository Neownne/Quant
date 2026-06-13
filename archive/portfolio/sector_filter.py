"""板块过滤：根据板块打分筛选个股候选池。"""
from __future__ import annotations

import pandas as pd


def filter_by_top_sectors(
    stock_preds: pd.DataFrame,
    sector_scores: pd.DataFrame,
    code_to_sector: dict[str, str],
    top_n_sectors: int = 3,
) -> pd.DataFrame:
    """只保留得分最高的前 N 个板块中的股票。

    在个股预测和选股之间插入：先确定哪些板块值得参与，
    再将候选池缩小到这些板块内的股票。

    参数
    ----
    stock_preds : 个股预测结果，须含 code, score, rank 列
    sector_scores : 板块打分结果，须含 sector, score, rank 列，已排序
    code_to_sector : {code: sector_label} 映射
    top_n_sectors : 保留前几个板块

    返回
    ----
    过滤后的 stock_preds（保留原有列），按原 score 排序
    """
    if sector_scores.empty or stock_preds.empty:
        return stock_preds.copy()

    # 取前 N 个板块
    top_sectors = set(sector_scores.head(top_n_sectors)["sector"].values)

    # 过滤：只保留板块在 top_sectors 中且有板块映射的股票
    result = stock_preds.copy()
    result["_sector"] = result["code"].map(code_to_sector)
    result = result[result["_sector"].isin(top_sectors)]
    result = result.drop(columns=["_sector"])

    return result.reset_index(drop=True)
