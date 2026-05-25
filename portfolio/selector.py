"""选股：模型打分 top-N + ST/停牌/次新过滤。"""
import pandas as pd
from datetime import date


def select_top_n(scores: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """从排序结果中选前 N 只。"""
    return scores.sort_values("rank").head(n).reset_index(drop=True)


def filter_stocks(
    stocks: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
    exclude_st: bool = True,
    min_list_days: int = 60,
) -> pd.DataFrame:
    """过滤不可交易的股票。

    参数
    ----
    stocks : 至少含 code, name 列
    ref_date : 参考日期（默认今天）
    exclude_st : 排除 ST
    min_list_days : 最小上市天数
    """
    result = stocks.copy()
    ref = ref_date or pd.Timestamp(date.today())

    if exclude_st and "name" in result.columns:
        result = result[~result["name"].str.contains("ST", na=False)]

    if "list_date" in result.columns:
        result["days_listed"] = (ref - pd.to_datetime(result["list_date"])).dt.days
        result = result[result["days_listed"] >= min_list_days]

    return result.reset_index(drop=True)
