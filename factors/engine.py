"""因子计算引擎。

用法:
    engine = FactorEngine(factor_names=["rsi_14", "mom_20"])
    factor_matrix = engine.compute(df_ohlcv)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import factors


class FactorEngine:
    """因子计算引擎。

    参数
    ----
    factor_names : 要计算的因子名列表，必须在 ALL_FACTORS 中注册。
    """

    def __init__(self, factor_names: list[str]):
        if factor_names:
            missing = set(factor_names) - set(factors.ALL_FACTORS.keys())
            if missing:
                raise KeyError(f"未知因子: {missing}")
        self.factor_names = factor_names

    def compute(self, df: pd.DataFrame, extra_data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        """计算因子矩阵。

        参数
        ----
        df : DataFrame, 须包含 code, trade_date, open, high, low, close, volume
             按 code 分组，trade_date 排序。
        extra_data : 可选，{列名: DataFrame}，用于注入财务/估值等额外列。
                     每个 DataFrame 须有 code 和 trade_date/report_date 列。

        返回
        ----
        pd.DataFrame: 列 = [code, trade_date] + factor_names
        """
        if not self.factor_names:
            return df[["code", "trade_date"]].copy()

        df = df.sort_values(["code", "trade_date"]).copy()

        # 合并额外数据
        if extra_data:
            for col_name, extra_df in extra_data.items():
                date_col = "report_date" if "report_date" in extra_df.columns else "trade_date"
                df = pd.merge(
                    df, extra_df[["code", date_col, col_name]],
                    left_on=["code", "trade_date"],
                    right_on=["code", date_col],
                    how="left",
                ).drop(columns=[date_col])

        result_parts = []

        for code, group in df.groupby("code"):
            part = group[["code", "trade_date"]].copy()
            for name in self.factor_names:
                fn = factors.ALL_FACTORS[name]
                part[name] = fn(group)
            result_parts.append(part)

        result = pd.concat(result_parts, ignore_index=True)
        return result


def neutralize(
    factor: pd.Series,
    exposures: pd.DataFrame,
    groups: pd.Series | None = None,
) -> pd.Series:
    """截面中性化：用线性回归去除 factor 中的 exposures 影响。

    参数
    ----
    factor : 因子值序列
    exposures : 暴露矩阵（如 log_mcap, industry dummies）
    groups : 可选，分组键（如 trade_date），在每个组内独立中性化

    返回
    ----
    残差序列（与原索引对齐）
    """
    if groups is not None:
        result = pd.Series(np.nan, index=factor.index, dtype=float)
        for _, idx in groups.groupby(groups).groups.items():
            result.loc[factor.index.intersection(idx)] = neutralize(
                factor.loc[factor.index.intersection(idx)],
                exposures.loc[exposures.index.intersection(idx)],
            )
        return result

    valid = factor.notna() & exposures.notna().all(axis=1)
    if valid.sum() < 10:
        return factor

    X = exposures.loc[valid].values.astype(float)
    y = factor.loc[valid].values.astype(float)

    if X.shape[1] == 0:
        return factor

    model = LinearRegression()
    model.fit(X, y)
    predicted = model.predict(X)
    residuals = y - predicted

    result = pd.Series(np.nan, index=factor.index, dtype=float)
    result.loc[valid] = residuals
    return result
