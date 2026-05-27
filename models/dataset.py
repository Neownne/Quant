"""数据集构造：因子计算 + 标签生成 + walk-forward 切分。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from factors import FactorEngine


def make_labels(
    df: pd.DataFrame,
    forward_days: int = 1,
    mode: str = "binary",
) -> pd.DataFrame:
    """为每行计算 T+N 收益率标签。

    参数
    ----
    df : 单只股票 DataFrame，须含 close, trade_date，已排序
    forward_days : 前瞻天数（1 = T+1 收益）
    mode : "binary" → 0/1 涨跌；"regression" → 连续收益率

    返回
    ----
    带 'label' 列的 DataFrame
    """
    df = df.copy()
    future_close = df["close"].shift(-forward_days)
    if mode == "binary":
        label_series = (future_close > df["close"]).astype(int)
        label_series = label_series.where(future_close.notna(), np.nan)
        df["label"] = label_series
    else:
        df["label"] = (future_close - df["close"]) / df["close"]
    return df


def walk_forward_split(
    df: pd.DataFrame,
    train_years: int = 3,
    val_years: int = 1,
    date_col: str = "trade_date",
    gap_days: int = 0,
):
    """Walk-forward 滚动窗口迭代器。

    参数
    ----
    df : 含 date_col 的 DataFrame
    train_years : 训练窗口年数
    val_years : 验证窗口年数
    gap_days : train 和 val 之间的间隔天数（避免 look-ahead）

    Yields
    ------
    (train_df, val_df)
    """
    df = df.sort_values(date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    all_dates = sorted(df[date_col].unique())
    if not all_dates:
        return

    start = all_dates[0]
    end = all_dates[-1]

    train_start = pd.Timestamp(start)
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        val_start = train_end + pd.DateOffset(days=gap_days)
        val_end = val_start + pd.DateOffset(years=val_years)

        if val_start >= pd.Timestamp(end):
            break

        actual_val_end = min(val_end, pd.Timestamp(end))
        train_mask = (df[date_col] >= train_start) & (df[date_col] < train_end)
        val_mask = (df[date_col] >= val_start) & (df[date_col] < actual_val_end)

        train_df = df[train_mask]
        val_df = df[val_mask]

        if len(train_df) > 0 and len(val_df) > 0:
            yield train_df, val_df

        train_start = train_start + pd.DateOffset(years=1)


def build_factor_dataset(
    ohlcv: pd.DataFrame,
    factor_names: list[str],
    label_mode: str = "binary",
    forward_days: int = 1,
    extra_data: dict[str, pd.DataFrame] | None = None,
    bar_per_day: int = 1,
) -> pd.DataFrame:
    """从 OHLCV 构建带标签的因子数据集。

    参数
    ----
    ohlcv : 须含 code, trade_date, open, high, low, close, volume
    factor_names : 因子名列表
    label_mode : "binary" | "regression"
    forward_days : 标签前瞻天数
    extra_data : 传递给 FactorEngine 的额外数据
    bar_per_day : 每日 bar 数（60min=4, daily=1）。分钟频因子值聚合成日频用于标签。

    返回
    ----
    pd.DataFrame: [code, trade_date] + factor_names + [label]
    """
    engine = FactorEngine(factor_names=factor_names, bar_per_day=bar_per_day)
    logger.info(f"计算 {len(factor_names)} 个因子 ...")
    result = engine.compute(ohlcv, extra_data=extra_data)

    result["trade_date"] = pd.to_datetime(result["trade_date"])

    # 分钟频：因子值聚合成日频（取每日最后一根 bar 的因子值）
    if bar_per_day > 1:
        group_cols = ["code", "trade_date"]
        factor_vals = result[group_cols + factor_names].copy()
        result = factor_vals.groupby(group_cols, as_index=False).last()

    # 构建日频 close 用于标签
    ohlcv_sub = ohlcv[["code", "trade_date", "close"]].copy()
    ohlcv_sub["trade_date"] = pd.to_datetime(ohlcv_sub["trade_date"])
    if bar_per_day > 1:
        ohlcv_sub = ohlcv_sub.groupby(["code", "trade_date"], as_index=False).last()

    result = result.merge(ohlcv_sub, on=["code", "trade_date"], how="left")

    # 按股票分组计算标签
    logger.info("生成标签 ...")
    labelled_parts = []
    for code, group in result.groupby("code"):
        group = group.sort_values("trade_date")
        labelled_parts.append(make_labels(group, forward_days, label_mode))

    result = pd.concat(labelled_parts, ignore_index=True)

    # 计算 T+1 连续收益率（供 IC 计算用）
    ret_parts = []
    for code, group in result.groupby("code"):
        group = group.sort_values("trade_date")
        group["ret_1d"] = group["close"].pct_change().shift(-1)
        ret_parts.append(group)
    result = pd.concat(ret_parts, ignore_index=True)

    # 如果前瞻天数 > 1，同时计算对应周期的收益率
    if forward_days > 1:
        ret_col = f"ret_{forward_days}d"
        ret_parts = []
        for code, group in result.groupby("code"):
            group = group.sort_values("trade_date")
            group[ret_col] = group["close"].pct_change(periods=forward_days).shift(-forward_days)
            ret_parts.append(group)
        result = pd.concat(ret_parts, ignore_index=True)

    result = result.drop(columns=["close"])
    logger.info(f"数据集: {len(result)} 行, {len(result.dropna())} 有效")
    return result
