"""板块数据集构建：从个股OHLCV聚合板块特征 + 标签 + walk-forward切分。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from factors.sector_breadth import compute_breadth_features, FEATURE_NAMES


def build_sector_dataset(
    ohlcv: pd.DataFrame,
    sector_map: dict[str, str],
    forward_days: int = 5,
    label_mode: str = "binary",
    lookback_days: int = 20,
) -> pd.DataFrame:
    """从个股 OHLCV 构建板块级带标签数据集。

    每天每个板块计算：
    1. 板块宽度特征（通过 compute_breadth_features）
    2. 板块未来 N 日等权收益（作为标签基准）
    3. 标签 = 板块 N 日收益是否跑赢当日所有板块中位数（二分类）

    参数
    ----
    ohlcv : 个股日线数据，须含 code, trade_date, open, high, low, close, volume, amount, turnover
    sector_map : {code: sector_label} 映射
    forward_days : 标签前瞻天数
    label_mode : "binary" → 0/1（跑赢/跑输中位数）；"regression" → 连续 N 日收益率
    lookback_days : 板块特征回溯天数

    返回
    ----
    pd.DataFrame: [trade_date, sector] + FEATURE_NAMES + [label, ret_Nd]
    """
    if ohlcv.empty:
        return pd.DataFrame()

    ohlcv = ohlcv.copy()
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])

    # 过滤有板块映射的股票（不在原始数据上加 sector 列，避免干扰 breadth 计算）
    mapped_codes = {c for c in ohlcv["code"].unique() if c in sector_map}
    ohlcv = ohlcv[ohlcv["code"].isin(mapped_codes)]
    all_dates = sorted(ohlcv["trade_date"].unique())

    if ohlcv.empty:
        return pd.DataFrame()

    records = []
    for i, date in enumerate(all_dates):
        # 计算板块特征
        feats = compute_breadth_features(ohlcv, sector_map, date, lookback_days=lookback_days)
        if not feats:
            continue

        # 计算各板块未来 N 日收益
        future_date_idx = i + forward_days
        if future_date_idx >= len(all_dates):
            continue

        future_date = all_dates[future_date_idx]
        today_df = ohlcv[ohlcv["trade_date"] == date]
        future_df = ohlcv[ohlcv["trade_date"] == future_date]

        for sector in feats:
            sector_codes = [c for c, s in sector_map.items() if s == sector]
            today_sector = today_df[today_df["code"].isin(sector_codes)]
            future_sector = future_df[future_df["code"].isin(sector_codes)]

            # 计算等权板块 N 日收益
            if today_sector.empty or future_sector.empty:
                continue

            rets = []
            for code in sector_codes:
                t_row = today_sector[today_sector["code"] == code]
                f_row = future_sector[future_sector["code"] == code]
                if not t_row.empty and not f_row.empty:
                    t_close = t_row.iloc[0]["close"]
                    f_close = f_row.iloc[0]["close"]
                    rets.append((f_close - t_close) / t_close)

            if not rets:
                continue

            ret_nd = float(np.mean(rets))
            row = {"trade_date": date, "sector": sector, "ret_nd": ret_nd, **feats[sector]}
            records.append(row)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    # 标签：基于截面中位数比较
    if label_mode == "binary":
        result["label"] = result.groupby("trade_date")["ret_nd"].transform(
            lambda x: (x > x.median()).astype(int)
        )
    else:
        result["label"] = result["ret_nd"]

    # 重命名 ret_nd -> ret_{N}d（与个股数据集命名一致）
    result = result.rename(columns={"ret_nd": f"ret_{forward_days}d"})

    logger.info(f"板块数据集: {len(result)} 行, {result['sector'].nunique()} 板块, "
                f"{result['trade_date'].nunique()} 天")
    return result
