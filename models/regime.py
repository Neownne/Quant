"""市场状态识别：大盘年线 + 波动率分位。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def detect_regime(
    index_df: pd.DataFrame,
    date_col: str = "trade_date",
    price_col: str = "close",
    ma_period: int = 250,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    """识别每日市场状态。

    规则：
    - bull: 价格 > MA250 且 20 日收益 > 0
    - bear: 价格 < MA250 且 20 日收益 < 0
    - sideways: 其余情况

    返回 DataFrame: [trade_date, regime]
    """
    df = index_df.sort_values(date_col).copy()
    df["ma250"] = df[price_col].rolling(ma_period, min_periods=60).mean()
    df["ret_20"] = df[price_col].pct_change(20)

    conditions = [
        (df[price_col] > df["ma250"]) & (df["ret_20"] > 0),
        (df[price_col] < df["ma250"]) & (df["ret_20"] < 0),
    ]
    choices = ["bull", "bear"]
    df["regime"] = np.select(conditions, choices, default="sideways")

    logger.info(
        f"市场状态: bull={(df['regime']=='bull').sum()}, "
        f"bear={(df['regime']=='bear').sum()}, "
        f"sideways={(df['regime']=='sideways').sum()}"
    )
    return df[[date_col, "regime"]]
