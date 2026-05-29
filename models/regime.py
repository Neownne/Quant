"""市场状态识别：5 状态细分（强牛/弱牛/慢熊/快熊/震荡）。

用法:
    from models.regime import detect_regime
    regime_df = detect_regime(index_df)  # → [trade_date, regime]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def detect_regime(
    index_df: pd.DataFrame,
    date_col: str = "trade_date",
    price_col: str = "close",
    ma_period: int = 250,
    strong_bull_threshold: float = 0.03,
    fast_bear_threshold: float = -0.03,
) -> pd.DataFrame:
    """识别每日市场状态（5 分类）。

    规则:
      strong_bull : close > MA250 且 20 日收益 > +3%  (强上涨趋势)
      weak_bull   : close > MA250 且 0 < 20 日收益 ≤ +3%  (弱牛/横盘牛)
      fast_bear   : close < MA250 且 20 日收益 < -3%  (暴跌/快熊)
      slow_bear   : close < MA250 且 -3% ≤ 20 日收益 < 0  (阴跌/慢熊)
      sideways    : 其余情况

    阈值选择依据：上证指数 2015-2026 年 bull 状态 ret_20 中位数 2.9%，
    bear 状态 ret_20 中位数 -3.4%。取 ±3% 作为分界线。

    返回 DataFrame: [trade_date, regime]
    """
    df = index_df.sort_values(date_col).copy()
    df["ma250"] = df[price_col].rolling(ma_period, min_periods=60).mean()
    df["ret_20"] = df[price_col].pct_change(20)

    above = df[price_col] > df["ma250"]
    below = df[price_col] < df["ma250"]
    strong_up = df["ret_20"] > strong_bull_threshold
    mild_up = (df["ret_20"] > 0) & (df["ret_20"] <= strong_bull_threshold)
    strong_down = df["ret_20"] < fast_bear_threshold
    mild_down = (df["ret_20"] < 0) & (df["ret_20"] >= fast_bear_threshold)

    conditions = [
        above & strong_up,
        above & mild_up,
        below & strong_down,
        below & mild_down,
    ]
    choices = ["strong_bull", "weak_bull", "fast_bear", "slow_bear"]
    df["regime"] = np.select(conditions, choices, default="sideways")

    counts = df["regime"].value_counts()
    logger.info(
        f"市场状态: strong_bull={counts.get('strong_bull',0)}, "
        f"weak_bull={counts.get('weak_bull',0)}, "
        f"fast_bear={counts.get('fast_bear',0)}, "
        f"slow_bear={counts.get('slow_bear',0)}, "
        f"sideways={counts.get('sideways',0)}"
    )
    return df[[date_col, "regime"]]


# ── 5-state aliases for per-state strategy tuning ──

REGIME_GROUPS = {
    "strong_bull": "bull",
    "weak_bull": "bull",
    "fast_bear": "bear",
    "slow_bear": "bear",
    "sideways": "sideways",
}

# Per-state strategy parameters (override TradingConfig defaults)
REGIME_PARAMS = {
    "strong_bull": {
        "top_n": 15,
        "rebalance_freq": 1,
        "stop_loss_pct": 0.08,
        "position_ratio": 1.0,
    },
    "weak_bull": {
        "top_n": 10,
        "rebalance_freq": 5,      # 周度调仓减少噪声交易
        "stop_loss_pct": 0.07,
        "position_ratio": 1.0,    # 保持满仓，弱牛也能赚钱
    },
    "fast_bear": {
        "top_n": 5,
        "rebalance_freq": 5,
        "stop_loss_pct": 0.04,
        "position_ratio": 0.3,    # 暴跌时空仓为主
    },
    "slow_bear": {
        "top_n": 5,               # 更少的持仓
        "rebalance_freq": 5,
        "stop_loss_pct": 0.04,    # 更紧的止损
        "position_ratio": 0.4,    # 阴跌也大幅降仓
    },
    "sideways": {
        "top_n": 10,
        "rebalance_freq": 5,
        "stop_loss_pct": 0.06,
        "position_ratio": 1.0,    # 震荡市满仓
    },
}
