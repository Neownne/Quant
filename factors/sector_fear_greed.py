"""板块恐贪指数 — 严格按 FundDB 恐贪指数方法论。

官方规则（funddb.cn/tool/fear）：
  针对每个指标，观察相对于其自身历史平均水平的偏离程度，赋予 0-100 分值，
  50 为中立，分值越高越贪婪。所有指标等权平均。

适配板块级别的 5 项子指标：
  1. 波动率      — 20日收益标准差（高波动=恐惧）
  2. 资金流向    — 板块资金净流向
  3. 动量强度    — 20日等权收益
  4. 股价强度    — 创20日新高股票占比
  5. 涨跌比      — 上涨/下跌家数

每个指标：z-score → sigmoid → 0-100，最终等权平均。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEAR_GREED_FEATURES = [
    "fg_volatility",
    "fg_money_flow",
    "fg_momentum",
    "fg_new_high_ratio",
    "fg_advance_decline",
]


def _zscore_to_fg(z: float) -> float:
    """z-score 映射到 0-100 恐贪分值。

    z=0 → 50 (中立), z=+2 → ~88, z=-2 → ~12
    使用 sigmoid 函数，范围 (0, 100)，以 50 为中心。
    """
    return 100.0 / (1.0 + np.exp(-z))


def compute_sector_fear_greed(
    ohlcv: pd.DataFrame,
    sector_map: dict[str, str],
    target_date: pd.Timestamp,
    history_days: int = 250,
) -> dict[str, dict[str, float]]:
    """按官方方法论计算指定日期各板块恐贪指数。

    对每个板块的每个指标：
    1. 计算 target_date 之前 history_days 天内的均值和标准差
    2. 今日值偏离均值的程度（z-score）
    3. sigmoid 映射到 0-100
    4. 5项等权平均得最终恐贪指数

    返回: {sector: {fg_*, fg_composite}}
    """
    from factors.sector_breadth import compute_breadth_features

    ohlcv = ohlcv.copy()
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    target_date = pd.to_datetime(target_date)
    all_dates = sorted(ohlcv["trade_date"].unique())

    if target_date not in all_dates:
        return {}

    t_idx = all_dates.index(target_date)
    hist_start = max(0, t_idx - history_days)

    # 收集历史每天的板块特征
    hist_indicators: dict[str, dict[str, list[float]]] = {}
    for i in range(hist_start, t_idx + 1):
        d = all_dates[i]
        feats = compute_breadth_features(ohlcv, sector_map, d, lookback_days=20)
        if not feats:
            continue
        for sec, f in feats.items():
            if sec not in hist_indicators:
                hist_indicators[sec] = {k: [] for k in FEAR_GREED_FEATURES}
            hist_indicators[sec]["fg_volatility"].append(f.get("sector_vol_20", 0))
            hist_indicators[sec]["fg_money_flow"].append(f.get("money_flow_pct", 0))
            hist_indicators[sec]["fg_momentum"].append(f.get("sector_mom_20", 0))
            hist_indicators[sec]["fg_new_high_ratio"].append(
                f.get("new_high_20d", 0) / max(f.get("n_advancers", 1) + f.get("n_decliners", 1), 1)
            )
            hist_indicators[sec]["fg_advance_decline"].append(
                f.get("advance_decline_ratio", 1.0)
            )

    if not hist_indicators:
        return {}

    result = {}
    for sec, hist in hist_indicators.items():
        scores = {}
        for key in FEAR_GREED_FEATURES:
            vals = np.array(hist[key], dtype=float)
            if len(vals) < 20 or vals.std() < 1e-10:
                scores[key] = 50.0  # 数据不足时保持中立
            else:
                today_val = vals[-1]
                hist_mean = vals[:-1].mean()  # 不含今日的历史均值
                hist_std = vals[:-1].std()
                z = (today_val - hist_mean) / hist_std if hist_std > 0 else 0.0
                # 波动率：高=恐惧，取反
                if key == "fg_volatility":
                    z = -z
                scores[key] = float(_zscore_to_fg(z))

        # 等权平均
        composite = float(np.mean([scores[k] for k in FEAR_GREED_FEATURES]))

        result[sec] = {**scores, "fg_composite": composite}

    return result
