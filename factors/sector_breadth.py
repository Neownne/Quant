"""板块宽度特征计算。

从个股 OHLCV 数据聚合计算各板块的市场宽度指标，包括：
- 涨跌家数/涨跌比
- 涨停/跌停家数
- 资金流向
- 板块动量与波动率
- 新高/新低家数
- 集中度

无需额外数据源，完全从 stock_daily 表计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 所有板块特征名称
FEATURE_NAMES = (
    "advance_decline_ratio",
    "n_advancers",
    "n_decliners",
    "n_limit_up",
    "n_limit_down",
    "up_volume_ratio",
    "sector_ret_mean",
    "sector_ret_std",
    "sector_turnover_mean",
    "new_high_20d",
    "new_low_20d",
    "money_flow_pct",
    "concentration_top3",
    "sector_mom_5",
    "sector_mom_20",
    "sector_vol_20",
)

# 各板块涨停阈值（涨跌幅限制）
_LIMIT_THRESHOLDS = {
    "科创": 0.20,
    "北证": 0.30,
}
_DEFAULT_LIMIT = 0.10  # 主板（含红利、大盘、小盘）


def _get_limit_threshold(sector: str) -> float:
    """获取某板块的涨跌停阈值。"""
    return _LIMIT_THRESHOLDS.get(sector, _DEFAULT_LIMIT)


def compute_single_day_features(
    ohlcv: pd.DataFrame,
    sector_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    """从单日截面数据计算板块特征（需要前一天收盘价以计算收益）。

    参数
    ----
    ohlcv : 至少包含最近两个交易日的数据，列含 code, trade_date, close, volume, amount, turnover
    sector_map : {code: sector_label} 映射

    返回
    ----
    {sector_label: {feature_name: value}} 嵌套字典
    """
    if ohlcv.empty:
        return {}

    dates = sorted(ohlcv["trade_date"].unique())
    if len(dates) < 2:
        return {}

    today = dates[-1]
    yesterday = dates[-2]

    today_df = ohlcv[ohlcv["trade_date"] == today].copy()
    yesterday_df = ohlcv[ohlcv["trade_date"] == yesterday][["code", "close"]].copy()
    yesterday_df = yesterday_df.rename(columns={"close": "prev_close"})

    # 合并当日数据与前日收盘
    merged = today_df.merge(yesterday_df, on="code", how="left")
    # 无法计算收益的去掉（新股或缺失数据）
    merged = merged.dropna(subset=["prev_close"])
    merged["ret"] = (merged["close"] - merged["prev_close"]) / merged["prev_close"]

    # 映射板块
    merged["sector"] = merged["code"].map(sector_map)
    merged = merged.dropna(subset=["sector"])

    result = {}
    for sector, group in merged.groupby("sector"):
        n = len(group)
        if n == 0:
            continue

        limit = _get_limit_threshold(sector)
        ret = group["ret"]

        advancers = (ret > 0).sum()
        decliners = (ret < 0).sum()

        # 涨停/跌停
        n_limit_up = (ret >= limit * 0.99).sum()  # 0.99 容忍浮点误差
        n_limit_down = (ret <= -limit * 0.99).sum()

        # 资金流向
        up_amount = group.loc[ret > 0, "amount"].sum()
        total_amount = group["amount"].sum()

        # 集中度（按成交额前3占比）
        top3_amount = group["amount"].nlargest(3).sum()

        features = {
            "n_advancers": int(advancers),
            "n_decliners": int(decliners),
            "advance_decline_ratio": advancers / decliners if decliners > 0 else float(advancers),
            "n_limit_up": int(n_limit_up),
            "n_limit_down": int(n_limit_down),
            "up_volume_ratio": up_amount / total_amount if total_amount > 0 else 0.0,
            "sector_ret_mean": float(ret.mean()),
            "sector_ret_std": float(ret.std()) if n > 1 else 0.0,
            "sector_turnover_mean": float(group["turnover"].mean()) if "turnover" in group.columns else 0.0,
            "money_flow_pct": (up_amount - (total_amount - up_amount)) / total_amount if total_amount > 0 else 0.0,
            "concentration_top3": top3_amount / total_amount if total_amount > 0 else 0.0,
            # 多日特征由 compute_breadth_features 填充，这里给默认值
            "new_high_20d": 0,
            "new_low_20d": 0,
            "sector_mom_5": 0.0,
            "sector_mom_20": 0.0,
            "sector_vol_20": 0.0,
        }
        result[sector] = features

    return result


def compute_breadth_features(
    ohlcv: pd.DataFrame,
    sector_map: dict[str, str],
    target_date: pd.Timestamp,
    lookback_days: int = 20,
) -> dict[str, dict[str, float]]:
    """计算指定日期的板块宽度特征（含多日回溯特征）。

    参数
    ----
    ohlcv : 多日 OHLCV 数据，列含 code, trade_date, close, high, low, volume, amount, turnover
    sector_map : {code: sector_label} 映射
    target_date : 目标交易日
    lookback_days : 新高/新低/波动率的回溯天数（默认20）

    返回
    ----
    {sector_label: {feature_name: value}} 嵌套字典，包含全部 FEATURE_NAMES 特征
    """
    ohlcv = ohlcv.copy()
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    target_date = pd.to_datetime(target_date)

    all_dates = sorted(ohlcv["trade_date"].unique())
    if target_date not in all_dates:
        return {}

    # 找到 target_date 在日期列表中的位置
    date_idx = all_dates.index(target_date)

    # 1. 单日截面特征（需要 target_date 和前一天）
    recent = ohlcv[ohlcv["trade_date"].isin(all_dates[max(0, date_idx - 1):date_idx + 1])]
    result = compute_single_day_features(recent, sector_map)

    if not result:
        return result

    # 2. 多日回溯特征
    lookback_start = all_dates[max(0, date_idx - lookback_days)]
    hist = ohlcv[(ohlcv["trade_date"] >= lookback_start) & (ohlcv["trade_date"] <= target_date)]
    hist["sector"] = hist["code"].map(sector_map)
    hist = hist.dropna(subset=["sector"])

    today_df = ohlcv[ohlcv["trade_date"] == target_date]

    for sector, group in hist.groupby("sector"):
        if sector not in result:
            continue

        codes_in_sector = group["code"].unique()
        sector_today = today_df[today_df["code"].isin(codes_in_sector)]

        # 20日新高/新低（按当日收盘价 vs 历史最高/最低）
        n_high = 0
        n_low = 0
        for code in codes_in_sector:
            stock_hist = group[group["code"] == code].sort_values("trade_date")
            if len(stock_hist) < 2:
                continue
            today_close = stock_hist.iloc[-1]["close"]
            hist_high = stock_hist.iloc[:-1]["high"].max()
            hist_low = stock_hist.iloc[:-1]["low"].min()
            if today_close >= hist_high:
                n_high += 1
            if today_close <= hist_low:
                n_low += 1

        result[sector]["new_high_20d"] = n_high
        result[sector]["new_low_20d"] = n_low

        # 板块动量：等权平均日收益
        daily_close_mean = group.groupby("trade_date")["close"].mean()
        daily_rets = daily_close_mean.pct_change().dropna()
        if isinstance(daily_rets, (int, float, np.floating)):
            continue  # 数据不足以计算动量

        if len(daily_rets) >= 5:
            result[sector]["sector_mom_5"] = float(daily_rets.iloc[-5:].mean())
        if len(daily_rets) >= 20:
            result[sector]["sector_mom_20"] = float(daily_rets.iloc[-20:].mean())
            result[sector]["sector_vol_20"] = float(daily_rets.iloc[-20:].std())

    return result
