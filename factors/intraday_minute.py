"""日内分钟频衍生因子。

数据源: stock_minute 表，60min K线，每只股票每天 4 根 bar。

A股交易时段与60min bar映射::

    上午 9:30-11:30         下午 13:00-15:00
    ─────────────────       ─────────────────
    bar0: 10:30  9:31-10:30  bar2: 14:00  13:01-14:00
    bar1: 11:30  10:31-11:30 bar3: 15:00  14:01-15:00

这些因子依赖 build_intraday_daily_features() 预聚合的日频列，通过 extra_data
管道注入 FactorEngine。无分钟数据的股票对应列为 NaN，XGBoost/LightGBM 原生处理。
"""

import numpy as np
import pandas as pd


def build_intraday_daily_features(
    minute_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """将 60min K 线聚合成日频特征列。

    Parameters
    ----------
    minute_df: 含列 [code, trade_time, period, open, high, low, close, volume, amount]
              仅处理 period='60' 的数据

    Returns
    -------
    dict[str, pd.DataFrame]: {列名: DataFrame[code, trade_date, value]}
    """
    df = minute_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_time"]).dt.date

    # 按 (code, trade_date) 分组聚合
    grouped = df.sort_values(["code", "trade_date", "trade_time"]).groupby(
        ["code", "trade_date"]
    )

    features: dict[str, pd.DataFrame] = {}

    # ── 构建每组的 bar 列表 ──
    # 我们依赖 position 而非时间标签，以应对可能的标签差异
    def _agg_bars(grp):
        """返回一天内 4 根 bar 的 OHLCV 数组，不足 4 根则补 NaN。"""
        if len(grp) < 4:
            return pd.Series({
                "n_bars": len(grp),
                "open_0": np.nan, "high_0": np.nan, "low_0": np.nan, "close_0": np.nan, "vol_0": np.nan,
                "open_1": np.nan, "high_1": np.nan, "low_1": np.nan, "close_1": np.nan, "vol_1": np.nan,
                "open_2": np.nan, "high_2": np.nan, "low_2": np.nan, "close_2": np.nan, "vol_2": np.nan,
                "open_3": np.nan, "high_3": np.nan, "low_3": np.nan, "close_3": np.nan, "vol_3": np.nan,
                "day_amount": np.nan, "day_volume": np.nan,
            })
        row = {"n_bars": len(grp)}
        for i in range(4):
            if i < len(grp):
                row[f"open_{i}"] = grp.iloc[i]["open"]
                row[f"high_{i}"] = grp.iloc[i]["high"]
                row[f"low_{i}"] = grp.iloc[i]["low"]
                row[f"close_{i}"] = grp.iloc[i]["close"]
                row[f"vol_{i}"] = grp.iloc[i]["volume"]
            else:
                row[f"open_{i}"] = np.nan
                row[f"high_{i}"] = np.nan
                row[f"low_{i}"] = np.nan
                row[f"close_{i}"] = np.nan
                row[f"vol_{i}"] = np.nan
        row["day_amount"] = grp["amount"].sum()
        row["day_volume"] = grp["volume"].sum()
        return pd.Series(row)

    bars = grouped.apply(_agg_bars).reset_index()
    # trade_date is already date from dt.date above
    bars["trade_date"] = pd.to_datetime(bars["trade_date"])

    day_cols: dict[str, pd.Series] = {}

    # ── am_ret: 上午收益 = bar1 close / bar0 open - 1 ──
    day_cols["am_ret_raw"] = bars["close_1"] / bars["open_0"] - 1.0

    # ── pm_ret: 下午收益 = bar3 close / bar2 open - 1 ──
    day_cols["pm_ret_raw"] = bars["close_3"] / bars["open_2"] - 1.0

    # ── intra_vol_skew: 4根bar各自(close/open-1)的标准差 ──
    bar_rets = []
    for i in range(4):
        bar_rets.append(bars[f"close_{i}"] / bars[f"open_{i}"] - 1.0)
    bar_rets_df = pd.concat(bar_rets, axis=1)
    day_cols["intra_vol_skew_raw"] = bar_rets_df.std(axis=1)

    # ── close_auction_strength: 尾盘相对强度 ──
    # (bar3 close - bar3 open) / max(前3根bar的high-low range, 1e-8)
    prev_range = pd.concat([
        bars["high_0"] - bars["low_0"],
        bars["high_1"] - bars["low_1"],
        bars["high_2"] - bars["low_2"],
    ], axis=1).max(axis=1)
    prev_range = prev_range.replace(0, np.nan)
    day_cols["close_auction_str_raw"] = (bars["close_3"] - bars["open_3"]) / prev_range

    # ── volume_concentration: max(单bar量) / 总日量 ──
    bar_vols = []
    for i in range(4):
        bar_vols.append(bars[f"vol_{i}"])
    bar_vols_df = pd.concat(bar_vols, axis=1)
    day_cols["vol_concentration_raw"] = bar_vols_df.max(axis=1) / bars["day_volume"].replace(0, np.nan)

    # ── vwap_gap: (日线close - 日内VWAP) / 日内VWAP ──
    # VWAP = Σ(bar.close × bar.vol) / Σ(bar.vol)
    vwap_num = pd.Series(0.0, index=bars.index)
    vwap_den = pd.Series(0.0, index=bars.index)
    for i in range(4):
        vwap_num += bars[f"close_{i}"].fillna(0) * bars[f"vol_{i}"].fillna(0)
        vwap_den += bars[f"vol_{i}"].fillna(0)
    vwap = vwap_num / vwap_den.replace(0, np.nan)
    # day_close is bar3 close (last bar of the day)
    day_cols["vwap_gap_raw"] = (bars["close_3"] - vwap) / vwap

    # ── am_pm_divergence: am_ret - pm_ret ──
    day_cols["am_pm_div_raw"] = day_cols["am_ret_raw"] - day_cols["pm_ret_raw"]

    # 封装为 extra_data 格式
    for col_name, series in day_cols.items():
        fdf = bars[["code", "trade_date"]].copy()
        fdf["value"] = series
        fdf = fdf.dropna(subset=["value"])
        features[col_name] = fdf.rename(columns={"value": col_name})

    return features


# ============================================================
#  因子函数 — 从 df 中读取预聚合列
#  签名: (df: pd.DataFrame) -> pd.Series
# ============================================================

def am_ret(df: pd.DataFrame) -> pd.Series:
    """上午时段收益: 9:31→11:30 涨跌幅。"""
    if "am_ret_raw" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["am_ret_raw"]


def pm_ret(df: pd.DataFrame) -> pd.Series:
    """下午时段收益: 13:01→15:00 涨跌幅。"""
    if "pm_ret_raw" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["pm_ret_raw"]


def intra_vol_skew(df: pd.DataFrame) -> pd.Series:
    """日内波动偏度: 4根60min bar各自收益率的标准差。"""
    if "intra_vol_skew_raw" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["intra_vol_skew_raw"]


def close_auction_strength(df: pd.DataFrame) -> pd.Series:
    """尾盘强度: 最后一根bar涨跌 / 前三根bar的H-L范围。"""
    if "close_auction_str_raw" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["close_auction_str_raw"]


def volume_concentration(df: pd.DataFrame) -> pd.Series:
    """量集中度: 最大单bar量 / 全天总量。"""
    if "vol_concentration_raw" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["vol_concentration_raw"]


def vwap_gap(df: pd.DataFrame) -> pd.Series:
    """VWAP偏离: (收盘价 - 日内VWAP) / 日内VWAP。"""
    if "vwap_gap_raw" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["vwap_gap_raw"]


def am_pm_divergence(df: pd.DataFrame) -> pd.Series:
    """午间反转: am_ret - pm_ret，正=上午强下午弱，负=上午弱下午强。"""
    if "am_pm_div_raw" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["am_pm_div_raw"]


INTRADAY_MINUTE_FACTORS: dict = {
    "am_ret": am_ret,
    "pm_ret": pm_ret,
    "intra_vol_skew": intra_vol_skew,
    "close_auction_strength": close_auction_strength,
    "volume_concentration": volume_concentration,
    "vwap_gap": vwap_gap,
    "am_pm_divergence": am_pm_divergence,
}
