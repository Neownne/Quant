#!/usr/bin/env python
"""信号层面特征工程 —— 从信号 CSV 出发，为每条信号直接计算特征。

两阶段：
  1. 个股特征：只加载信号股票的日线数据（600 天回溯），向量化计算涨停因子
  2. 市场/行业特征：跨股票查询，每天聚合一次（mkt_lu_count, sector_lu_count, sector_rank_pct）

性能目标：< 5 分钟完成 31195 条信号的因子计算。

用法:
    python scripts/featurize_signals.py --signals data/signals/bt_signals_full.csv --out data/signals/bt_signals_features.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data

# ── 涨停阈值（板别感知）──
_LIMIT_MAP = {"688": 0.20, "8": 0.30, "4": 0.30, "300": 0.20, "301": 0.20}
_DEFAULT_LIMIT = 0.10


def _get_limit(c):
    for p, limit in _LIMIT_MAP.items():
        if str(c).startswith(p):
            return limit
    return _DEFAULT_LIMIT


def parse_args():
    p = argparse.ArgumentParser(description="信号层面特征工程")
    p.add_argument("--signals", default="data/signals/bt_signals_full.csv",
                   help="输入信号 CSV 路径")
    p.add_argument("--out", default="data/signals/bt_signals_features.csv",
                   help="输出特征 CSV 路径")
    p.add_argument("--lookback", type=int, default=60,
                   help="因子计算回溯天数（默认 60）")
    return p.parse_args()


def load_signals(csv_path: str) -> pd.DataFrame:
    """加载信号 CSV，标准化 code 和 date 列。"""
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.rename(columns={"date": "signal_date"})  # 避免与 trade_date 混淆
    else:
        raise ValueError("信号 CSV 缺少 'date' 列")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    logger.info(f"  信号: {len(df)} 条, {df['code'].nunique()} 只, {df['signal_date'].nunique()} 天")
    return df


def load_industry_map(engine) -> dict:
    """从 stock_basic 加载 code -> industry 映射。"""
    df = pd.read_sql(
        "SELECT code, industry FROM stock_basic WHERE is_st = FALSE",
        engine,
    )
    df["code"] = df["code"].astype(str).str.zfill(6)
    return dict(zip(df["code"], df["industry"]))


# ═══════════════════════════════════════════════════════════════════
# Phase 1: 个股特征（纯向量化，只对信号股票计算）
# ═══════════════════════════════════════════════════════════════════

def compute_individual_features(daily: pd.DataFrame) -> pd.DataFrame:
    """纯向量化计算所有个股涨停因子。

    入参 daily 需包含: code, trade_date, open, high, low, close, volume
    可选: turnover, market_cap

    返回所有行，新增 ~40 个因子列。
    """
    df = daily.sort_values(["code", "trade_date"]).copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 前收盘 & 收益率
    df["prev_close"] = df.groupby("code")["close"].shift(1)
    df["ret"] = (df["close"] - df["prev_close"]) / df["prev_close"]

    # 涨停标记
    df["is_lu"] = df.apply(
        lambda r: r["ret"] >= _get_limit(r["code"]) * 0.98 if pd.notna(r["ret"]) else False,
        axis=1,
    ).astype(int)

    # ── 涨停模式因子 ──
    # lu_streak
    def _streak(s):
        cnt = 0; res = []
        for v in s:
            cnt = cnt + 1 if v else 0
            res.append(cnt)
        return pd.Series(res, index=s.index)
    df["lu_streak"] = df.groupby("code")["is_lu"].transform(_streak).astype(float)

    # lu_count_5d, 20d, 60d
    for w in [5, 20, 60]:
        df[f"lu_count_{w}d"] = df.groupby("code")["is_lu"].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        )

    # lu_max_streak_20d, 60d
    for w in [20, 60]:
        def _max_streak(s, ww=w):
            res = []
            for i in range(len(s)):
                start = max(0, i - ww + 1)
                win = s.iloc[start:i+1]
                mx = cur = 0
                for v in win:
                    if v: cur += 1; mx = max(mx, cur)
                    else: cur = 0
                res.append(float(mx))
            return pd.Series(res, index=s.index)
        df[f"lu_max_streak_{w}d"] = df.groupby("code")["is_lu"].transform(_max_streak)

    # lu_days_since_last
    def _days_since(s):
        last = -999; res = []
        for i, v in enumerate(s):
            if v: last = i
            res.append(float(i - last) if last >= 0 else np.nan)
        return pd.Series(res, index=s.index)
    df["lu_days_since_last"] = df.groupby("code")["is_lu"].transform(_days_since)

    # lu_first_board
    df["lu_prev5_sum"] = df.groupby("code")["is_lu"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    df["lu_first_board"] = ((df["is_lu"] == 1) & (df["lu_prev5_sum"] == 0)).astype(float)

    # lu_is_second_board
    df["is_lu_lag1"] = df.groupby("code")["is_lu"].shift(1).fillna(0)
    df["is_lu_lag2"] = df.groupby("code")["is_lu"].shift(2).fillna(0)
    df["lu_is_second_board"] = ((df["is_lu"] == 1) & (df["is_lu_lag1"] == 1) &
                                 (df["is_lu_lag2"] == 0)).astype(float)

    # lu_freq_accel
    lu10 = df.groupby("code")["is_lu"].transform(lambda x: x.rolling(10, min_periods=1).sum())
    lu30 = df.groupby("code")["is_lu"].transform(lambda x: x.rolling(30, min_periods=1).sum())
    df["lu_freq_accel"] = lu10 / lu30.replace(0, np.nan) - 1.0

    # ── 封板质量因子（仅涨停日有效）──
    lu_mask = df["is_lu"] == 1
    df["lu_seal_quality"] = np.where(lu_mask, df["close"] / df["high"].replace(0, np.nan), np.nan)
    df["lu_vol_intensity"] = np.where(lu_mask,
        df["volume"] / df.groupby("code")["volume"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()), np.nan)
    df["lu_open_strength"] = np.where(lu_mask,
        (df["open"] - df["prev_close"]) / df["prev_close"].replace(0, np.nan), np.nan)
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["lu_body_ratio"] = np.where(lu_mask, (df["close"] - df["open"]) / hl_range, np.nan)
    df["lu_upper_shadow_ratio"] = np.where(lu_mask,
        (df["high"] - df[["open", "close"]].max(axis=1)) / hl_range, np.nan)
    df["lu_amplitude"] = np.where(lu_mask,
        (df["high"] - df["low"]) / df["prev_close"].replace(0, np.nan), np.nan)
    df["lu_intraday_reversal"] = np.where(lu_mask, (df["close"] - df["low"]) / hl_range, np.nan)

    avg20 = df.groupby("code")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    std20 = df.groupby("code")["volume"].transform(lambda x: x.rolling(20, min_periods=5).std())
    threshold = avg20 + 2 * std20
    df["lu_volume_climax"] = np.where(lu_mask, df["volume"] / threshold.replace(0, np.nan), np.nan)

    # ── 首板前蓄力因子 ──
    first_mask = df["lu_first_board"] == 1
    # pre_lu_vol_trend
    def _vol_trend(vol, is_lu):
        res = pd.Series(np.nan, index=vol.index)
        for i in range(5, len(vol)):
            if is_lu.iloc[i] and is_lu.iloc[i-5:i].sum() == 0:
                prev = vol.iloc[i-5:i]
                if len(prev) >= 3 and prev.mean() > 0:
                    x = np.arange(len(prev))
                    slope = np.polyfit(x, prev.values, 1)[0]
                    res.iloc[i] = slope / prev.mean()
        return res
    df["pre_lu_vol_trend"] = df.groupby("code").apply(
        lambda g: _vol_trend(g["volume"], g["is_lu"]), include_groups=False
    ).reset_index(level=0, drop=True)

    # pre_lu_ret_5d
    close5 = df.groupby("code")["close"].shift(5)
    prev_close_s = df.groupby("code")["close"].shift(1)
    df["pre_lu_ret_5d"] = np.where(
        first_mask, (prev_close_s - close5) / close5.replace(0, np.nan), np.nan)

    # pre_lu_turnover_cv
    if "turnover" in df.columns:
        to_cv = df.groupby("code")["turnover"].transform(
            lambda x: x.rolling(10).std() / x.rolling(10).mean().replace(0, np.nan)
        )
        df["pre_lu_turnover_cv"] = np.where(first_mask, to_cv, np.nan)

    # ── 连板确认因子 ──
    df["vol_lag1"] = df.groupby("code")["volume"].shift(1)
    df["lu_vol_contraction"] = np.where(
        (df["is_lu"] == 1) & (df["is_lu_lag1"] == 1),
        df["volume"] / df["vol_lag1"].replace(0, np.nan), np.nan
    )

    seal = np.where(pd.isna(df["lu_seal_quality"]), 0.9, df["lu_seal_quality"])
    volc = np.where(pd.isna(df["lu_vol_contraction"]), 1.0,
                    1.0 / df["lu_vol_contraction"].clip(0.5, 3.0))
    df["lu_streak_quality"] = df["lu_streak"] * seal * volc

    if "turnover" in df.columns:
        to_int = df["turnover"] / df.groupby("code")["turnover"].transform(
            lambda x: x.rolling(20, min_periods=5).mean())
        df["lu_turnover_intensity"] = np.where(lu_mask, to_int, np.nan)

    # ── 相对强度 ──
    df["ma20"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    pos = df["close"] / df["ma20"].replace(0, np.nan) - 1
    pos5 = (df.groupby("code")["close"].shift(5) /
            df.groupby("code")["ma20"].shift(5).replace(0, np.nan) - 1)
    df["lu_excess_return_5d"] = np.where(lu_mask, pos - pos5, np.nan)

    ret20 = df.groupby("code")["close"].transform(lambda x: x.pct_change(20))
    df["lu_relative_strength_20d"] = np.where(lu_mask, ret20, np.nan)

    # ── 板型分类 ──
    df["lu_is_yiziban"] = np.where(lu_mask,
        ((df["high"] - df["low"]).abs() < df["close"] * 0.001), 0.0).astype(float)
    df["lu_is_tziban"] = np.where(lu_mask,
        ((df["high"] - df["open"]).abs() < df["close"] * 0.01) &
        (df["close"] < df["high"] * 0.995), 0.0).astype(float)
    real_body = (df["close"] - df["open"]).abs()
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    df["lu_is_lanban"] = np.where(lu_mask,
        (upper_shadow > real_body) & (upper_shadow > 0), 0.0).astype(float)
    df["lu_is_strong_board"] = np.where(lu_mask,
        (real_body > hl_range * 0.6) & (upper_shadow < hl_range * 0.1), 0.0).astype(float)

    df["lu_board_strength"] = np.where(lu_mask,
        df["lu_seal_quality"].fillna(0.9) * (1 - df["lu_upper_shadow_ratio"].fillna(0.5)) +
        df["lu_is_strong_board"].fillna(0) * 0.2, np.nan)

    df["lu_board_type_change"] = np.where(lu_mask & (df["is_lu_lag1"] == 1),
        ((real_body - real_body.groupby(df["code"]).shift(1)).abs() / df["close"]), np.nan)

    # ── 资金流代理 ──
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = typical_price * df["volume"]
    mf_diff = typical_price.diff()
    pos_flow = (raw_mf.where(mf_diff > 0, 0)
                .groupby(df["code"])
                .transform(lambda x: x.rolling(14, min_periods=7).sum()))
    neg_flow = (raw_mf.where(mf_diff < 0, 0).abs()
                .groupby(df["code"])
                .transform(lambda x: x.rolling(14, min_periods=7).sum()))
    mf_ratio = pos_flow / neg_flow.replace(0, np.nan)
    df["mfi_14"] = 100 - 100 / (1 + mf_ratio)

    cl_hl_ratio = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range.replace(0, np.nan)
    cmf_raw = cl_hl_ratio * df["volume"]
    cmf_sum = cmf_raw.groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=5).sum())
    vol_sum = df["volume"].groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=5).sum())
    df["cmf_20"] = cmf_sum / vol_sum.replace(0, np.nan)

    fi_raw = df["close"].diff() * df["volume"]
    df["force_index"] = fi_raw.groupby(df["code"]).transform(
        lambda x: x.ewm(span=13, adjust=False).mean())

    hl_avg = (df["high"] + df["low"]) / 2
    hl_avg_prev = hl_avg.groupby(df["code"]).shift(1)
    box_ratio = df["volume"] / hl_range.replace(0, np.nan) / 1e8
    eom_raw = (hl_avg - hl_avg_prev) / box_ratio.replace(0, np.nan)
    df["eom_14"] = eom_raw.groupby(df["code"]).transform(
        lambda x: x.rolling(14, min_periods=7).mean())

    vpt_raw = df["volume"] * df["close"].pct_change().fillna(0)
    df["vpt_cum"] = vpt_raw.groupby(df["code"]).transform(lambda x: x.cumsum())
    df["vpt_divergence"] = df.groupby("code")["vpt_cum"].transform(
        lambda x: x.pct_change(5) - x.pct_change(20))

    high_vol = df["high"] * df["volume"]
    low_vol = df["low"] * df["volume"]
    df["money_flow_pressure"] = (high_vol - low_vol) / (high_vol + low_vol).replace(0, np.nan)

    # ── 波动率结构 ──
    ret = df["ret"]
    vol5 = ret.groupby(df["code"]).transform(lambda x: x.rolling(5, min_periods=3).std())
    vol20 = ret.groupby(df["code"]).transform(lambda x: x.rolling(20, min_periods=10).std())
    df["vol_ratio_ret_5_20"] = vol5 / vol20.replace(0, np.nan)

    up_ret = ret.clip(lower=0)
    down_ret = (-ret).clip(lower=0)
    df["downside_vol_20"] = down_ret.groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=10).std())
    df["upside_vol_20"] = up_ret.groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=10).std())
    df["vol_asymmetry"] = df["downside_vol_20"] / df["upside_vol_20"].replace(0, np.nan)

    df["vol_regime"] = vol20.groupby(df["code"]).transform(
        lambda x: x.rolling(60, min_periods=20).apply(
            lambda y: (y.iloc[-1] > y).mean() if len(y) > 5 else 0.5))

    df["ret_skew_20"] = ret.groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=10).skew())

    # ── 时序形态 ──
    ma5 = df.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    ma10 = df.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df["ma_convergence"] = ((ma5 / ma10.replace(0, np.nan) - 1).abs() +
                             (ma10 / df["ma20"].replace(0, np.nan) - 1).abs())
    df["ma_convergence"] = -df["ma_convergence"]

    hh20 = df.groupby("code")["high"].transform(lambda x: x.rolling(20, min_periods=10).max())
    ll20 = df.groupby("code")["low"].transform(lambda x: x.rolling(20, min_periods=10).min())
    df["price_compactness"] = -(hh20 / ll20.replace(0, np.nan) - 1)

    df["volume_breakout"] = np.where(
        (df["volume"] > avg20 * 2) & (ret > 0.02), 1.0, 0.0)

    df["volume_contraction_signal"] = np.where(
        (df["volume"] < avg20 * 0.5) & (hl_range / df["close"] < 0.03), 1.0, 0.0)

    def _count_low_vol(vol, avg):
        cnt = 0; res = []
        for i in range(len(vol)):
            if vol.iloc[i] < avg.iloc[i] * 0.7: cnt += 1
            else: cnt = 0
            res.append(float(cnt))
        return pd.Series(res, index=vol.index)
    df["low_vol_streak"] = df.groupby("code").apply(
        lambda g: _count_low_vol(g["volume"], g["volume"].rolling(20, min_periods=5).mean()),
        include_groups=False
    ).reset_index(level=0, drop=True)

    # ── 通用对标因子 ──
    delta = df.groupby("code")["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df["code"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    avg_loss = loss.groupby(df["code"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    df["mom_20"] = df.groupby("code")["close"].transform(lambda x: x.pct_change(20))
    df["rev_5"] = -df.groupby("code")["close"].transform(lambda x: x.pct_change(5))

    v5 = df.groupby("code")["volume"].transform(lambda x: x.rolling(5).mean())
    v20 = df.groupby("code")["volume"].transform(lambda x: x.rolling(20).mean())
    df["vol_ratio_5_20"] = v5 / v20.replace(0, np.nan)

    if "turnover" in df.columns:
        df["turnover_5"] = df.groupby("code")["turnover"].transform(
            lambda x: x.rolling(5).mean())

    ma20c = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    std20c = df.groupby("code")["close"].transform(lambda x: x.rolling(20).std())
    df["bb_position"] = (df["close"] - ma20c) / (2 * std20c.replace(0, np.nan))

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["prev_close"]).abs(),
        (df["low"] - df["prev_close"]).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.groupby(df["code"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())

    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / hl_range

    if "market_cap" in df.columns:
        df["log_mcap"] = np.log(df["market_cap"].clip(lower=1e6))

    # 清理临时列
    drop_cols = ["prev_close", "is_lu", "lu_prev5_sum", "is_lu_lag1", "is_lu_lag2",
                  "vol_lag1", "ma20", "vpt_cum", "typical_price", "raw_mf",
                  "mf_diff", "pos_flow", "neg_flow", "mf_ratio", "cl_hl_ratio",
                  "cmf_raw", "fi_raw", "hl_avg", "hl_avg_prev", "box_ratio", "eom_raw",
                  "vpt_raw", "high_vol", "low_vol", "ma5", "ma10",
                  "hl_range", "real_body", "upper_shadow", "avg20", "std20", "threshold",
                  "close5", "vol5", "vol20", "hh20", "ll20", "tr"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 2: 市场/行业特征（跨股票聚合）
# ═══════════════════════════════════════════════════════════════════

def compute_market_features(engine, signal_dates: pd.DatetimeIndex,
                            industry_map: dict) -> pd.DataFrame:
    """计算市场/行业特征：每日聚合所有股票的涨跌停情况。

    返回 DataFrame，列为:
      code, trade_date, industry, mkt_lu_count, mkt_lu_pct, mkt_stock_count,
      sector_lu_count, sector_lu_pct, sector_stock_count,
      sector_ret_mean, sector_ret_std, sector_rank_pct
    """
    t0 = time.time()
    # 向前扩展 10 天以获取 prev_close 和收益率计算所需的缓冲
    full_start = (signal_dates.min() - timedelta(days=10)).strftime("%Y-%m-%d")
    full_end = signal_dates.max().strftime("%Y-%m-%d")

    logger.info(f"  加载全市场日线 (close) {full_start} ~ {full_end} ...")
    from sqlalchemy import text as sa_text
    daily_all = pd.read_sql(
        sa_text("SELECT code, trade_date, close "
                "FROM stock_daily "
                "WHERE trade_date BETWEEN :start AND :end "
                "ORDER BY code, trade_date"),
        engine,
        params={"start": full_start, "end": full_end},
    )
    daily_all["code"] = daily_all["code"].astype(str).str.zfill(6)
    daily_all["trade_date"] = pd.to_datetime(daily_all["trade_date"])
    logger.info(f"  全市场日线: {len(daily_all)} 行 ({time.time() - t0:.1f}s)")

    # 计算收益率
    daily_all = daily_all.sort_values(["code", "trade_date"])
    daily_all["prev_close"] = daily_all.groupby("code")["close"].shift(1)
    daily_all["ret"] = ((daily_all["close"] - daily_all["prev_close"]) /
                         daily_all["prev_close"])

    # 涨停 / 跌停标记（板别感知）
    daily_all["is_lu"] = daily_all.apply(
        lambda r: 1 if (pd.notna(r["ret"]) and
                        r["ret"] >= _get_limit(r["code"]) * 0.98) else 0,
        axis=1,
    )
    daily_all["is_ld"] = daily_all.apply(
        lambda r: 1 if (pd.notna(r["ret"]) and
                        r["ret"] <= -_get_limit(r["code"]) * 0.98) else 0,
        axis=1,
    )
    daily_all = daily_all.dropna(subset=["ret"])

    # 映射行业
    daily_all["industry"] = daily_all["code"].map(industry_map).fillna("其他")

    # 只保留信号日的行
    signal_date_set = set(signal_dates)
    daily_sig = daily_all[daily_all["trade_date"].isin(signal_date_set)].copy()
    logger.info(f"  信号日全市场数据: {len(daily_sig)} 行 ({time.time() - t0:.1f}s)")

    # ── 按日聚合：全市场特征 ──
    mkt_agg = daily_sig.groupby("trade_date").agg(
        mkt_lu_count=("is_lu", "sum"),
        mkt_stock_count=("code", "count"),
        mkt_ld_count=("is_ld", "sum"),
    ).reset_index()
    mkt_agg["mkt_lu_pct"] = mkt_agg["mkt_lu_count"] / mkt_agg["mkt_stock_count"].replace(0, np.nan)
    mkt_agg["mkt_ld_pct"] = mkt_agg["mkt_ld_count"] / mkt_agg["mkt_stock_count"].replace(0, np.nan)

    # ── 按日+行业聚合：行业特征 ──
    sector_agg = daily_sig.groupby(["trade_date", "industry"]).agg(
        sector_lu_count=("is_lu", "sum"),
        sector_stock_count=("code", "count"),
        sector_ret_mean=("ret", "mean"),
        sector_ret_std=("ret", "std"),
    ).reset_index()
    sector_agg["sector_lu_pct"] = (sector_agg["sector_lu_count"] /
                                    sector_agg["sector_stock_count"].replace(0, np.nan))

    # ── 个股在行业内的收益排名 ──
    daily_sig["sector_rank_pct"] = daily_sig.groupby(
        ["trade_date", "industry"]
    )["ret"].transform(lambda x: x.rank(pct=True, na_option="bottom"))

    # ── 组装：股票级行 + 市场级 + 行业级 feature ──
    result = daily_sig[["code", "trade_date", "industry", "sector_rank_pct"]].copy()
    # 市场特征：按 trade_date merge
    result = result.merge(mkt_agg, on="trade_date", how="left")
    # 行业特征：按 (trade_date, industry) merge
    result = result.merge(sector_agg, on=["trade_date", "industry"], how="left")

    logger.info(f"  市场特征完成 ({time.time() - t0:.1f}s)")
    return result


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()

    # ── 1. 加载信号 ──
    logger.info("=" * 60)
    logger.info("Step 1: 加载信号")
    signals = load_signals(args.signals)
    signal_codes = sorted(signals["code"].unique().tolist())
    n_signals = len(signals)

    # ── 2. 加载日线数据（信号股票 + 回溯窗口）──
    logger.info("=" * 60)
    logger.info("Step 2: 加载信号股票日线数据")

    signal_dates = signals["signal_date"]
    pre_start = (signal_dates.min() - timedelta(days=args.lookback + 30)).strftime("%Y-%m-%d")
    end_date = signal_dates.max().strftime("%Y-%m-%d")

    engine = get_engine()
    daily = load_daily_data(engine, signal_codes, pre_start, end_date,
                            cols=["open", "high", "low", "close", "volume", "amount", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    logger.info(f"  日线: {len(daily)} 行, {daily['code'].nunique()} 只")

    # ── 3. 加载市值 ──
    logger.info("=" * 60)
    logger.info("Step 3: 加载市值数据")
    mcap = load_mcap_data(engine, signal_codes, pre_start, end_date, use_proxy=True)
    if not mcap.empty:
        mcap["code"] = mcap["code"].astype(str).str.zfill(6)
        mcap["trade_date"] = pd.to_datetime(mcap["trade_date"])
        daily = daily.merge(mcap[["code", "trade_date", "market_cap"]],
                            on=["code", "trade_date"], how="left")
        logger.info(f"  市值: {len(mcap)} 行")
    else:
        daily["market_cap"] = np.nan
        logger.warning("  无市值数据")

    # ── 4. 计算个股特征 ──
    logger.info("=" * 60)
    logger.info("Step 4: 计算个股特征")
    t_feat = time.time()
    daily = compute_individual_features(daily)
    logger.info(f"  个股特征完成 ({time.time() - t_feat:.1f}s), {len(daily.columns)} 列")

    # ── 5. 提取信号日行 ──
    logger.info("=" * 60)
    logger.info("Step 5: 提取信号日因子值")
    daily["key"] = (daily["trade_date"].astype(str) + "_" + daily["code"])
    signals["key"] = (signals["signal_date"].astype(str) + "_" + signals["code"])

    # 识别因子列：排除 OHLCV 原始列、非数值列、以及 categorical/metadata 列
    oh_columns = ["code", "trade_date", "key", "open", "high", "low", "close",
                   "volume", "amount", "turnover", "market_cap", "ret", "industry"]
    factor_cols = [c for c in daily.columns
                   if c not in oh_columns
                   and not c.startswith("ret_fwd")]
    # 只保留数值型因子
    factor_cols = [c for c in factor_cols if pd.api.types.is_numeric_dtype(daily[c])]

    # 同样排除 market feature merge 中可能混入的非数值列
    non_factor_meta = ["industry", "key"]

    sig_features = signals[["key", "signal_date", "code"]].merge(
        daily[["key"] + factor_cols], on="key", how="left"
    )
    logger.info(f"  匹配: {sig_features['key'].notna().sum()} / {n_signals} 条信号有因子值")

    # ── 6. 加载行业映射 + 计算市场特征 ──
    logger.info("=" * 60)
    logger.info("Step 6: 计算市场/行业特征")
    industry_map = load_industry_map(engine)
    logger.info(f"  行业映射: {len(industry_map)} 只")

    mkt_feat = compute_market_features(engine, signal_dates, industry_map)

    # ── 7. 合并市场特征到信号 ──
    logger.info("=" * 60)
    logger.info("Step 7: 合并市场特征")
    # mkt_feat 列: code, trade_date, industry, sector_rank_pct,
    #              mkt_lu_count, mkt_lu_pct, mkt_stock_count, mkt_ld_count, mkt_ld_pct,
    #              sector_lu_count, sector_lu_pct, sector_stock_count,
    #              sector_ret_mean, sector_ret_std
    mkt_cols = [c for c in mkt_feat.columns
                if c not in ("code", "trade_date")]
    # 确保 (code, trade_date) 唯一
    mkt_for_merge = mkt_feat[["code", "trade_date"] + mkt_cols].drop_duplicates(
        subset=["code", "trade_date"]
    )
    sig_features = sig_features.merge(
        mkt_for_merge,
        left_on=["code", "signal_date"], right_on=["code", "trade_date"],
        how="left"
    ).drop(columns=["trade_date"], errors="ignore")

    # 补充行业信息（市场特征已带 industry，但信号可能未匹配到，用 map 兜底）
    if "industry" not in sig_features.columns:
        sig_features["industry"] = sig_features["code"].map(industry_map).fillna("其他")
    else:
        sig_features["industry"] = sig_features["industry"].fillna(
            sig_features["code"].map(industry_map)
        ).fillna("其他")

    # ── 8. 合并回原始信号列 ──
    logger.info("=" * 60)
    logger.info("Step 8: 组装最终输出")
    # 使用原始 signals 中的 info 列
    info_cols = [c for c in signals.columns
                 if c in ["rank", "name", "score", "close", "is_limit_up", "is_limit_down"]]
    final = sig_features.merge(
        signals[["key"] + info_cols], on="key", how="left"
    )
    # 重命名 signal_date -> date 保持与输入一致
    final = final.rename(columns={"signal_date": "date"})

    # 排列列顺序：原始信号列在前，因子列在后
    output_info_cols = ["date", "code", "rank", "name", "score", "close",
                         "is_limit_up", "is_limit_down"]
    output_info_cols = [c for c in output_info_cols if c in final.columns]
    factor_output_cols = [c for c in final.columns
                          if c not in output_info_cols + ["key", "industry"]]
    # 再次确保因子列都是数值型
    factor_output_cols = [c for c in factor_output_cols
                          if pd.api.types.is_numeric_dtype(final[c])]
    final = final[output_info_cols + factor_output_cols]

    # ── 9. 保存 ──
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    final.to_csv(args.out, index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    n_factor_cols = len(factor_output_cols)

    # ── 10. 验证报告 ──
    logger.info("=" * 60)
    logger.info("验证报告")
    logger.info(f"  耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  输入: {n_signals} 条信号")
    logger.info(f"  输出: {len(final)} 行, {n_factor_cols} 个因子列")

    # 行数检查
    if len(final) == n_signals:
        logger.info("  PASS: 输出行数 = 输入行数")
    else:
        logger.warning(f"  WARN: 输出行数 ({len(final)}) != 输入行数 ({n_signals})")

    # NaN 检查
    nan_report = {}
    for col in factor_output_cols:
        nan_pct = final[col].isna().mean()
        nan_report[col] = nan_pct
    high_nan_cols = [k for k, v in nan_report.items() if v > 0.8]
    if high_nan_cols:
        logger.warning(f"  NaN > 80%: {len(high_nan_cols)} 列")
        for c in sorted(high_nan_cols, key=lambda x: nan_report[x], reverse=True)[:10]:
            logger.warning(f"    {c}: {nan_report[c]:.1%}")
    else:
        logger.info(f"  PASS: 所有因子列 NaN 占比 ≤ 80%")
    avg_nan = np.mean(list(nan_report.values()))
    logger.info(f"  平均 NaN 占比: {avg_nan:.1%}")

    # 类型检查
    non_numeric = []
    for col in factor_output_cols:
        if not pd.api.types.is_numeric_dtype(final[col]):
            non_numeric.append(col)
    if non_numeric:
        logger.warning(f"  WARN: {len(non_numeric)} 个非数值型因子列: {non_numeric[:10]}")
    else:
        logger.info("  PASS: 所有因子列均为数值型")

    logger.info(f"  输出文件: {args.out}")
    logger.info("=" * 60)

    engine.dispose()


if __name__ == "__main__":
    main()
