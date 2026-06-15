#!/usr/bin/env python
"""因子 IC 快速验证 —— 聚焦涨停因子 + 对标通用因子，纯向量化计算。

用法:
    python scripts/validate_factors.py --start 2025-01-01
    python scripts/validate_factors.py --start 2020-01-01 --end 2026-06-14
"""

from __future__ import annotations

import argparse, os, sys, time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data

OUT_DIR = "data/factor_ic"

# ── 涨停阈值（板别感知）──
_LIMIT_MAP = {"688": 0.20, "8": 0.30, "4": 0.30, "300": 0.20, "301": 0.20}
_DEFAULT_LIMIT = 0.10

def _get_limit(c):
    for p, l in _LIMIT_MAP.items():
        if str(c).startswith(p):
            return l
    return _DEFAULT_LIMIT


def parse_args():
    p = argparse.ArgumentParser(description="因子 IC 快速验证")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default=None)
    return p.parse_args()


def load_signal_data(args):
    """加载信号 CSV 或从 DB 读取。"""
    csv_path = "data/signals/bt_signals_latest.csv"
    if os.path.exists(csv_path):
        sig = pd.read_csv(csv_path)
        sig = sig.rename(columns={"date": "trade_date"}) if "date" in sig.columns else sig
    else:
        logger.error(f"信号文件不存在: {csv_path}，请先运行 gen_signals.py")
        sys.exit(1)

    sig["trade_date"] = pd.to_datetime(sig["trade_date"])
    sig["code"] = sig["code"].astype(str).str.zfill(6)
    sig = sig[(sig["trade_date"] >= pd.Timestamp(args.start))]
    if args.end:
        sig = sig[(sig["trade_date"] <= pd.Timestamp(args.end))]
    return sig


def compute_factors_fast(daily: pd.DataFrame) -> pd.DataFrame:
    """纯向量化计算所有涨停因子 + 对标通用因子。

    不依赖 FactorEngine，直接对全量 daily DataFrame 做 groupby transform。
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

    # ════ 涨停模式因子 ════
    # lu_streak: 当前连板数
    def _streak(s):
        cnt = 0
        res = []
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
                    if v:
                        cur += 1
                        mx = max(mx, cur)
                    else:
                        cur = 0
                res.append(float(mx))
            return pd.Series(res, index=s.index)
        df[f"lu_max_streak_{w}d"] = df.groupby("code")["is_lu"].transform(_max_streak)

    # lu_days_since_last
    def _days_since(s):
        last = -999
        res = []
        for i, v in enumerate(s):
            if v:
                last = i
            res.append(float(i - last) if last >= 0 else np.nan)
        return pd.Series(res, index=s.index)
    df["lu_days_since_last"] = df.groupby("code")["is_lu"].transform(_days_since)

    # lu_first_board: 当日涨停且前5日无涨停
    df["lu_prev5_sum"] = df.groupby("code")["is_lu"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    df["lu_first_board"] = ((df["is_lu"] == 1) & (df["lu_prev5_sum"] == 0)).astype(float)

    # lu_is_second_board: 昨日涨停且今日涨停且前日非涨停
    df["is_lu_lag1"] = df.groupby("code")["is_lu"].shift(1).fillna(0)
    df["is_lu_lag2"] = df.groupby("code")["is_lu"].shift(2).fillna(0)
    df["lu_is_second_board"] = ((df["is_lu"] == 1) & (df["is_lu_lag1"] == 1) &
                                 (df["is_lu_lag2"] == 0)).astype(float)

    # lu_freq_accel: 近10日频率 / 近30日频率 - 1
    lu10 = df.groupby("code")["is_lu"].transform(lambda x: x.rolling(10, min_periods=1).sum())
    lu30 = df.groupby("code")["is_lu"].transform(lambda x: x.rolling(30, min_periods=1).sum())
    df["lu_freq_accel"] = lu10 / lu30.replace(0, np.nan) - 1.0

    # ════ 封板质量因子（仅涨停日有效）═══
    lu_mask = df["is_lu"] == 1
    df["lu_seal_quality"] = np.where(lu_mask, df["close"] / df["high"].replace(0, np.nan), np.nan)
    df["lu_vol_intensity"] = np.where(lu_mask,
        df["volume"] / df.groupby("code")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean()),
        np.nan)
    df["lu_open_strength"] = np.where(lu_mask,
        (df["open"] - df["prev_close"]) / df["prev_close"].replace(0, np.nan), np.nan)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["lu_body_ratio"] = np.where(lu_mask, (df["close"] - df["open"]) / hl, np.nan)
    df["lu_upper_shadow_ratio"] = np.where(lu_mask, (df["high"] - df[["open", "close"]].max(axis=1)) / hl, np.nan)
    df["lu_amplitude"] = np.where(lu_mask, (df["high"] - df["low"]) / df["prev_close"].replace(0, np.nan), np.nan)
    df["lu_intraday_reversal"] = np.where(lu_mask, (df["close"] - df["low"]) / hl, np.nan)

    avg20 = df.groupby("code")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    std20 = df.groupby("code")["volume"].transform(lambda x: x.rolling(20, min_periods=5).std())
    threshold = avg20 + 2 * std20
    df["lu_volume_climax"] = np.where(lu_mask, df["volume"] / threshold.replace(0, np.nan), np.nan)

    # ════ 首板前蓄力因子 ════
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
    prev_close = df.groupby("code")["close"].shift(1)
    df["pre_lu_ret_5d"] = np.where(first_mask, (prev_close - close5) / close5.replace(0, np.nan), np.nan)

    # pre_lu_turnover_cv
    if "turnover" in df.columns:
        to_cv = df.groupby("code")["turnover"].transform(
            lambda x: x.rolling(10).std() / x.rolling(10).mean().replace(0, np.nan)
        )
        df["pre_lu_turnover_cv"] = np.where(first_mask, to_cv, np.nan)

    # ════ 连板确认因子 ════
    # lu_vol_contraction
    df["vol_lag1"] = df.groupby("code")["volume"].shift(1)
    df["lu_vol_contraction"] = np.where(
        (df["is_lu"] == 1) & (df["is_lu_lag1"] == 1),
        df["volume"] / df["vol_lag1"].replace(0, np.nan), np.nan
    )

    # ❌ lu_next_day_gap 已删除 — shift(-1)=明天开盘价，含未来函数

    # lu_streak_quality
    seal = np.where(pd.isna(df["lu_seal_quality"]), 0.9, df["lu_seal_quality"])
    volc = np.where(pd.isna(df["lu_vol_contraction"]), 1.0, 1.0 / df["lu_vol_contraction"].clip(0.5, 3.0))
    df["lu_streak_quality"] = df["lu_streak"] * seal * volc

    # lu_turnover_intensity
    if "turnover" in df.columns:
        to_int = df["turnover"] / df.groupby("code")["turnover"].transform(
            lambda x: x.rolling(20, min_periods=5).mean())
        df["lu_turnover_intensity"] = np.where(lu_mask, to_int, np.nan)

    # ════ 相对强度 ════
    # lu_excess_return_5d: 近5日均线位置变化
    df["ma20"] = df.groupby("code")["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    pos = df["close"] / df["ma20"].replace(0, np.nan) - 1
    pos5 = df.groupby("code")["close"].shift(5) / df.groupby("code")["ma20"].shift(5).replace(0, np.nan) - 1
    df["lu_excess_return_5d"] = np.where(lu_mask, pos - pos5, np.nan)

    # lu_relative_strength_20d
    ret20 = df.groupby("code")["close"].transform(lambda x: x.pct_change(20))
    df["lu_relative_strength_20d"] = np.where(lu_mask, ret20, np.nan)

    # ═══════════════════════════════════════════════════════════════
    # Category 7: 板型分类 (Board Type Classification)
    # ═══════════════════════════════════════════════════════════════
    # 一字板: open≈high≈close (极强封板，几乎买不到)
    df["lu_is_yiziban"] = np.where(lu_mask,
        ((df["high"] - df["low"]).abs() < df["close"] * 0.001), 0.0).astype(float)

    # T字板: open≈high 但 close < high (盘中曾打开)
    upper_body = df["high"] - df[["open", "close"]].max(axis=1)
    df["lu_is_tziban"] = np.where(lu_mask,
        ((df["high"] - df["open"]).abs() < df["close"] * 0.01) & (df["close"] < df["high"] * 0.995), 0.0).astype(float)

    # 烂板: 上影线 > 实体 (炸板痕迹重)
    real_body = (df["close"] - df["open"]).abs()
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    df["lu_is_lanban"] = np.where(lu_mask,
        (upper_shadow > real_body) & (upper_shadow > 0), 0.0).astype(float)

    # 强实体板: 实体 > 振幅*0.6 且上影线 < 振幅*0.1
    hl_range = df["high"] - df["low"]
    df["lu_is_strong_board"] = np.where(lu_mask,
        (real_body > hl_range * 0.6) & (upper_shadow < hl_range * 0.1), 0.0).astype(float)

    # 板强度综合: 封板质量 * (1-上影线占比) + 实体板加分
    df["lu_board_strength"] = np.where(lu_mask,
        df["lu_seal_quality"].fillna(0.9) * (1 - df["lu_upper_shadow_ratio"].fillna(0.5)) +
        df["lu_is_strong_board"].fillna(0) * 0.2, np.nan)

    # 板型变化: 今日板型 vs 昨日板型是否一致
    df["lu_board_type_change"] = np.where(lu_mask & (df["is_lu_lag1"] == 1),
        ((real_body - real_body.groupby(df["code"]).shift(1)).abs() / df["close"]), np.nan)

    # ═══════════════════════════════════════════════════════════════
    # Category 8: 资金流代理因子 (Fund Flow Proxies)
    # ═══════════════════════════════════════════════════════════════
    # Money Flow Index (MFI): 价格×成交量的RSI
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = typical_price * df["volume"]
    mf_diff = typical_price.diff()
    pos_flow = raw_mf.where(mf_diff > 0, 0).groupby(df["code"]).transform(
        lambda x: x.rolling(14, min_periods=7).sum())
    neg_flow = raw_mf.where(mf_diff < 0, 0).abs().groupby(df["code"]).transform(
        lambda x: x.rolling(14, min_periods=7).sum())
    mf_ratio = pos_flow / neg_flow.replace(0, np.nan)
    df["mfi_14"] = 100 - 100 / (1 + mf_ratio)

    # Chaikin Money Flow (CMF): ((close-low)-(high-close))/(high-low) * vol 的20日均值
    cl_hl_ratio = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range.replace(0, np.nan)
    cmf_raw = cl_hl_ratio * df["volume"]
    df["cmf_20"] = cmf_raw.groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=5).sum() / x.rolling(20, min_periods=5).sum() * 1000
    ) / 1000  # normalize

    # Force Index: close_diff * volume 的指数平滑
    fi_raw = df["close"].diff() * df["volume"]
    df["force_index"] = fi_raw.groupby(df["code"]).transform(
        lambda x: x.ewm(span=13, adjust=False).mean())

    # Ease of Movement (EoM): ((high+low)/2 - prev_high_low_avg) / (volume / (high-low))
    hl_avg = (df["high"] + df["low"]) / 2
    hl_avg_prev = hl_avg.groupby(df["code"]).shift(1)
    box_ratio = df["volume"] / hl_range.replace(0, np.nan) / 1e8  # scale
    eom_raw = (hl_avg - hl_avg_prev) / box_ratio.replace(0, np.nan)
    df["eom_14"] = eom_raw.groupby(df["code"]).transform(
        lambda x: x.rolling(14, min_periods=7).mean())

    # Volume-Price Trend divergence: VPT变化 vs 价格变化的方向一致性
    vpt_raw = df["volume"] * df["close"].pct_change().fillna(0)
    df["vpt_cum"] = vpt_raw.groupby(df["code"]).transform(lambda x: x.cumsum())
    df["vpt_divergence"] = df.groupby("code")["vpt_cum"].transform(
        lambda x: x.pct_change(5) - x.pct_change(20))

    # 大单资金流向代理: (high*vol) vs (low*vol) 的比例
    high_vol = df["high"] * df["volume"]
    low_vol = df["low"] * df["volume"]
    df["money_flow_pressure"] = (high_vol - low_vol) / (high_vol + low_vol).replace(0, np.nan)

    # ═══════════════════════════════════════════════════════════════
    # Category 9: 波动率结构 (Volatility Structure)
    # ═══════════════════════════════════════════════════════════════
    ret = df["ret"]
    # 波动率比率: 短期vol/长期vol — 膨胀vs收缩
    vol5 = ret.groupby(df["code"]).transform(lambda x: x.rolling(5, min_periods=3).std())
    vol20 = ret.groupby(df["code"]).transform(lambda x: x.rolling(20, min_periods=10).std())
    df["vol_ratio_5_20"] = vol5 / vol20.replace(0, np.nan)

    # 下行波动率 / 上行波动率
    up_ret = ret.clip(lower=0)
    down_ret = (-ret).clip(lower=0)
    df["downside_vol_20"] = down_ret.groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=10).std())
    df["upside_vol_20"] = up_ret.groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=10).std())
    df["vol_asymmetry"] = df["downside_vol_20"] / df["upside_vol_20"].replace(0, np.nan)

    # 波动率状态: 当前vol在历史上的分位 (20日vol的60日分位)
    df["vol_regime"] = vol20.groupby(df["code"]).transform(
        lambda x: x.rolling(60, min_periods=20).apply(
            lambda y: (y.iloc[-1] > y).mean() if len(y) > 5 else 0.5))

    # 收益偏度（20日）
    df["ret_skew_20"] = ret.groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=10).skew())

    # ═══════════════════════════════════════════════════════════════
    # Category 10: 时序形态因子 (Time-series Patterns)
    # ═══════════════════════════════════════════════════════════════
    # 均线收敛度: MA5/MA10/MA20 之间的相对距离
    ma5 = df.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    ma10 = df.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=5).mean())
    # ma20 already computed above
    df["ma_convergence"] = (ma5 / ma10.replace(0, np.nan) - 1).abs() + \
                           (ma10 / df["ma20"].replace(0, np.nan) - 1).abs()
    df["ma_convergence"] = -df["ma_convergence"]  # 取负: 越收敛值越大

    # 价格紧凑度: (N日最高/N日最低 - 1) 越小越紧凑
    hh20 = df.groupby("code")["high"].transform(lambda x: x.rolling(20, min_periods=10).max())
    ll20 = df.groupby("code")["low"].transform(lambda x: x.rolling(20, min_periods=10).min())
    df["price_compactness"] = -(hh20 / ll20.replace(0, np.nan) - 1)

    # 放量突破: volume > 2*avg_vol AND 今日涨幅 > 2%
    df["volume_breakout"] = np.where(
        (df["volume"] > avg20 * 2) & (ret > 0.02), 1.0, 0.0)

    # 缩量整理: volume < 0.5*avg_vol 且振幅<3%
    df["volume_contraction_signal"] = np.where(
        (df["volume"] < avg20 * 0.5) & (hl_range / df["close"] < 0.03), 1.0, 0.0)

    # 连续缩量天数
    def _count_low_vol(vol, avg):
        cnt = 0
        res = []
        for i in range(len(vol)):
            if vol.iloc[i] < avg.iloc[i] * 0.7:
                cnt += 1
            else:
                cnt = 0
            res.append(float(cnt))
        return pd.Series(res, index=vol.index)
    df["low_vol_streak"] = df.groupby("code").apply(
        lambda g: _count_low_vol(g["volume"], g["volume"].rolling(20).mean()),
        include_groups=False
    ).reset_index(level=0, drop=True)

    # ════ 对标通用因子 (计算关键的代表性因子) ════
    # rsi_14
    delta = df.groupby("code")["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df["code"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    avg_loss = loss.groupby(df["code"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # mom_20
    df["mom_20"] = df.groupby("code")["close"].transform(lambda x: x.pct_change(20))

    # rev_5
    df["rev_5"] = -df.groupby("code")["close"].transform(lambda x: x.pct_change(5))

    # vol_ratio_5_20
    v5 = df.groupby("code")["volume"].transform(lambda x: x.rolling(5).mean())
    v20 = df.groupby("code")["volume"].transform(lambda x: x.rolling(20).mean())
    df["vol_ratio_5_20"] = v5 / v20.replace(0, np.nan)

    # turnover_5
    if "turnover" in df.columns:
        df["turnover_5"] = df.groupby("code")["turnover"].transform(lambda x: x.rolling(5).mean())

    # bb_position
    ma20c = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    std20c = df.groupby("code")["close"].transform(lambda x: x.rolling(20).std())
    df["bb_position"] = (df["close"] - ma20c) / (2 * std20c.replace(0, np.nan))

    # atr_14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["prev_close"]).abs(),
        (df["low"] - df["prev_close"]).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.groupby(df["code"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())

    # upper_shadow: 上影线比例（越小越好，涨停场景即封板强度）
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / hl

    # log_mcap (市值)
    if "market_cap" in df.columns:
        df["log_mcap"] = np.log(df["market_cap"].clip(lower=1e6))

    # 清理临时列
    drop_cols = ["prev_close", "is_lu", "lu_prev5_sum", "is_lu_lag1", "is_lu_lag2",
                 "vol_lag1", "ma20", "vpt_cum", "typical_price", "raw_mf",
                 "mf_diff", "pos_flow", "neg_flow", "mf_ratio", "cl_hl_ratio",
                 "cmf_raw", "fi_raw", "hl_avg", "hl_avg_prev", "box_ratio", "eom_raw",
                 "vpt_raw", "high_vol", "low_vol", "ma5", "ma10"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df


def compute_cross_sectional_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算截面排名因子 + 板块共振因子（需全市场同日数据）。

    必须在 compute_factors_fast 之后调用，因为需要 peer 比较。
    """
    df = df.copy()

    # ── 截面排名因子（在每个 trade_date 内排名取百分位）──
    rank_factors = [
        "lu_streak", "lu_vol_intensity", "lu_seal_quality", "lu_amplitude",
        "lu_turnover_intensity", "lu_streak_quality", "lu_board_strength",
        "volume", "turnover", "close",
    ]
    for fc in rank_factors:
        if fc in df.columns:
            df[f"rank_{fc}"] = df.groupby("trade_date")[fc].transform(
                lambda x: x.rank(pct=True, na_option="bottom"))

    # ── 板块共振因子（需 industry 列）──
    if "industry" in df.columns:
        # 同行业涨停家数
        df["sector_lu_n"] = df.groupby(["trade_date", "industry"])["is_lu"].transform("sum") \
            if "is_lu" in df.columns else np.nan

        # 同行业涨停占比
        df["sector_n"] = df.groupby(["trade_date", "industry"])["code"].transform("count")
        df["sector_lu_pct"] = df["sector_lu_n"] / df["sector_n"].replace(0, np.nan)

        # 行业平均收益
        df["sector_ret_mean"] = df.groupby(["trade_date", "industry"])["ret"].transform("mean")

        # 个股超额行业收益
        df["sector_excess_ret"] = df["ret"] - df["sector_ret_mean"]

        # 个股在行业内的收益排名
        df["sector_rank_pct"] = df.groupby(["trade_date", "industry"])["ret"].transform(
            lambda x: x.rank(pct=True, na_option="bottom"))

        # 是否行业龙头（收益最高 + 最早涨停）
        ret_max = df.groupby(["trade_date", "industry"])["ret"].transform("max")
        df["is_sector_leader"] = ((df["ret"] == ret_max) & (df["ret"] > 0.02)).astype(float)

        # 行业涨停家数变化（vs 5日前）
        df["sector_lu_n_5d_ago"] = df.groupby(["code"])["sector_lu_n"].shift(5)
        df["sector_lu_momentum"] = df["sector_lu_n"] - df["sector_lu_n_5d_ago"].fillna(0)

        # 清理
        df.drop(columns=["sector_n", "sector_lu_n_5d_ago", "is_lu"], errors="ignore", inplace=True)
    else:
        # 无行业数据时占位
        for c in ["sector_lu_n", "sector_lu_pct", "sector_ret_mean", "sector_excess_ret",
                   "sector_rank_pct", "is_sector_leader", "sector_lu_momentum"]:
            df[c] = np.nan
        if "is_lu" in df.columns:
            df.drop(columns=["is_lu"], inplace=True)

    return df


def compute_rank_ic(factor: np.ndarray, ret: np.ndarray) -> float:
    mask = np.isfinite(factor) & np.isfinite(ret)
    if mask.sum() < 10:
        return np.nan
    ic, _ = spearmanr(factor[mask], ret[mask])
    return ic if not np.isnan(ic) else np.nan


def main():
    args = parse_args()
    end_date = args.end or date.today().strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # ── 1. 加载信号 ──
    signals = load_signal_data(args)
    n_signals = len(signals)
    signal_codes = sorted(signals["code"].unique().tolist())
    logger.info(f"信号: {n_signals} 条, {len(signal_codes)} 只, {signals['trade_date'].nunique()} 天")

    # ── 2. 加载日线数据 ──
    pre_start = (pd.Timestamp(args.start) - timedelta(days=90)).strftime("%Y-%m-%d")
    post_end = (pd.Timestamp(end_date) + timedelta(days=30)).strftime("%Y-%m-%d")

    engine = get_engine()
    daily = load_daily_data(engine, signal_codes, pre_start, post_end,
                            cols=["open", "high", "low", "close", "volume", "amount", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    logger.info(f"日线数据: {len(daily)} 行")

    # ── 加载行业映射 ──
    basic = pd.read_sql("SELECT code, industry FROM stock_basic WHERE is_st = FALSE", engine)
    industry_map = dict(zip(basic["code"].astype(str).str.zfill(6), basic["industry"]))
    daily["industry"] = daily["code"].map(industry_map).fillna("其他")
    engine.dispose()
    logger.info(f"行业映射: {len(industry_map)} 只, 行业数: {daily['industry'].nunique()}")

    # ── 3. 计算所有因子 ──
    logger.info("计算个股因子...")
    daily = compute_factors_fast(daily)
    logger.info(f"个股因子完成: {daily.shape}")

    # ── 3.5 截面排名 + 板块共振因子 ──
    logger.info("计算截面/板块因子...")
    daily = compute_cross_sectional_factors(daily)
    logger.info(f"全部因子完成: {daily.shape}")

    # ── 4. 前瞻收益 ──
    for h in [5, 10, 20]:
        daily[f"ret_fwd_{h}d"] = daily.groupby("code")["close"].transform(
            lambda x: x.shift(-h) / x - 1
        )

    # ── 5. 提取信号日因子 ──
    logger.info("提取信号日因子值...")
    signals["key"] = (signals["trade_date"].astype(str) + "_" + signals["code"].astype(str).str.zfill(6))
    daily["key"] = (daily["trade_date"].astype(str) + "_" + daily["code"].astype(str).str.zfill(6))

    factor_cols = [c for c in daily.columns if c not in
                   ["code", "trade_date", "key", "open", "high", "low", "close", "volume",
                    "amount", "turnover", "market_cap", "ret", "industry"]
                   and pd.api.types.is_numeric_dtype(daily[c])
                   and not c.startswith("ret_fwd_")]  # 标签列，非因子

    sig_factors = signals[["key", "trade_date", "code"]].merge(
        daily[["key"] + factor_cols], on="key", how="inner"
    )
    logger.info(f"匹配到因子值的信号: {len(sig_factors)} / {n_signals}")

    # ── 6. IC 分析 ──
    horizons = [5, 10, 20]
    all_results = []

    for h in horizons:
        ret_col = f"ret_fwd_{h}d"
        valid = sig_factors.dropna(subset=[ret_col])
        logger.info(f"{h}d: 有效信号 {len(valid)}")

        records = []
        for dt, g in valid.groupby("trade_date"):
            row = {"trade_date": dt}
            for fc in factor_cols:
                row[fc] = compute_rank_ic(g[fc].values, g[ret_col].values)
            records.append(row)

        ic_df = pd.DataFrame(records)
        # 汇总
        for fc in factor_cols:
            ics = ic_df[fc].dropna()
            if len(ics) > 5:
                ic_mean = ics.mean()
                ic_std = ics.std(ddof=0)
                icir = ic_mean / ic_std if ic_std > 0 else np.nan
                all_results.append({
                    "factor": fc, "horizon": f"{h}d",
                    "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir, "n_days": len(ics),
                })

    summary = pd.DataFrame(all_results)

    # ── 7. 排序输出 ──
    for h_label in ["5d", "10d", "20d"]:
        sub = summary[summary["horizon"] == h_label].sort_values("ic_mean", key=abs, ascending=False)
        if sub.empty:
            continue
        top = sub.head(25)
        n = len(sub)
        n_pos = (sub["ic_mean"] > 0).sum()
        n_sig = (sub["ic_mean"].abs() > 0.02).sum()
        lu_sig = len([f for f in sub["factor"] if f.startswith("lu_") or f.startswith("pre_lu_")])

        print(f"\n{'='*80}")
        print(f"  {h_label} 前瞻 RankIC — Top-25 ({n} 个因子, IC>0: {n_pos}, |IC|>0.02: {n_sig}, 涨停因子: {lu_sig})")
        print(f"{'='*80}")
        print(f"{'因子':<30s} {'IC_mean':>8s} {'IC_std':>8s} {'ICIR':>8s} {'n_days':>7s} {'类型':>8s}")
        print("-" * 75)
        for _, row in top.iterrows():
            kind = "🟢涨停" if (row["factor"].startswith("lu_") or row["factor"].startswith("pre_lu_")) else ""
            print(f"{row['factor']:<30s} {row['ic_mean']:>+8.4f} {row['ic_std']:>8.4f} "
                  f"{row['icir']:>+8.3f} {row['n_days']:>7.0f} {kind:>8s}")

    # ── 8. 涨停因子专项汇总 ──
    lu_summary = summary[summary["factor"].str.startswith(("lu_", "pre_lu_"))]
    print(f"\n{'='*80}")
    print(f"  涨停专用因子 IC 汇总 ({len(lu_summary['factor'].unique())} 个)")
    print(f"{'='*80}")

    # 按 10d IC 排序
    lu_10d = lu_summary[lu_summary["horizon"] == "10d"].sort_values("ic_mean", key=abs, ascending=False)
    if not lu_10d.empty:
        print(f"\n{'因子':<30s} {'5d_IC':>8s} {'10d_IC':>8s} {'20d_IC':>8s} {'ICIR(10d)':>10s} {'评价':>8s}")
        print("-" * 80)
        for _, row in lu_10d.iterrows():
            ic5 = lu_summary[(lu_summary["factor"] == row["factor"]) & (lu_summary["horizon"] == "5d")]
            ic20 = lu_summary[(lu_summary["factor"] == row["factor"]) & (lu_summary["horizon"] == "20d")]
            ic5_val = f"{ic5['ic_mean'].iloc[0]:+.4f}" if len(ic5) > 0 else "N/A"
            ic20_val = f"{ic20['ic_mean'].iloc[0]:+.4f}" if len(ic20) > 0 else "N/A"
            # 评价
            ic_abs = abs(row["ic_mean"])
            icir_abs = abs(row["icir"]) if pd.notna(row["icir"]) else 0
            if ic_abs > 0.03 and icir_abs > 0.5:
                grade = "⭐⭐⭐"
            elif ic_abs > 0.02 and icir_abs > 0.3:
                grade = "⭐⭐"
            elif ic_abs > 0.01:
                grade = "⭐"
            else:
                grade = "❌"
            print(f"{row['factor']:<30s} {ic5_val:>8s} {row['ic_mean']:>+8.4f} {ic20_val:>8s} "
                  f"{row['icir']:>+10.3f} {grade:>8s}")

    # ── 9. 保存 ──
    csv_path = os.path.join(OUT_DIR, f"ic_fast_{args.start.replace('-','')}_{end_date.replace('-','')}.csv")
    summary.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    n_sig = (summary["ic_mean"].abs() > 0.02).sum()
    lu_n_sig = len(lu_summary[lu_summary["ic_mean"].abs() > 0.02]["factor"].unique())
    print(f"\n✅ 完成 ({elapsed:.0f}s) | |IC|>0.02: {n_sig} 个因子 (含 {lu_n_sig} 个涨停因子)")
    print(f"   结果: {csv_path}")


if __name__ == "__main__":
    main()
