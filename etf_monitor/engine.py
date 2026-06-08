"""ETF 三因子分析引擎（与 etf-three-factor-v7.skill 一致）。

量能概率 50% + 方向概率 20% + 份额概率 30%。
"""
import numpy as np
import pandas as pd


def vprob(vol_ratio: float) -> float:
    """量能概率 — 与 etf_v7_threefactor.py:vprob 一致的分段映射。

    <0.5x → 0-5, 0.5-1→5-17, 1-1.3→17-35, 1.3-1.5→35-55,
    1.5-2→55-80, 2-3→80-95, 3-5→95-98, >5→98-100
    """
    r = vol_ratio
    if r < 0.5:   return max(0, r / 0.5 * 5)
    if r < 1.0:   return 5 + (r - 0.5) / 0.5 * 12
    if r < 1.3:   return 17 + (r - 1) / 0.3 * 18
    if r < 1.5:   return 35 + (r - 1.3) / 0.2 * 20
    if r < 2.0:   return 55 + (r - 1.5) / 0.5 * 25
    if r < 3.0:   return 80 + (r - 2) / 1.0 * 15
    if r < 5.0:   return 95 + (r - 3) / 2.0 * 3
    return min(100, 98 + (r - 5) / 5.0 * 2)


def dprob(chg: float, t5_etf: float, t5_idx: float, vr: float, idx_chg: float) -> float:
    """方向概率 — 与 etf_v7_threefactor.py:dprob 一致。

    4子维度(f1/f2/f3/f4)加权 + 普涨折扣(rally_discount)。
    """
    # 普涨折扣
    if idx_chg > 2.0:       rally_discount = 0.60
    elif idx_chg > 1.5:     rally_discount = 0.70
    elif idx_chg > 1.0:     rally_discount = 0.80
    elif idx_chg > 0.5:     rally_discount = 0.90
    else:                   rally_discount = 1.0

    # f1: 涨跌×量价×大盘环境
    if chg > 0.3 and t5_idx < -1:             f1 = 95
    elif chg > 0 and t5_idx < -0.5:           f1 = 85
    elif chg > 0 and t5_idx < 0:              f1 = 70
    elif abs(chg) < 0.15 and t5_idx < -1:     f1 = 80
    elif abs(chg) < 0.3 and t5_idx < -0.5:    f1 = 65
    elif chg > 1 and vr > 1.5 and idx_chg > 1: f1 = 25
    elif chg > 1 and vr > 1.5:                f1 = 45
    elif chg > 0.5 and vr > 1.3 and idx_chg > 1: f1 = 35
    elif chg > 0.5 and vr > 1.3:              f1 = 50
    elif chg > 0:                             f1 = 40
    elif chg < -1.5 and vr > 2:               f1 = 8
    elif chg < -0.5 and vr > 1.5:             f1 = 15
    else:                                     f1 = 25

    # f2: ETF vs 指数超额
    gap = t5_etf - t5_idx
    if gap > 3:      f2 = 95
    elif gap > 2:    f2 = 85
    elif gap > 1.2:  f2 = 75
    elif gap > 0.6:  f2 = 60
    elif gap > 0.2:  f2 = 50
    elif gap > -0.2: f2 = 40
    elif gap > -0.6: f2 = 30
    else:            f2 = 15

    # f3: 指数超跌 bounces
    if t5_idx < -4:     f3 = 95
    elif t5_idx < -3:   f3 = 90
    elif t5_idx < -2:   f3 = 80
    elif t5_idx < -1:   f3 = 70
    elif t5_idx < -0.5: f3 = 55
    elif t5_idx < 0:    f3 = 45
    elif t5_idx < 1:    f3 = 35
    elif t5_idx < 3:    f3 = 20
    else:               f3 = 10

    f4 = 35  # 基准分

    raw = f1 * 0.4 + f2 * 0.3 + f3 * 0.2 + f4 * 0.1
    return round(raw * rally_discount, 1)


def sprob(share_delta_pct: float | None) -> float | None:
    """份额概率 — 与 etf_v7_threefactor.py:sprob 一致。

    >10%→95, >5%→80-95, >3%→65-80, >1%→45-65, 0-1%→30-45,
    -1-0%→15-30, -5--1%→5-15, <-5%→0-5
    """
    if share_delta_pct is None:
        return None
    ap = abs(share_delta_pct)
    if share_delta_pct > 10:   return 95.0
    if share_delta_pct > 5:    return 80 + (share_delta_pct - 5) / 5 * 15
    if share_delta_pct > 3:    return 65 + (share_delta_pct - 3) / 2 * 15
    if share_delta_pct > 1:    return 45 + (share_delta_pct - 1) / 2 * 20
    if share_delta_pct > 0:    return 30 + share_delta_pct / 1 * 15
    if share_delta_pct > -1:   return 15 + (share_delta_pct + 1) / 1 * 15
    if share_delta_pct > -5:   return 5 + (share_delta_pct + 5) / 4 * 10
    return max(0, 5 + (share_delta_pct + 5) / 5 * 5)


def analyze_single(code: str, name: str, kline: pd.DataFrame,
                   idx_kline: pd.DataFrame, shares_delta_pct: float | None = None) -> dict:
    """单 ETF 三因子分析 — 与 etf_v7_threefactor.py:analyze_all 一致。"""
    if kline.empty or len(kline) < 22:
        return {"code": code, "name": name, "error": "K线数据不足(需≥22天)"}

    close = kline["close"].astype(float)
    volume = kline["volume"].astype(float)
    idx_close = idx_kline["close"].astype(float) if not idx_kline.empty else None

    i = len(close) - 1  # latest day index
    v = volume.iloc[i] / 10000
    ma = volume.iloc[i-20:i].mean() / 10000
    vr = v / ma if ma > 0 else 1.0
    pc = close.iloc[i-1]
    chg = (close.iloc[i] - pc) / pc * 100 if pc > 0 else 0
    t5_etf = (close.iloc[i] - close.iloc[i-5]) / close.iloc[i-5] * 100 if i >= 5 and close.iloc[i-5] > 0 else 0
    t5_idx = 0.0
    idx_chg = 0.0
    if idx_close is not None and len(idx_close) > i:
        ii = i
        if len(idx_close) > ii and idx_close.iloc[ii-1] > 0:
            idx_chg = (idx_close.iloc[ii] - idx_close.iloc[ii-1]) / idx_close.iloc[ii-1] * 100
        if ii >= 5 and idx_close.iloc[ii-5] > 0:
            t5_idx = (idx_close.iloc[ii] - idx_close.iloc[ii-5]) / idx_close.iloc[ii-5] * 100

    vp = vprob(vr)
    dp = dprob(chg, t5_etf, t5_idx, vr, idx_chg)
    sp = sprob(shares_delta_pct)

    if sp is not None:
        cp = round(vp * 0.5 + dp * 0.2 + sp * 0.3, 1)
    else:
        cp = round(vp * 0.7 + dp * 0.3, 1)

    signal = "high" if cp >= 70 else ("mid" if cp >= 50 else "normal")

    return {
        "code": code, "name": name,
        "close": round(float(close.iloc[-1]), 3),
        "chg_pct": round(chg, 2),
        "volume_ma20": round(float(ma * 10000), 0),
        "vol_ratio": round(vr, 2),
        "vol_prob": round(vp, 1),
        "dir_prob": round(dp, 1),
        "share_prob": round(sp, 1) if sp is not None else None,
        "shares_delta_pct": round(shares_delta_pct, 2) if shares_delta_pct is not None else None,
        "composite_prob": round(cp, 1),
        "signal_level": signal,
    }


def analyze_all(kline_map: dict[str, pd.DataFrame], idx_kline: pd.DataFrame,
                shares_map: dict[str, float], etfs: dict = None) -> list[dict]:
    """批量分析全部 ETF。"""
    if etfs is None:
        from etf_monitor.config import load_etfs
        from data.db import get_engine
        etfs = load_etfs(get_engine())
    results = []
    for code, info in etfs.items():
        kl = kline_map.get(code)
        if kl is None or kl.empty:
            results.append({"code": code, "name": info["name"], "error": "无K线数据"})
            continue
        r = analyze_single(code, info["name"], kl, idx_kline, shares_map.get(code))
        results.append(r)
    return results
