"""ETF 三因子分析引擎：量能/方向/份额 → 综合概率。"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_volume_prob(vol_ratio: float) -> float:
    """倍量→概率分段线性映射。

    1.0x → 50 (正常)
    1.5x → 65
    2.0x → 80
    2.5x → 90
    3.0x → 100 (钳制上限)
    """
    if vol_ratio <= 1.0:
        return 50.0
    if vol_ratio >= 3.0:
        return 100.0
    # 线性: 1.0→50, 3.0→100
    return 50.0 + (vol_ratio - 1.0) / 2.0 * 50.0


def compute_direction_prob(
    etf_chg: float,
    t5_etf: float,
    t5_idx: float,
    vol_ratio: float,
    idx_chg: float,
) -> float:
    """方向概率 = 4子维度加权 + 普涨折扣。

    子维度:
    - 当日涨跌 (30%): chg>0→100, chg≤0→0
    - 近5日强弱 (30%): t5_etf - t5_idx
    - 量价配合 (20%): chg>0且放量→100
    - 指数环境 (20%): idx_chg映射
    普涨折扣: ETF+指数同涨→×0.8 (可能是普涨而非国家队)
    """
    # 当日涨跌
    d1 = 100.0 if etf_chg > 0 else 0.0

    # 近5日强弱 (ETF vs 指数)
    alpha = t5_etf - t5_idx
    d2 = min(100.0, max(0.0, 50.0 + alpha * 100))

    # 量价配合
    d3 = 100.0 if (etf_chg > 0 and vol_ratio > 1.2) else (50.0 if vol_ratio > 1.0 else 0.0)

    # 指数环境
    d4 = min(100.0, max(0.0, 50.0 + idx_chg * 200))

    prob = d1 * 0.30 + d2 * 0.30 + d3 * 0.20 + d4 * 0.20

    # 普涨折扣
    if etf_chg > 0 and idx_chg > 0:
        prob *= 0.85

    return min(100.0, max(0.0, prob))


def compute_share_prob(delta_pct: float) -> float:
    """份额变化%→概率映射。

    0% → 50, 5% → 70, 10% → 85, 15%+ → 95
    """
    if delta_pct <= 0:
        return 50.0
    if delta_pct >= 15:
        return 95.0
    # 分段线性
    if delta_pct <= 5:
        return 50.0 + delta_pct / 5.0 * 20.0
    if delta_pct <= 10:
        return 70.0 + (delta_pct - 5.0) / 5.0 * 15.0
    return 85.0 + (delta_pct - 10.0) / 5.0 * 10.0


def analyze_single(
    code: str, name: str,
    kline: pd.DataFrame,
    idx_kline: pd.DataFrame,
    shares_delta_pct: float | None = None,
) -> dict:
    """对单只 ETF 执行三因子分析。"""
    if kline.empty or len(kline) < 21:
        return {"code": code, "name": name, "error": "K线数据不足"}

    close = kline["close"].astype(float)
    volume = kline["volume"].astype(float)
    latest_close = close.iloc[-1]
    prev_close = close.iloc[-2]

    # 量能因子
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1.0
    vol_prob = compute_volume_prob(vol_ratio)

    # 方向因子
    etf_chg = (latest_close / prev_close - 1) * 100
    t5_etf = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
    idx_chg = 0.0
    t5_idx = 0.0
    if not idx_kline.empty and len(idx_kline) >= 6:
        idx_close = idx_kline["close"].astype(float)
        idx_chg = (idx_close.iloc[-1] / idx_close.iloc[-2] - 1) * 100
        t5_idx = (idx_close.iloc[-1] / idx_close.iloc[-6] - 1) * 100
    dir_prob = compute_direction_prob(etf_chg, t5_etf, t5_idx, vol_ratio, idx_chg)

    # 份额因子
    share_prob = compute_share_prob(shares_delta_pct) if shares_delta_pct is not None else None

    # 综合概率
    if share_prob is not None:
        composite = vol_prob * 0.50 + dir_prob * 0.20 + share_prob * 0.30
    else:
        composite = vol_prob * 0.70 + dir_prob * 0.30

    # 信号分级
    if composite >= 70:
        signal = "high"
    elif composite >= 50:
        signal = "mid"
    else:
        signal = "normal"

    return {
        "code": code, "name": name,
        "close": round(float(latest_close), 3),
        "chg_pct": round(etf_chg, 2),
        "volume_ma20": round(float(vol_ma20), 0),
        "vol_ratio": round(vol_ratio, 2),
        "vol_prob": round(vol_prob, 1),
        "dir_prob": round(dir_prob, 1),
        "share_prob": round(share_prob, 1) if share_prob is not None else None,
        "shares_delta_pct": round(shares_delta_pct, 2) if shares_delta_pct is not None else None,
        "composite_prob": round(composite, 1),
        "signal_level": signal,
    }


def analyze_all(
    kline_map: dict[str, pd.DataFrame],
    idx_kline: pd.DataFrame,
    shares_map: dict[str, float],
) -> list[dict]:
    """批量分析全部 ETF。"""
    from etf_monitor.config import ETFS
    results = []
    for code, info in ETFS.items():
        kl = kline_map.get(code)
        if kl is None or kl.empty:
            results.append({"code": code, "name": info["name"], "error": "无K线数据"})
            continue
        shares_delta = shares_map.get(code)
        r = analyze_single(code, info["name"], kl, idx_kline, shares_delta)
        results.append(r)
    return results
