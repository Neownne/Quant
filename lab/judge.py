"""策略评判 —— 复合评分 + 排名 + 报告生成。"""
from __future__ import annotations

import json
import math
from datetime import date
from typing import Any

from loguru import logger


def _safe_float(v: Any) -> float:
    """安全转 float，None/非数字 → 0.0。"""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _minmax_norm(values: list[float], clip_lo: float | None = None,
                 clip_hi: float | None = None) -> list[float]:
    """Min-max 归一化到 [0, 1]，可选裁剪。"""
    if not values or max(values) == min(values):
        return [0.5] * len(values)
    vmin, vmax = min(values), max(values)
    normed = [(v - vmin) / (vmax - vmin) for v in values]
    if clip_lo is not None or clip_hi is not None:
        lo = clip_lo if clip_lo is not None else min(normed)
        hi = clip_hi if clip_hi is not None else max(normed)
        normed = [max(lo, min(hi, n)) for n in normed]
    return normed


def judge(results: list[dict]) -> list[dict]:
    """对回测结果计算复合评分并排名。

    评分权重：
      - 0.25 × Sharpe（归一化）
      - 0.20 × 年化收益（归一化）
      - 0.20 × (1 - 最大回撤)（归一化）
      - 0.15 × Calmar 比率（归一化）
      - 0.10 × 稳定性（正收益年份占比）
      - 0.10 × 一致性（逐年 Sharpe 标准差，越低越好）
    """
    if not results:
        return []

    # 提取原始指标
    sharpes = [_safe_float(r.get("sharpe")) for r in results]
    returns = [_safe_float(r.get("return")) for r in results]
    mdds = [_safe_float(r.get("max_drawdown")) for r in results]

    # Calmar = return / |MDD|
    calmars = []
    for ret, mdd in zip(returns, mdds):
        if mdd and mdd > 0.001:
            calmars.append(abs(ret) / mdd)
        else:
            calmars.append(0.0)

    # 归一化
    sharpe_n = _minmax_norm(sharpes, clip_lo=0.0, clip_hi=1.0)
    ret_n = _minmax_norm(returns, clip_lo=0.0, clip_hi=1.0)
    mdd_n = _minmax_norm([1.0 - m for m in mdds], clip_lo=0.0, clip_hi=1.0)  # 回撤越小越好
    calmar_n = _minmax_norm(calmars, clip_lo=0.0, clip_hi=1.0)

    # 稳定性：从 metrics_json 中找 annual_returns
    stabilities = []
    consistencies = []
    for r in results:
        # 如果有分年数据
        ann = r.get("annual_returns", {})
        if ann and isinstance(ann, dict):
            yr_returns = [_safe_float(v) for v in ann.values()]
            yr_sharpes = []  # 逐年 Sharpe 需要额外数据
            pos_years = sum(1 for v in yr_returns if v > 0)
            stabilities.append(pos_years / max(len(yr_returns), 1))
            # 用逐年收益的标准差代理一致性
            if len(yr_returns) >= 2:
                mean_ret = sum(yr_returns) / len(yr_returns)
                std_ret = math.sqrt(sum((v - mean_ret) ** 2 for v in yr_returns) / len(yr_returns))
                consistencies.append(1.0 / (1.0 + std_ret))  # 标准差越小分数越高
            else:
                consistencies.append(0.5)
        else:
            stabilities.append(0.5)
            consistencies.append(0.5)

    stab_n = _minmax_norm(stabilities, clip_lo=0.0, clip_hi=1.0)
    cons_n = _minmax_norm(consistencies, clip_lo=0.0, clip_hi=1.0)

    # 复合评分
    for i, r in enumerate(results):
        r["_score"] = round(
            0.25 * sharpe_n[i] +
            0.20 * ret_n[i] +
            0.20 * mdd_n[i] +
            0.15 * calmar_n[i] +
            0.10 * stab_n[i] +
            0.10 * cons_n[i], 4
        )
        r["_sharpe_norm"] = round(sharpe_n[i], 4)
        r["_calmar"] = round(calmars[i], 4)

    # 排名
    ranked = sorted(results, key=lambda x: x["_score"], reverse=True)
    n = len(ranked)
    for i, r in enumerate(ranked):
        r["_rank"] = i + 1
        if (i + 1) <= n * 0.25 and sharpes[results.index(r)] > 0.5 and mdds[results.index(r)] < 0.25:
            r["_verdict"] = "promising"
        elif sharpes[results.index(r)] < 0 or mdds[results.index(r)] > 0.30:
            r["_verdict"] = "reject"
        elif (i + 1) > n * 0.75:
            r["_verdict"] = "reject"
        else:
            r["_verdict"] = "baseline"

    return ranked


def print_report(ranked: list[dict]) -> None:
    """终端打印排名表。"""
    print(f"\n{'='*90}")
    print(f"  策略变体排名报告")
    print(f"{'='*90}")
    header = f"  {'排名':<4} {'变体':<25s} {'评分':>6s} {'Sharpe':>7s} {'收益':>8s} {'回撤':>7s} {'Calmar':>7s} {'判定':<10s}"
    print(header)
    print("  " + "-" * 82)
    for r in ranked:
        verdict_icon = {"promising": "🟢", "baseline": "🟡", "reject": "🔴"}.get(r["_verdict"], "⚪")
        print(f"  {r['_rank']:<4} {r['variant_name']:<25s} "
              f"{r['_score']:>6.3f} "
              f"{_safe_float(r.get('sharpe')):>7.2f} "
              f"{_safe_float(r.get('return')):>7.1%} "
              f"{_safe_float(r.get('max_drawdown')):>6.1%} "
              f"{r.get('_calmar', 0):>7.2f} "
              f"{verdict_icon} {r['_verdict']:<7s}")

    # 汇总
    promising = [r for r in ranked if r["_verdict"] == "promising"]
    baseline = [r for r in ranked if r["_verdict"] == "baseline"]
    rejected = [r for r in ranked if r["_verdict"] == "reject"]
    print(f"\n  🟢 promising: {len(promising)} | 🟡 baseline: {len(baseline)} | 🔴 reject: {len(rejected)}")
    if promising:
        print(f"  最优: {promising[0]['variant_name']} (评分: {promising[0]['_score']:.3f})")


def save_report(ranked: list[dict], path: str) -> None:
    """保存排名报告为 JSON。"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 清理内部字段
    clean = []
    for r in ranked:
        d = {k: v for k, v in r.items()}
        d["score"] = d.pop("_score", 0)
        d["rank"] = d.pop("_rank", 0)
        d["verdict"] = d.pop("_verdict", "unknown")
        clean.append(d)
    report = {
        "date": date.today().strftime("%Y-%m-%d"),
        "variants": clean,
        "summary": {
            "total": len(clean),
            "promising": sum(1 for r in clean if r["verdict"] == "promising"),
            "baseline": sum(1 for r in clean if r["verdict"] == "baseline"),
            "reject": sum(1 for r in clean if r["verdict"] == "reject"),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"报告已保存: {path}")
