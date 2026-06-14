"""网格搜索变体生成器 —— 基于 E4 基线系统性参数扫描。"""
from __future__ import annotations

from itertools import product

from lab.variant import StrategyVariant, E4_BASELINE

# ── 网格维度 ──
MCAP_GRID = [
    (20, 300, "市值20-300亿（收紧下限）"),
    (30, 500, "市值30-500亿（E4基线）"),
    (50, 200, "市值50-200亿（N字策略）"),
    (50, 800, "市值50-800亿（放宽上限）"),
    (100, 1000, "市值100-1000亿（大盘）"),
]

LOOKBACK_GRID = [
    (10, "近10日涨停"),
    (20, "近20日涨停（E4基线）"),
    (30, "近30日涨停"),
    (60, "近60日涨停"),
]

LU_COUNT_GRID = [
    (1, ">1次涨停"),
    (2, ">2次涨停"),
    (3, ">3次涨停"),
]

SCORING_MODES = [
    ("none", {}),
    ("decay", {"lu_decay": True, "description": "时间衰减加权"}),
    ("quality", {"lu_quality": True, "description": "涨停质量过滤"}),
    ("streak", {"lu_streak": True, "description": "连板加分"}),
    ("all", {"lu_score": True, "lu_decay": True, "lu_quality": True, "lu_streak": True,
             "description": "全评分增强"}),
]

FILTER_MODES = [
    ("no_filter", {}),
    ("trend", {"trend_filter": True, "description": "CSI1000趋势过滤"}),
    ("no5day", {"no_5day_streak": True, "description": "排除5连板"}),
]

TOP_N_VARIANTS = [3, 5, 10]


def generate_grid_variants(base: StrategyVariant = None) -> list[StrategyVariant]:
    """生成系统化的网格搜索变体集合。

    不会穷举所有组合（组合爆炸），而是按维度独立扫描：
    - 市值维度：5 个变体
    - lookback 维度：4 个变体
    - lu_count 维度：3 个变体
    - 评分增强：5 个变体
    - 过滤模式：2 个变体
    - top_n：3 个变体

    总计约 22 个变体，每个变体只改一个维度，其他用 E4 默认。
    """
    if base is None:
        base = E4_BASELINE

    variants = []

    # 1. 基线（从 E4_BASELINE 拷贝，只覆盖 name/description/source）
    base_fields = {k: v for k, v in base.__dict__.items()
                   if k in StrategyVariant.__dataclass_fields__
                   and not k.startswith("_")
                   and k not in ("name", "description", "source")}
    variants.append(StrategyVariant(
        name="E4_baseline", description="E4 基线：4条件去跌停",
        source="manual", **base_fields,
    ))

    # 2. 市值维度
    for (lo, hi, desc) in MCAP_GRID:
        if lo == 30 and hi == 500:
            continue  # 基线已覆盖
        label = f"mcap_{lo}_{hi}"
        variants.append(StrategyVariant(
            name=f"E5_{label}",
            mcap_min=lo, mcap_max=hi,
            description=desc,
            source="grid_search",
        ))

    # 3. lookback 维度
    for (lb, desc) in LOOKBACK_GRID:
        if lb == 20:
            continue
        variants.append(StrategyVariant(
            name=f"E5_lb{lb}d", lu_lookback=lb,
            description=desc, source="grid_search",
        ))

    # 4. lu_count 维度
    for (cnt, desc) in LU_COUNT_GRID:
        if cnt == 1:
            continue
        variants.append(StrategyVariant(
            name=f"E5_lu>{cnt}", lu_count=cnt,
            description=desc, source="grid_search",
        ))

    # 5. 评分增强
    for (sname, sflags) in SCORING_MODES:
        if sname == "none":
            continue
        v = StrategyVariant(
            name=f"E5_score_{sname}",
            description=sflags.get("description", sname),
            source="grid_search",
        )
        for k in ["lu_score", "lu_decay", "lu_quality", "lu_streak"]:
            if k in sflags:
                setattr(v, k, sflags[k])
        variants.append(v)

    # 6. 过滤模式
    for (fname, fflags) in FILTER_MODES:
        if fname == "no_filter":
            continue
        v = StrategyVariant(
            name=f"E5_filt_{fname}",
            description=fflags.get("description", fname),
            source="grid_search",
        )
        for k in ["trend_filter", "no_5day_streak"]:
            if k in fflags:
                setattr(v, k, fflags[k])
        variants.append(v)

    # 7. top_n 维度
    for tn in TOP_N_VARIANTS:
        if tn == 5:
            continue
        variants.append(StrategyVariant(
            name=f"E5_top{tn}", top_n=tn,
            description=f"持仓 {tn} 只",
            source="grid_search",
        ))

    # 8. 反直觉发现（来自华安证券 2026 报告）
    variants.append(StrategyVariant(
        name="E5_uan_mcap_U",
        description="市值U型：<15亿小盘或>100亿大盘（跳过30-50亿中段）",
        mcap_min=5, mcap_max=1000, source="web_search",
        source_url="华安证券 2026-03 首板回调策略",
    ))

    variants.append(StrategyVariant(
        name="E5_tight_stop",
        description="收紧止损：stop_loss=5%（华安自适应止损-复苏期）",
        stop_loss_pct=0.05, source="web_search",
        source_url="华安证券 2026-03 阶段自适应止损",
    ))

    # 9. 移动止盈（跟踪最高点回落）
    for trail_pct in [0.08, 0.10, 0.12, 0.15]:
        variants.append(StrategyVariant(
            name=f"E5_trail{int(trail_pct*100)}",
            description=f"移动止盈-从最高点回落{trail_pct:.0%}卖出",
            use_trailing_stop=True, trailing_stop_pct=trail_pct,
            source="grid_search",
        ))

    # 10. 金字塔加仓
    for py_thresh, py_ratio in [(0.05, 0.5), (0.10, 0.5), (0.05, 1.0)]:
        label = f"E5_pyr{int(py_thresh*100)}x{int(py_ratio*100)}"
        variants.append(StrategyVariant(
            name=label,
            description=f"金字塔加仓: 盈利>{py_thresh:.0%}加{py_ratio:.0%}仓位",
            use_pyramid=True, pyramid_threshold=py_thresh, pyramid_ratio=py_ratio,
            source="grid_search",
        ))

    # 11. 入场冷却
    for cool_days, pos_day in [(3, False), (5, False), (3, True), (5, True)]:
        sd = f"{'pos' if pos_day else 'cool'}{cool_days}"
        desc = f"冷却{cool_days}天" + ("+收阳确认" if pos_day else "")
        variants.append(StrategyVariant(
            name=f"E5_{sd}",
            description=desc,
            use_cooling=True, cooling_days=cool_days,
            require_positive_day=pos_day,
            source="grid_search",
        ))

    # 12. 组合：移动止盈+金字塔+冷却 (最佳猜测组合)
    variants.append(StrategyVariant(
        name="E5_trail10_pyr5_cool3",
        description="三合一: 10%移动止盈+5%金字塔加仓+3天冷却",
        use_trailing_stop=True, trailing_stop_pct=0.10,
        use_pyramid=True, pyramid_threshold=0.05, pyramid_ratio=0.5,
        use_cooling=True, cooling_days=3,
        source="grid_search",
    ))

    # 去重
    seen = set()
    unique = []
    for v in variants:
        if v.name not in seen:
            seen.add(v.name)
            unique.append(v)
    return unique
