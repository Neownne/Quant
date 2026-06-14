"""策略搜索模块 —— 查询模板库 + 搜索结果 → 变体 JSON。

Web 搜索由 Claude Code 的 WebSearch/WebFetch 工具执行（Python 子进程无法直接调用）。
本模块提供：
  1. 分轮搜索查询模板
  2. 搜索结果 → StrategyVariant 的结构化提取规范
  3. 参数验证与去重
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from typing import Any

from lab.variant import StrategyVariant

# ── 多轮搜索查询模板 ──
SEARCH_ROUNDS: list[list[str]] = [
    # 第1轮：涨停策略最新研究
    [
        "A股 涨停策略 量化 因子 2026 最新研究",
        "limit-up strategy A-share stock selection factor 2026",
    ],
    # 第2轮：行业/板块轮动
    [
        "A股 行业轮动策略 量化 回测 2026 最新报告",
        "sector rotation A-share quantitative strategy 2026",
    ],
    # 第3轮：行为金融/另类因子
    [
        "A股 行为金融 因子 涨停 情绪 2026",
        "涨停板 首板 回调 策略 买入条件 止损 参数 2026",
    ],
    # 第4轮：风控/仓位管理
    [
        "量化策略 仓位管理 自适应止损 2026 A股",
    ],
]

# ── 参数合理范围（验证用）──
PARAM_BOUNDS = {
    "mcap_min": (1, 10000), "mcap_max": (1, 10000),
    "price_min": (0.1, 500), "price_max": (0.1, 500),
    "lu_lookback": (5, 120), "lu_count": (0, 20),
    "min_conditions": (1, 5), "min_listed_days": (60, 500),
    "top_n": (1, 50), "stop_loss_pct": (0.01, 0.30),
}


def extract_params_from_text(text: str) -> dict[str, Any] | None:
    """从搜索文本中尝试提取策略参数。

    这是一个启发式提取器，识别常见的数值模式：
    - "市值 X-Y 亿" → mcap_min, mcap_max
    - "股价 X-Y 元" → price_min, price_max
    - "近 N 日涨停" → lu_lookback
    - "涨停 > N 次" → lu_count
    - "止损 -X%" → stop_loss_pct

    返回 None 表示无法提取足够参数。
    """
    import re

    params: dict[str, Any] = {}
    confidence = 0  # 置信度分数

    # 市值区间
    m = re.search(r"市值[：:\s]*(\d+)\s*[-~至到]\s*(\d+)\s*亿", text)
    if not m:
        m = re.search(r"市值[：:\s]*(\d+)\s*[-~至到]\s*(\d+)", text)
    if m:
        params["mcap_min"] = float(m.group(1))
        params["mcap_max"] = float(m.group(2))
        confidence += 2

    # 股价区间
    m = re.search(r"股?价[：:\s]*(\d+\.?\d*)\s*[-~至到]\s*(\d+\.?\d*)\s*元", text)
    if m:
        params["price_min"] = float(m.group(1))
        params["price_max"] = float(m.group(2))
        confidence += 2

    # 涨停回溯天数
    m = re.search(r"近\s*(\d+)\s*日.*涨停", text)
    if m:
        params["lu_lookback"] = int(m.group(1))
        confidence += 2

    # 涨停次数阈值
    m = re.search(r"涨停.*?[>＞]\s*(\d+)\s*次", text)
    if m:
        params["lu_count"] = int(m.group(1))
        confidence += 2

    # 均线
    m = re.search(r"MA\s*(\d+)\s*[>＞]\s*MA\s*(\d+)", text)
    if m:
        params["ma_fast"] = int(m.group(1))
        params["ma_slow"] = int(m.group(2))
        confidence += 1

    # 止损
    m = re.search(r"止损[：:\s]*[-−]?\s*(\d+\.?\d*)\s*%", text)
    if m:
        params["stop_loss_pct"] = float(m.group(1)) / 100.0
        confidence += 2

    # 止盈
    m = re.search(r"止盈[：:\s]*[+＋]?\s*(\d+\.?\d*)\s*%", text)
    if m:
        params["take_profit_pct"] = float(m.group(1)) / 100.0
        confidence += 1

    # 持仓数
    m = re.search(r"持仓[：:\s]*(\d+)\s*只", text)
    if m:
        params["top_n"] = int(m.group(1))
        confidence += 1

    # 回撤
    m = re.search(r"最大回撤[：:\s]*[-−]?\s*(\d+\.?\d*)\s*%", text)
    if m:
        params["reported_mdd"] = float(m.group(1)) / 100.0

    # 夏普
    m = re.search(r"[Ss]harpe?[：:\s]*(\d+\.?\d*)", text)
    if m:
        params["reported_sharpe"] = float(m.group(1))

    # 成交量过滤
    if re.search(r"量比|成交量.*[>＞].*均量|放量", text):
        params["use_volume_filter"] = True
        m = re.search(r"量比[：:\s]*[>＞]\s*(\d+\.?\d*)", text)
        if m:
            params["volume_ratio_min"] = float(m.group(1))
        m = re.search(r"成交量.*?(\d+\.?\d*)\s*倍", text)
        if m:
            params["volume_ratio_min"] = float(m.group(1))
        confidence += 1

    # RSI
    m = re.search(r"RSI[：:\s]*(\d+)\s*[-~至到]\s*(\d+)", text)
    if m:
        params["use_rsi_filter"] = True
        params["rsi_min"] = float(m.group(1))
        params["rsi_max"] = float(m.group(2))
        confidence += 1

    if confidence < 3:  # 至少 3 个参数才认为有效
        return None

    return params


def validated_variant(params: dict[str, Any], source_url: str = "",
                      title: str = "") -> StrategyVariant | None:
    """验证参数范围并创建 StrategyVariant。"""
    # 检查范围
    for key, (lo, hi) in PARAM_BOUNDS.items():
        if key in params and params[key] is not None:
            val = params[key]
            if not (lo <= val <= hi):
                return None  # 超出合理范围

    # 交叉验证
    if "mcap_min" in params and "mcap_max" in params:
        if params["mcap_min"] >= params["mcap_max"]:
            return None

    # 生成唯一名称
    slug = hashlib.md5((source_url + title).encode()).hexdigest()[:8]

    return StrategyVariant(
        name=f"web_{slug}",
        description=title[:200] if title else "Web 搜索发现的策略变体",
        source="web_search",
        source_url=source_url,
        source_date=date.today().strftime("%Y-%m-%d"),
        **{k: v for k, v in params.items()
           if k in StrategyVariant.__dataclass_fields__},
    )


def get_search_queries(num_rounds: int = 3) -> list[list[str]]:
    """获取前 N 轮搜索查询。"""
    return SEARCH_ROUNDS[:num_rounds]


def search_multi_rounds(num_rounds: int = 3, output_dir: str = "lab/variants"):
    """多轮搜索入口 —— 打印搜索指引。

    实际的 WebSearch/WebFetch 调用由 Claude Code 会话执行。
    本函数打印搜索计划，指导 Claude 如何执行搜索。
    """
    rounds = get_search_queries(num_rounds)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  策略搜索计划 — {num_rounds} 轮")
    print(f"{'='*70}")
    for i, queries in enumerate(rounds):
        print(f"\n  第 {i+1} 轮:")
        for q in queries:
            print(f"    → {q}")

    print(f"\n  ═══════════════════════════════════════════")
    print(f"  Claude Code 会话中运行以下命令来执行搜索：")
    print(f"  ")
    print(f"  from lab.searcher import run_search_session")
    print(f"  await run_search_session(num_rounds={num_rounds})")
    print(f"  ═══════════════════════════════════════════\n")

    return rounds
