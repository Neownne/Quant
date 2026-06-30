#!/usr/bin/env python
"""演化可视化模块 — 6 面板 Dashboard + 单轮详情 + 终端进度条。

用法:
    from factors.viz import plot_evolution_dashboard, plot_round_detail, plot_terminal_summary

    db = json.load(open("data/factor_db.json"))
    plot_evolution_dashboard(db, save_path="data/evolution_dashboard.png")
    plot_terminal_summary(db)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter

# ── ANSI 颜色 ────────────────────────────────────────────────────────
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# ── 全局样式 ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 9,
    "axes.titlesize": 11,
    "figure.facecolor": "#f8f9fa",
})


def _pct_formatter() -> FuncFormatter:
    """百分比 y 轴格式化器."""
    return FuncFormatter(lambda x, _: f"{x * 100:.0f}%")


def _is_valid(value):
    """检查值是否有效（非 NaN、非 None）."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# 6 面板进化仪表盘
# ═══════════════════════════════════════════════════════════════════════

def plot_evolution_dashboard(db: dict, save_path: str | None = None):
    """多面板进化仪表盘，2 行 × 3 列。

    Panel:
      1. 适应度趋势 — best_fitness 折线图
      2. IC 趋势 — abs(best_ic) 折线图，IC=0.02 参考线
      3. 回测收益 & 回撤 — 双轴：bt_annual（绿，左轴）+ bt_mdd（红，右轴），40% 目标线
      4. 有效个体数 — n_valid 柱状图
      5. 运算符使用率 — 占位文本
      6. 数据源覆盖率 — 占位文本

    若 db 为空则显示 "No evolution data yet" 提示。
    """
    history = db.get("history", []) if isinstance(db, dict) else []

    # ── 空数据 ──
    if not history:
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.text(0.5, 0.5, "No evolution data yet",
                ha="center", va="center", fontsize=24, color="#6c757d",
                transform=ax.transAxes)
        ax.set_axis_off()
        _save_or_show(fig, save_path)
        return

    rounds_list = [entry.get("round", i + 1) for i, entry in enumerate(history)]

    # ── 提取各指标（安全访问） ──
    best_fitness = [_safe_float(entry, "best_fitness", np.nan) for entry in history]
    best_ic = [_safe_float(entry, "best_ic", np.nan) for entry in history]
    bt_annual = [_safe_float(entry, "bt_annual", np.nan) for entry in history]
    bt_mdd = [_safe_float(entry, "bt_mdd", np.nan) for entry in history]
    n_valid = [entry.get("n_valid", entry.get("n_individuals", np.nan)) for entry in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    # ── 1. 适应度趋势 ──
    ax = axes[0, 0]
    ax.plot(rounds_list, best_fitness, marker="o", color="#1f77b4",
            linewidth=1.5, markersize=4)
    ax.axhline(y=0, color="#6c757d", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.fill_between(rounds_list, 0, best_fitness, alpha=0.15, color="#1f77b4")
    ax.set_title("Fitness Trend")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))

    # ── 2. IC 趋势 ──
    ax = axes[0, 1]
    abs_ic = [abs(v) if _is_valid(v) else np.nan for v in best_ic]
    ax.plot(rounds_list, abs_ic, marker="s", color="#ff7f0e",
            linewidth=1.5, markersize=4)
    ax.axhline(y=0.02, color="#d62728", linestyle="--", linewidth=0.8,
               alpha=0.7, label="IC=0.02")
    ax.set_title("IC Trend (|best_ic|)")
    ax.set_ylabel("|IC|")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.4f}"))

    # ── 3. 回测收益 & 回撤（双轴） ──
    ax = axes[0, 2]
    ax2 = ax.twinx()
    line1, = ax.plot(rounds_list, bt_annual, marker="^", color="#2ca02c",
                     linewidth=1.5, markersize=4, label="Annual Return")
    line2, = ax2.plot(rounds_list, bt_mdd, marker="v", color="#d62728",
                      linewidth=1.5, markersize=4, label="MDD")
    ax.axhline(y=0.40, color="#2ca02c", linestyle="--", linewidth=0.8,
               alpha=0.5, label="+40%")
    ax.set_title("Backtest Returns & Drawdown")
    ax.set_ylabel("Annual Return", color="#2ca02c")
    ax2.set_ylabel("MDD", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.yaxis.set_major_formatter(_pct_formatter())
    ax2.yaxis.set_major_formatter(_pct_formatter())
    lines = [line1, line2, ax.get_lines()[0]]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── 4. 有效个体数 ──
    ax = axes[1, 0]
    colors = ["#17becf" if v and v > 0 else "#d62728" for v in n_valid]
    ax.bar(rounds_list, n_valid, color=colors, width=0.7, alpha=0.85)
    ax.set_title("Valid Individuals per Round")
    ax.set_ylabel("n_valid")
    ax.grid(True, alpha=0.3, axis="y")

    # ── 5. 运算符使用率（占位） ──
    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Operator Usage\n(analysis report data required)",
            ha="center", va="center", fontsize=12, color="#6c757d",
            transform=ax.transAxes)
    ax.set_title("Operator Usage")
    ax.set_axis_off()

    # ── 6. 数据源覆盖率（占位） ──
    ax = axes[1, 2]
    ax.text(0.5, 0.5, "Data Source Coverage\n(analysis report data required)",
            ha="center", va="center", fontsize=12, color="#6c757d",
            transform=ax.transAxes)
    ax.set_title("Data Source Coverage")
    ax.set_axis_off()

    fig.suptitle("Factor Evolution Dashboard", fontsize=14, fontweight="bold", y=0.98)
    _save_or_show(fig, save_path)


# ═══════════════════════════════════════════════════════════════════════
# 单轮详情图
# ═══════════════════════════════════════════════════════════════════════

def plot_round_detail(report: dict, save_path: str | None = None):
    """单轮详情 2×2 子图。

    Panel:
      1. 树深度 vs IC 散点图 — top_factors 的 (depth, ic) 散点，标注因子名
      2. 行业 IC 分组柱状图 — 各行业 IC 水平柱状图
      3. 运算符使用率饼图 — 运算符使用次数饼图
      4. 因子冗余度 — 居中大数字 high_corr_count
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    # ── 1. 树深度 vs IC 散点图 ──
    ax = axes[0, 0]
    top_factors = report.get("top_factors", [])
    if top_factors:
        depths = [f.get("depth", i + 1) for i, f in enumerate(top_factors)]
        ics = [f.get("ic", np.nan) for f in top_factors]
        names = [f.get("name", f"f_{i}") for i, f in enumerate(top_factors)]
        ax.scatter(depths, ics, c="#1f77b4", s=60, alpha=0.7, edgecolors="white")
        for dx, dy, nm in zip(depths, ics, names):
            ax.annotate(nm, (dx, dy), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, alpha=0.8)
        ax.axhline(y=0.02, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title("Tree Depth vs IC")
        ax.set_xlabel("Depth")
        ax.set_ylabel("IC")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No top factor data",
                ha="center", va="center", fontsize=12, color="#6c757d",
                transform=ax.transAxes)
        ax.set_title("Tree Depth vs IC")
        ax.set_axis_off()

    # ── 2. 行业 IC 分组柱状图 ──
    ax = axes[0, 1]
    industry_ic = report.get("industry_ic", {})
    if industry_ic and len(industry_ic) > 0:
        industries = list(industry_ic.keys())
        ics = [industry_ic[ind] for ind in industries]
        colors_bar = ["#2ca02c" if v > 0 else "#d62728" for v in ics]
        y_pos = range(len(industries))
        ax.barh(y_pos, ics, color=colors_bar, height=0.6, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(industries, fontsize=8)
        ax.axvline(x=0, color="#6c757d", linewidth=0.8)
        ax.set_title("Industry IC")
        ax.set_xlabel("IC")
        ax.grid(True, alpha=0.3, axis="x")
    else:
        ax.text(0.5, 0.5, "No industry IC data",
                ha="center", va="center", fontsize=12, color="#6c757d",
                transform=ax.transAxes)
        ax.set_title("Industry IC")
        ax.set_axis_off()

    # ── 3. 运算符使用率饼图 ──
    ax = axes[1, 0]
    op_usage = report.get("operator_usage", {})
    if op_usage and len(op_usage) > 0:
        ops = list(op_usage.keys())
        counts = list(op_usage.values())
        wedges, texts, autotexts = ax.pie(counts, labels=ops, autopct="%1.1f%%",
                                          startangle=140, textprops={"fontsize": 8})
        for at in autotexts:
            at.set_fontsize(7)
        ax.set_title("Operator Usage")
    else:
        ax.text(0.5, 0.5, "No operator usage data",
                ha="center", va="center", fontsize=12, color="#6c757d",
                transform=ax.transAxes)
        ax.set_title("Operator Usage")
        ax.set_axis_off()

    # ── 4. 因子冗余度 ──
    ax = axes[1, 1]
    high_corr = report.get("high_corr_count", None)
    if high_corr is not None:
        color = "#d62728" if high_corr > 5 else "#2ca02c"
        ax.text(0.5, 0.55, str(int(high_corr)), ha="center", va="center",
                fontsize=64, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.25, "High-Correlation Pairs", ha="center", va="center",
                fontsize=12, color="#6c757d", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No redundancy data",
                ha="center", va="center", fontsize=12, color="#6c757d",
                transform=ax.transAxes)
    ax.set_title("Factor Redundancy")
    ax.set_axis_off()

    fig.suptitle(f"Round Detail", fontsize=14, fontweight="bold", y=0.98)
    _save_or_show(fig, save_path)


# ═══════════════════════════════════════════════════════════════════════
# 终端进度条
# ═══════════════════════════════════════════════════════════════════════

def plot_terminal_summary(db: dict):
    """终端 ANSI 彩色单行进度总结，针对最新轮次。

    格式:
      轮  12 [████████░░░░░░░░░░░░░░░░░░░░] IC=+0.0523 适应度=+0.185 年化=+23.4% MDD=12.1%

    颜色规则:
      IC: 绿 |IC|>0.04 / 黄 >0.02 / 红 否则
      适应度: 绿 >0.2 / 黄 >0 / 红 否则
      年化: 绿 >0 / 红 否则
      MDD: 绿 <0.15 / 黄 <0.25 / 红 否则
    """
    history = db.get("history", []) if isinstance(db, dict) else []

    if not history:
        print("No evolution rounds completed yet.")
        return

    total_rounds = db.get("rounds", len(history))
    latest = history[-1]
    rn = latest.get("round", len(history))

    # ── 进度条：30 字符宽，当前 decade ──
    bar_width = 30
    current_decade = (rn - 1) // 10
    decade_start = current_decade * 10 + 1
    decade_pos = rn - decade_start
    filled = min(decade_pos, bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    # ── 指标 ──
    ic_val = _safe_float(latest, "best_ic", 0.0)
    fit_val = _safe_float(latest, "best_fitness", 0.0)
    ann_val = _safe_float(latest, "bt_annual", 0.0)
    mdd_val = _safe_float(latest, "bt_mdd", 0.0)

    # ── 颜色判断 ──
    ic_color = _color_for_ic(ic_val)
    fit_color = _color_for_fitness(fit_val)
    ann_color = GREEN if ann_val > 0 else RED
    mdd_color = GREEN if mdd_val < 0.15 else (YELLOW if mdd_val < 0.25 else RED)

    line = (
        f" 轮{rn:4d} [{bar}] "
        f"IC={ic_color}{ic_val:+.4f}{RESET} "
        f"适应度={fit_color}{fit_val:+.3f}{RESET} "
        f"年化={ann_color}{ann_val:+.1%}{RESET} "
        f"MDD={mdd_color}{mdd_val:.1%}{RESET}"
    )
    print(line)


# ═══════════════════════════════════════════════════════════════════════
# 内部辅助函数
# ═══════════════════════════════════════════════════════════════════════

def _safe_float(entry: dict, key: str, default: float = 0.0) -> float:
    """安全取出浮点数，处理缺失键和 NaN。"""
    val = entry.get(key, None)
    if val is None:
        return default
    try:
        val = float(val)
        if np.isnan(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def _color_for_ic(ic_val: float) -> str:
    """IC 颜色：|IC|>0.04 绿色，>0.02 黄色，其余红色。"""
    if abs(ic_val) > 0.04:
        return GREEN
    elif abs(ic_val) > 0.02:
        return YELLOW
    else:
        return RED


def _color_for_fitness(fit_val: float) -> str:
    """适应度颜色：>0.2 绿色，>0 黄色，其余红色。"""
    if fit_val > 0.2:
        return GREEN
    elif fit_val > 0:
        return YELLOW
    else:
        return RED


def plot_live_dashboard(db: dict):
    """实时终端仪表盘 — 多面板 ANSI 彩色面板，替代简单的单行进度条。

    每轮结束后调用，显示：
    1. 适应度/IC/年化趋势 sparkline
    2. 最近 N 轮回测数据表格
    3. 数据源覆盖率摘要
    """
    history = db.get("history", [])
    if not history:
        print(f"{YELLOW}  (no data yet){RESET}")
        return

    latest = history[-1]
    last_n = history[-8:]  # show last 8 rounds

    # ── Sparkline helper ──
    def sparkline(values, width=20):
        if len(values) < 2:
            return " " * width
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return "─" * width
        chars = " ▁▂▃▄▅▆▇█"
        result = []
        for v in values:
            idx = int((v - vmin) / (vmax - vmin) * 8)
            result.append(chars[min(idx, 8)])
        # right-align to width
        s = "".join(result)
        if len(s) < width:
            s = " " * (width - len(s)) + s
        return s[-width:]

    def mini_table(rows, headers, col_widths):
        lines = []
        # header
        hdr = " │ ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        lines.append(f"    {hdr}")
        lines.append(f"    {'─' * len(hdr)}")
        for row in rows:
            line = " │ ".join(str(r).ljust(w) for r, w in zip(row, col_widths))
            lines.append(f"    {line}")
        return "\n".join(lines)

    print()
    print(f"  ╔{'═'*58}╗")
    print(f"  ║ {GREEN}v4.0 进化仪表盘{RESET} — 第 {latest['round']} 轮{' ' * (42 - len(str(latest['round']))) }║")
    print(f"  ╠{'═'*58}╣")

    # ── 趋势 sparklines ──
    ic_vals = [abs(_safe_float(h, "best_ic")) for h in last_n]
    ann_vals = [_safe_float(h, "bt_annual") for h in last_n]
    mdd_vals = [_safe_float(h, "bt_mdd") for h in last_n]
    fit_vals = [_safe_float(h, "best_fitness") for h in last_n]

    cur_ic = abs(_safe_float(latest, "best_ic"))
    cur_ann = _safe_float(latest, "bt_annual")
    cur_mdd = _safe_float(latest, "bt_mdd")
    cur_fit = _safe_float(latest, "best_fitness")

    print(f"  ║  适应度 {_color_for_fitness(cur_fit)}{cur_fit:+.3f}{RESET}  {sparkline(fit_vals)}")
    print(f"  ║  训练IC  {_color_for_ic(cur_ic)}{cur_ic:+.4f}{RESET}  {sparkline(ic_vals)}")
    print(f"  ║  年化    {GREEN if cur_ann > 0 else RED}{cur_ann:+.1%}{RESET}  {sparkline(ann_vals)}")
    print(f"  ║  MDD     {GREEN if cur_mdd < 0.15 else YELLOW if cur_mdd < 0.25 else RED}{cur_mdd:.1%}{RESET}  {sparkline([-v for v in mdd_vals])}")

    # ── 最近轮回测表格 ──
    print(f"  ╠{'═'*58}╣")
    table_rows = []
    for h in last_n:
        table_rows.append([
            f"R{h['round']}",
            f"{_safe_float(h,'bt_annual'):+.1%}",
            f"{_safe_float(h,'bt_mdd'):.0%}",
            f"{_safe_float(h,'bt_sharpe'):+.2f}",
            f"{_safe_float(h,'bt_wr'):.0%}",
            str(int(_safe_float(h, 'bt_trades'))),
        ])
    print(mini_table(table_rows,
                     ["轮", "年化", "MDD", "夏普", "胜率", "笔数"],
                     [4, 7, 5, 6, 5, 5]))

    # ── 当前轮详情 ──
    n_valid = latest.get("n_valid", "?")
    n_ind = latest.get("n_individuals", "?")
    ndcg = _safe_float(latest, "ml_ndcg")
    print(f"  ╠{'═'*58}╣")
    print(f"  ║  有效个体: {n_valid}/{n_ind}  │  ML NDCG: {ndcg:.4f}  │  耗时: {latest.get('round', '?')}轮")

    # Best factor
    top_factors = latest.get("top_factors", [])
    if top_factors:
        name = top_factors[0].get("name", "?")[:50]
        print(f"  ║  最佳: {name}")

    print(f"  ╚{'═'*58}╝")
    print()


def _save_or_show(fig: plt.Figure, save_path: str | None):
    """保存或显示图表。"""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()
