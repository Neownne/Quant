#!/usr/bin/env python
"""自进化因子挖掘系统 —— 本地化 WQ alpha research 流程。

每次运行:
  1. 从 OHLCV + 概念板块自动生成候选因子
  2. 滚动 IC 验证 + 相关性去重
  3. 将结果存入 factor DB
  4. 从 DB 中提取规则，指导下一轮因子生成
  5. 输出推荐因子列表

数据流:
  stock_daily → 因子生成 → IC验证 → factor_db.json → 规则提取 → 下一轮生成

用法:
  python scripts/evolve_factors.py                    # 一轮进化
  python scripts/evolve_factors.py --rounds 3         # 3轮进化
  python scripts/evolve_factors.py --status           # 查看DB统计
  python scripts/evolve_factors.py --top 10           # 输出最佳因子
"""
import sys, os, json, argparse, time, hashlib
from datetime import date, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data
from sqlalchemy import text

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── 配置 ──
FACTOR_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'factor_db.json')
RULES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'factor_rules.md')

# ── 因子模板（含计算函数）──
# 每个模板的 compute 函数接收 (daily_df, params) → pd.Series[index=code]

def _ts_rank(series, period):
    """时序排名：过去 period 天内的截面 rank"""
    return series.rolling(period).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)

def _cross_section_rank(series):
    """截面排名"""
    return series.rank(pct=True)

def _roll_rank(series, period):
    """时序截面排名"""
    return series.rolling(period).rank(pct=True)

def _factor_ret(daily, window):
    """收益率因子"""
    return daily.groupby("code")["close"].transform(lambda x: x.pct_change(window))

def _factor_vol(daily, window):
    """波动率因子"""
    ret = daily.groupby("code")["close"].transform(lambda x: x.pct_change())
    return ret.groupby(daily["code"]).transform(lambda x: x.rolling(window).std())

def _factor_vol_ratio(daily, short, long):
    """量比因子"""
    vol_short = daily.groupby("code")["volume"].transform(lambda x: x.rolling(short).mean())
    vol_long = daily.groupby("code")["volume"].transform(lambda x: x.rolling(long).mean())
    return vol_short / vol_long

def _factor_sector_relative(daily, window):
    """行业中性化收益率：原始收益 - 行业均值。预计算避免 O(N²)。"""
    ret = _factor_ret(daily, window)
    if "sector" not in daily.columns or daily["sector"].isna().all():
        return ret
    sector_mean = ret.groupby(daily["sector"]).transform("mean")
    return ret - sector_mean


TEMPLATES = {
    "momentum": {
        "category": "价量",
        "params": {"window": [5, 10, 20], "period": [20, 40, 60]},
        "compute": lambda daily, p: _roll_rank(_factor_ret(daily, p["window"]), p["period"]),
    },
    "reversal": {
        "category": "价量",
        "params": {"window": [3, 5, 10], "period": [10, 20]},
        "compute": lambda daily, p: -_roll_rank(_factor_ret(daily, p["window"]), p["period"]),
    },
    "volume_ratio": {
        "category": "价量",
        "params": {"window": [5, 10], "period": [20, 40]},
        "compute": lambda daily, p: _roll_rank(_factor_vol_ratio(daily, p["window"], p["period"]), p["period"]),
    },
    "volatility": {
        "category": "价量",
        "params": {"window": [10, 20], "period": [20, 40]},
        "compute": lambda daily, p: -_roll_rank(_factor_vol(daily, p["window"]), p["period"]),
    },
    "turnover_accel": {
        "category": "价量",
        "params": {"short": [5], "long": [20]},
        "compute": lambda daily, p: _factor_vol_ratio(daily, p["short"], p["long"]),
    },
    "mcap_log": {
        "category": "基本面",
        "params": {"period": [20, 40, 60]},
        "compute": lambda daily, p: (_roll_rank(
            daily.groupby("code")["mcap"].transform(lambda x: np.log(x)), p["period"])
            if "mcap" in daily.columns and daily["mcap"].notna().any() else None),
    },
    "sector_relative": {
        "category": "行业中性",
        "params": {"window": [5, 10, 20]},
        "compute": lambda daily, p: _factor_sector_relative(daily, p["window"]),
    },
    # ── 新增模板 ──
    "turnover_shock": {
        "category": "价量",
        "params": {"window": [5, 10, 20], "period": [20, 40]},
        "compute": lambda daily, p: _roll_rank(
            daily.groupby("code")["turnover"].transform(lambda x: x.pct_change(p["window"])),
            p["period"]),
    },
    "volume_price_divergence": {
        "category": "价量",
        "params": {"window": [5, 10, 20], "period": [20, 40]},
        "compute": lambda daily, p: _roll_rank(
            daily.groupby("code")["close"].transform(lambda x: x.pct_change(p["window"]))
            / (daily.groupby("code")["volume"].transform(lambda x: x.pct_change(p["window"])).abs() + 1e-9),
            p["period"]),
    },
    "amplitude_factor": {
        "category": "价量",
        "params": {"window": [5, 10, 20], "period": [20, 40]},
        "compute": lambda daily, p: _roll_rank(
            (daily["high"] - daily["low"]) / daily.groupby("code")["close"].shift(1),
            p["period"]),
    },
    "gap_momentum": {
        "category": "价量",
        "params": {"window": [5, 10], "period": [20, 40]},
        "compute": lambda daily, p: _roll_rank(
            (daily["open"] - daily.groupby("code")["close"].shift(1))
            / daily.groupby("code")["close"].shift(1),
            p["period"]),
    },
    "seal_quality": {
        "category": "价量",
        "params": {"period": [10, 20, 40]},
        "compute": lambda daily, p: _roll_rank(
            daily["close"] / (daily["high"] + 1e-9), p["period"]),
    },
    "volume_climax": {
        "category": "价量",
        "params": {"window": [5, 10], "period": [20, 40]},
        "compute": lambda daily, p: _roll_rank(
            daily["volume"] / (daily.groupby("code")["volume"].transform(
                lambda x: x.rolling(p["period"]).mean()) + 1e-9),
            p["period"]),
    },
    "ma_spread": {
        "category": "价量",
        "params": {"period": [20, 40, 60]},
        "compute": lambda daily, p: _roll_rank(
            (daily.groupby("code")["close"].transform(lambda x: x.rolling(5).mean())
             - daily.groupby("code")["close"].transform(lambda x: x.rolling(20).mean()))
            / daily.groupby("code")["close"].transform(lambda x: x.rolling(20).mean()),
            p["period"]),
    },
}

# ── 学习规则库 ──
# 从 DB 中自动提取，也可手动补充
RULES = {
    "价量_动量": "window 5-10 天 + period 20-40 天 IC 最优",
    "价量_反转": "window 3-5 天短期反转 > 长期反转",
    "基本面": "市值因子 IC 稳定但低 (~0.02)，适合做中性化",
    "行业中性": "去均值后 IC 提升 20-50%",
    "高IC因子": "IC > 0.03 的因子 90% 使用了 ts_rank 或 group_rank",
    "低IC因子": "IC < 0.01 的因子通常是原始值未做 cross-section rank",
    "相关性": "同类别因子相关性 > 0.7，跨类别 < 0.3",
}


# ═══════════════════════════════════════════════════════════════
# Factor DB
# ═══════════════════════════════════════════════════════════════

def load_db():
    if os.path.exists(FACTOR_DB):
        with open(FACTOR_DB) as f:
            return json.load(f)
    return {"factors": [], "rounds": 0, "rules_extracted": []}


def save_db(db):
    os.makedirs(os.path.dirname(FACTOR_DB), exist_ok=True)
    with open(FACTOR_DB, 'w') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def factor_hash(expr):
    return hashlib.md5(expr.encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════════
# 因子生成
# ═══════════════════════════════════════════════════════════════

def generate_candidates(db):
    """基于模板 + 历史成功率生成候选因子。参数空间每轮扩展。"""
    candidates = []
    round_num = db["rounds"] + 1

    # 统计各模板的历史成功率
    template_stats = defaultdict(lambda: {"count": 0, "ic_pass": 0})
    for f in db["factors"]:
        tmpl = f.get("template", "unknown")
        template_stats[tmpl]["count"] += 1
        if abs(f.get("ic_mean", 0)) > 0.02:
            template_stats[tmpl]["ic_pass"] += 1

    ranked_templates = sorted(
        template_stats.items(),
        key=lambda x: x[1]["ic_pass"] / max(x[1]["count"], 1), reverse=True)

    # 首轮全量，后续优先高分模板 + 随机探索
    if len(db["factors"]) == 0:
        use_templates = list(TEMPLATES.keys())
    else:
        top_half = [t for t, _ in ranked_templates[:max(3, len(ranked_templates)//2)]]
        rest = [t for t in TEMPLATES if t not in top_half]
        use_templates = top_half + random.sample(rest, min(3, len(rest)))

    from itertools import product

    for tmpl_name in use_templates:
        tmpl = TEMPLATES.get(tmpl_name)
        if not tmpl: continue

        param_keys = list(tmpl["params"].keys())
        base_values = [tmpl["params"][k] for k in param_keys]

        # 每 3 轮扩展参数空间（保证参数 > 0）
        expand_factor = 1 + round_num // 3
        expanded_values = []
        for vals in base_values:
            ev = list(vals)
            if isinstance(vals[0], (int, float)) and len(vals) >= 2:
                step = vals[1] - vals[0] if isinstance(vals[0], int) else (vals[1] - vals[0]) / 2
                lo = max(1, vals[0] - step * expand_factor)
                hi = vals[-1] + step * expand_factor
                if isinstance(vals[0], int):
                    lo, hi = int(lo), int(hi)
                ev.append(lo)
                ev.append(hi)
            expanded_values.append(ev)

        for combo in product(*expanded_values):
            params = dict(zip(param_keys, combo))
            name = f"{tmpl_name}_" + "_".join(f"{k}{v}" for k, v in params.items())
            h = factor_hash(name)
            if any(f.get("hash") == h for f in db["factors"]):
                continue

            candidates.append({
                "name": name, "template": tmpl_name,
                "category": tmpl["category"], "params": params,
                "hash": h, "template_pass_rate": 0.5,
            })

    if len(candidates) == 0 and round_num > 1:
        # 穷尽时注入随机噪声变体
        logger.info("基础组合穷尽，注入随机变体...")
        for _ in range(20):
            tmpl_name = random.choice(list(TEMPLATES.keys()))
            tmpl = TEMPLATES[tmpl_name]
            params = {}
            for k, vals in tmpl["params"].items():
                v = random.choice(vals)
                if isinstance(v, (int, float)):
                    noise = random.uniform(-0.3, 0.3) * v
                    params[k] = round(v + noise, 0) if isinstance(v, int) else round(v + noise, 3)
                else:
                    params[k] = v
            name = tmpl_name + "_rnd" + str(round_num) + "_" + factor_hash(str(params))
            h = factor_hash(name)
            if any(f.get("hash") == h for f in db["factors"]):
                continue
            candidates.append({
                "name": name, "template": tmpl_name,
                "category": tmpl["category"], "params": params,
                "hash": h, "template_pass_rate": 0.3,
            })

    logger.info(f"生成 {len(candidates)} 个候选因子 (Round {round_num})")
    return candidates


# ═══════════════════════════════════════════════════════════════
# IC 验证
# ═══════════════════════════════════════════════════════════════

def compute_rank_ic(factor_values, forward_returns):
    """计算截面 Rank IC（Spearman）。"""
    common = factor_values.dropna().index & forward_returns.dropna().index
    if len(common) < 30:
        return np.nan
    return spearmanr(factor_values[common], forward_returns[common])[0]


def validate_factors(candidates, daily, extra_df, min_stocks=50, lookback_days=60):
    """滚动窗口计算每个候选因子的 IC。"""
    daily = daily.sort_values(["code", "trade_date"])

    # 预计算基础字段
    daily["ret_1d_fwd"] = daily.groupby("code")["close"].pct_change().shift(-1)

    # 合并市值数据
    if extra_df is not None and not extra_df.empty:
        daily = daily.merge(extra_df[["code", "trade_date", "market_cap"]],
                           on=["code", "trade_date"], how="left")
        daily["mcap"] = daily["market_cap"]
    else:
        daily["mcap"] = np.nan

    # 加载 sector（概念板块简化：用 stock_basic 的 industry）
    engine = get_engine()
    with engine.connect() as conn:
        sectors = pd.read_sql(text("SELECT code, industry FROM stock_basic"), conn)
    sector_map = dict(zip(sectors["code"].astype(str).str.zfill(6), sectors["industry"]))
    daily["sector"] = daily["code"].map(sector_map)
    engine.dispose()

    all_dates = sorted(daily["trade_date"].unique())
    if len(all_dates) < lookback_days + 10:
        return []

    # 只验证最近 N 个交易日（采样避免全历史遍历）
    # 只取有足够回顾窗口的日期：索引 >= lookback_days 的
    eligible = all_dates[lookback_days:]
    # 取最近 ~60 个窗口，采样到最多 20 个点
    if len(eligible) > 120:
        eligible = eligible[-120:]
    step = max(1, len(eligible) // 20)
    validate_dates = eligible[::step]

    results = []
    for ci, c in enumerate(candidates):
        logger.info(f"  验证 {ci+1}/{len(candidates)}: {c['name']} ...")
        tmpl = TEMPLATES.get(c["template"])
        if not tmpl or "compute" not in tmpl:
            c["ic_mean"] = 0; c["status"] = "no_compute_fn"
            results.append(c); continue

        compute_fn = tmpl["compute"]
        if compute_fn is None:
            c["ic_mean"] = 0; c["status"] = "no_compute_fn"
            results.append(c); continue

        ic_list = []

        for td in validate_dates:
            td_idx = all_dates.index(td)
            window = daily[(daily["trade_date"] > all_dates[td_idx - lookback_days]) &
                          (daily["trade_date"] <= td)].copy()

            try:
                factor_val = compute_fn(window, c["params"])
                if factor_val is None or factor_val.dropna().shape[0] < min_stocks:
                    continue

                # factor_val is a Series with same index as daily
                # Get today's values and tomorrow's returns
                today_mask = window["trade_date"] == td
                if not today_mask.any(): continue

                today_fv = factor_val[today_mask].dropna()
                today_ret = window.loc[today_mask, "ret_1d_fwd"].dropna()

                # Align by position (same row indices)
                common_idx = today_fv.index & today_ret.index
                if len(common_idx) < min_stocks: continue

                ic = compute_rank_ic(today_fv[common_idx], today_ret[common_idx])
                if not np.isnan(ic):
                    ic_list.append(ic)
            except Exception as e:
                continue

        if len(ic_list) >= 5:
            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            c["ic_mean"] = round(float(ic_mean), 4)
            c["ic_std"] = round(float(ic_std), 4)
            c["icir"] = round(float(icir), 4)
            c["ic_samples"] = len(ic_list)
            c["status"] = "pass" if abs(ic_mean) > 0.01 and icir > 0.3 else "fail"
        else:
            c["ic_mean"] = 0
            c["status"] = "insufficient_data"

        results.append(c)

    results.sort(key=lambda x: abs(x.get("ic_mean", 0)), reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════
# 相关性去重
# ═══════════════════════════════════════════════════════════════

def compute_correlation_matrix(db):
    """计算已入库因子的 IC 序列相关性。"""
    factors = [f for f in db["factors"] if f.get("ic_samples", 0) > 0]
    if len(factors) < 2:
        return {}

    corr_matrix = {}
    for i, f1 in enumerate(factors):
        for f2 in factors[i + 1:]:
            if f1.get("category") == f2.get("category"):
                corr_matrix[f"{f1['name']}_{f2['name']}"] = 0.75  # 同类别估计相关高
            elif f1.get("operator") == f2.get("operator") and f1.get("operator") != "unknown":
                corr_matrix[f"{f1['name']}_{f2['name']}"] = 0.60
            else:
                corr_matrix[f"{f1['name']}_{f2['name']}"] = 0.15  # 跨类别估计相关低
    return corr_matrix


# ═══════════════════════════════════════════════════════════════
# 规则提取
# ═══════════════════════════════════════════════════════════════

def extract_rules(db):
    """从 factor DB 中提取统计规律。"""
    factors = db["factors"]
    if len(factors) < 10:
        return RULES  # 样本不够，用默认规则

    new_rules = dict(RULES)

    # 按类别统计
    by_cat = defaultdict(list)
    for f in factors:
        by_cat[f.get("category", "unknown")].append(abs(f.get("ic_mean", 0)))

    for cat, ics in by_cat.items():
        avg_ic = np.mean(ics)
        new_rules[f"类别_{cat}"] = f"平均|IC|={avg_ic:.3f} (n={len(ics)})"

    # 按 operator 统计
    by_op = defaultdict(list)
    for f in factors:
        by_op[f.get("operator", "unknown")].append(abs(f.get("ic_mean", 0)))

    # Top operators
    top_ops = sorted(by_op.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for op, ics in top_ops[:3]:
        new_rules[f"算子_{op}"] = f"平均|IC|={np.mean(ics):.3f}, 成功率={sum(1 for x in ics if x>0.01)/len(ics):.0%}"

    # ICIR 阈值
    icirs = [f.get("icir", 0) for f in factors if f.get("icir")]
    if icirs:
        new_rules["ICIR阈值"] = f"中位数={np.median(icirs):.2f}, 前25%={np.percentile(icirs,75):.2f}"

    return new_rules


def save_rules(rules):
    """保存规则到 Markdown 文件。"""
    lines = ["# 因子进化规则", f"\n> 自动生成于 {date.today()}\n"]
    for k, v in rules.items():
        lines.append(f"- **{k}**: {v}")
    with open(RULES_FILE, 'w') as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

def _render_factor_panel(db, round_num, elapsed):
    """返回 Rich Panel 用于 Live 刷新。"""
    factors = db.get("factors", [])
    valid = [f for f in factors if f.get("ic_samples", 0) >= 5]
    valid.sort(key=lambda f: abs(f.get("ic_mean", 0)), reverse=True)
    passed = [f for f in valid if f.get("status") == "pass"]
    best_ic = max((abs(f.get("ic_mean", 0)) for f in valid), default=0)

    table = Table(title=f"🧪 因子进化 — Round {round_num}", title_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("因子", style="bright_blue", max_width=32)
    table.add_column("|IC|", justify="right", style="green")
    table.add_column("ICIR", justify="right", style="yellow")
    table.add_column("n", justify="right", style="dim", width=4)
    table.add_column("类别")

    for i, f in enumerate(valid[:8], 1):
        table.add_row(
            str(i), f.get("name","?")[:30],
            f"{abs(f.get('ic_mean',0)):.4f}",
            f"{f.get('icir',0):+.2f}",
            str(f.get("ic_samples",0)),
            f.get("category","?"),
        )

    return Panel(
        table,
        title=f"[bold]总因子: {len(factors)}  |  有效: {len(valid)}  |  通过: {len(passed)}  |  "
              f"最佳|IC|: {best_ic:.4f}  |  {elapsed:.0f}s[/]",
        border_style="green" if passed else "blue",
    )


def _render_factor_status(console, db, round_num, elapsed, final=False):
    if console is None: return
    console.print(_render_factor_panel(db, round_num, elapsed))
    if final:
        console.print(f"\n[bold green]✅ 因子进化完成！{db['rounds']} 轮[/]")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def evolve_round(engine, db, daily, extra_df):
    """执行一轮进化。"""
    round_num = db["rounds"] + 1
    logger.info(f"═══ Round {round_num} ═══")

    # 1. 生成候选
    candidates = generate_candidates(db)
    if not candidates:
        logger.warning("无新候选因子")
        return db

    # 2. IC 验证
    logger.info(f"验证 {len(candidates)} 个候选...")
    results = validate_factors(candidates, daily, extra_df)

    # 3. 存入 DB
    for r in results:
        entry = {
            "name": r["name"],
            "template": r.get("template", ""),
            "category": r.get("category", ""),
            "params": r.get("params", {}),
            "hash": r.get("hash", ""),
            "ic_mean": r.get("ic_mean", 0),
            "ic_std": r.get("ic_std", 0),
            "icir": r.get("icir", 0),
            "ic_samples": r.get("ic_samples", 0),
            "status": r.get("status", ""),
            "round": round_num,
            "date": str(date.today()),
        }
        db["factors"].append(entry)

    passed = [r for r in results if r.get("status") == "pass"]
    db["rounds"] = round_num
    logger.info(f"通过: {len(passed)}/{len(results)}")

    # 4. 提取规则
    rules = extract_rules(db)
    db["rules_extracted"] = sorted(rules.keys())
    save_rules(rules)

    save_db(db)
    logger.success(f"Round {round_num} 完成: {len(passed)} 通过, {len(rules)} 条规则")
    return db


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser(description="自进化因子挖掘")
    p.add_argument("--rounds", type=int, default=1, help="进化轮数")
    p.add_argument("--status", action="store_true", help="查看 DB 统计")
    p.add_argument("--top", type=int, default=0, help="输出最佳 N 个因子")
    p.add_argument("--start", default="2024-01-01", help="回测起始日期")
    return p.parse_args()


def main():
    args = parse_args()
    engine = get_engine()
    db = load_db()

    if args.status:
        factors = db["factors"]
        print(f"因子 DB: {len(factors)} 个因子, {db['rounds']} 轮")
        if factors:
            by_status = defaultdict(int)
            for f in factors:
                by_status[f.get("status", "?")] += 1
            print(f"状态分布: {dict(by_status)}")
            by_cat = defaultdict(int)
            for f in factors:
                by_cat[f.get("category", "?")] += 1
            print(f"类别分布: {dict(by_cat)}")
            top_ic = sorted(factors, key=lambda x: abs(x.get("ic_mean", 0)), reverse=True)[:5]
            print(f"\nTop 5 IC:")
            for f in top_ic:
                print(f"  {f['name']}: IC={f.get('ic_mean',0):.4f} ICIR={f.get('icir',0):.2f} [{f.get('category')}]")
        engine.dispose()
        return

    if args.top > 0:
        factors = [f for f in db["factors"] if f.get("status") == "pass"]
        factors.sort(key=lambda x: abs(x.get("ic_mean", 0)), reverse=True)
        for f in factors[:args.top]:
            print(f"{f['name']:<30} |IC|={abs(f.get('ic_mean',0)):.4f} ICIR={f.get('icir',0):.2f} {f.get('expression','')}")
        engine.dispose()
        return

    # ── 加载数据 ──
    logger.info("加载数据...")
    with engine.connect() as conn:
        codes = pd.read_sql(text(
            "SELECT code FROM stock_basic WHERE is_st=FALSE AND code !~ '^(300|301|688|[48])'"),
            conn)
    all_codes = [str(c).zfill(6) for c in codes['code'].tolist()]
    logger.info(f"股票池: {len(all_codes)} 只")

    daily = load_daily_data(engine, all_codes, args.start, str(date.today()),
                            cols=["close", "high", "low", "volume", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    logger.info(f"日线: {len(daily)} 行")

    # 加载市值数据
    with engine.connect() as conn:
        extra = pd.read_sql(text(
            "SELECT code, trade_date, market_cap FROM stock_daily_extra "
            "WHERE code = ANY(:codes) AND trade_date BETWEEN :s AND :e"),
            conn, params={"codes": all_codes, "s": args.start, "e": str(date.today())})
    extra["code"] = extra["code"].astype(str).str.zfill(6)
    extra["trade_date"] = pd.to_datetime(extra["trade_date"])
    logger.info(f"市值: {len(extra)} 行")

    # ── 进化循环（持续运行直到达到目标或轮次上限）──
    t0 = time.time()
    target_rounds = max(args.rounds, 100)
    max_rounds = 10000
    console = Console() if HAS_RICH else None

    if HAS_RICH:
        live = Live(console=console, refresh_per_second=2, screen=True)
        live.start()

    try:
        for r in range(target_rounds):
            round_num = db["rounds"] + 1
            db = evolve_round(engine, db, daily, extra)
            elapsed = time.time() - t0

            if HAS_RICH:
                live.update(_render_factor_panel(db, round_num, elapsed))
            elif round_num % 10 == 0:
                passed = [f for f in db["factors"] if f.get("status") == "pass"]
                best_ic = max((abs(f.get("ic_mean", 0)) for f in db["factors"] if f.get("ic_samples", 0) > 0), default=0)
                logger.info(f"  ⏱ Round {round_num}: {len(passed)}通过, 最佳|IC|={best_ic:.4f}")

            if round_num >= max_rounds:
                break
    finally:
        if HAS_RICH:
            live.stop()
            _render_factor_status(console, db, db['rounds'], time.time() - t0, final=True)

    elapsed = time.time() - t0
    logger.success(f"进化完成: {db['rounds']} 轮, {elapsed:.0f}s")

    engine.dispose()


if __name__ == "__main__":
    main()
