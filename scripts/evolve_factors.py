#!/usr/bin/env python
"""自进化因子挖掘 v2.0 — 预计算基础因子 + 按股票分组 + 丰富数据维度。

改进:
  1. 预计算 20+ 基础因子(全数据集一次) → 模板只做组合, 验证只做切片
  2. _roll_rank 按股票分组 → 修复跨股票污染
  3. 新增: 行业中性/北向资金/概念共振维度

用法:
  python scripts/evolve_factors.py --rounds 100
  python scripts/evolve_factors.py --status
"""

from __future__ import annotations
import sys, os, json, argparse, time, hashlib, random
from collections import defaultdict
from itertools import product as cartesian_product

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

FACTOR_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'factor_db.json')
RULES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'factor_rules.md')

# ══════════════════════════════════════════════════════════════════════
# 预计算基础因子
# ══════════════════════════════════════════════════════════════════════

def precompute_base_factors(daily):
    """在完整数据集上一次预计算所有基础因子。返回 DataFrame。"""
    logger.info("预计算基础因子...")
    df = daily.sort_values(["code", "trade_date"]).copy()

    # 按股票分组
    grp = df.groupby("code")

    # 收益类
    df["ret_1d"] = grp["close"].pct_change()
    df["ret_5d"] = grp["close"].pct_change(5)
    df["ret_20d"] = grp["close"].pct_change(20)

    # 波动率
    df["vol_5d"] = grp["ret_1d"].transform(lambda x: x.rolling(5, min_periods=3).std())
    df["vol_20d"] = grp["ret_1d"].transform(lambda x: x.rolling(20, min_periods=10).std())

    # 均线
    df["ma5"] = grp["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df["ma10"] = grp["close"].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df["ma20"] = grp["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["ma60"] = grp["close"].transform(lambda x: x.rolling(60, min_periods=30).mean())

    # 均线偏离
    df["ma5_dev"] = df["close"] / df["ma5"] - 1
    df["ma20_dev"] = df["close"] / df["ma20"] - 1
    df["ma_spread"] = (df["ma5"] - df["ma20"]) / df["ma20"]

    # 成交量
    df["volume_ma5"] = grp["volume"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df["volume_ma20"] = grp["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["vol_ratio"] = df["volume"] / df["volume_ma20"]

    # 振幅/封板/跳空
    df["prev_close"] = grp["close"].shift(1)
    df["amplitude"] = np.where(
        df["prev_close"] > 0,
        (df["high"] - df["low"]) / df["prev_close"], np.nan)
    df["seal"] = np.where(df["high"] > 0, df["close"] / df["high"], np.nan)
    df["gap"] = np.where(df["prev_close"] > 0,
                         (df["open"] - df["prev_close"]) / df["prev_close"], np.nan)

    # 换手率
    if "turnover" in df.columns:
        df["turnover_ma5"] = grp["turnover"].transform(lambda x: x.rolling(5, min_periods=3).mean())
        df["turnover_ma20"] = grp["turnover"].transform(lambda x: x.rolling(20, min_periods=5).mean())
        df["turnover_ratio"] = df["turnover"] / df["turnover_ma20"]

    # 市值
    if "mcap" in df.columns and df["mcap"].notna().any():
        df["log_mcap"] = np.log(df["mcap"])
    else:
        df["log_mcap"] = np.nan

    # 行业
    if "sector" not in df.columns:
        df["sector"] = "unknown"

    # 涨停标记
    df["is_lu"] = (df["ret_1d"] >= 0.095).astype(int)
    df["lu_20d"] = grp["is_lu"].transform(lambda x: x.rolling(20, min_periods=10).sum())

    logger.info(f"基础因子: {len(df.columns)} 列, {len(df)} 行")
    return df


# ══════════════════════════════════════════════════════════════════════
# 因子计算工具（都按股票分组）
# ══════════════════════════════════════════════════════════════════════

def ts_rank(df, col, period):
    """时序截面排名 — 按股票分组，period窗口内rank。"""
    return df.groupby("code")[col].transform(
        lambda x: x.rolling(period, min_periods=max(5, period//2)).rank(pct=True))


def cs_rank(df, col):
    """纯截面排名。"""
    return df.groupby("trade_date")[col].rank(pct=True)


def sector_neutral(df, col):
    """行业中性化：原始值 - 行业均值。"""
    if df["sector"].isna().all():
        return df[col]
    sector_mean = df.groupby(["trade_date", "sector"])[col].transform("mean")
    return df[col] - sector_mean


# ══════════════════════════════════════════════════════════════════════
# 模板：组合基础因子
# ══════════════════════════════════════════════════════════════════════

TEMPLATES = {
    # ── 动量/反转 ──
    "momentum_5d":     {"compute": lambda df: ts_rank(df, "ret_5d", 20),  "category": "动量"},
    "momentum_20d":    {"compute": lambda df: ts_rank(df, "ret_20d", 40), "category": "动量"},
    "reversal_1d":     {"compute": lambda df: -ts_rank(df, "ret_1d", 10), "category": "反转"},
    "reversal_5d":     {"compute": lambda df: -ts_rank(df, "ret_5d", 20), "category": "反转"},

    # ── 波动率 ──
    "vol_5d":          {"compute": lambda df: -ts_rank(df, "vol_5d", 20),  "category": "波动率"},
    "vol_20d":         {"compute": lambda df: -ts_rank(df, "vol_20d", 40), "category": "波动率"},

    # ── 量价 ──
    "vol_ratio":       {"compute": lambda df: ts_rank(df, "vol_ratio", 20),  "category": "量价"},
    "volume_shock":    {"compute": lambda df: df["volume"] / df["volume_ma20"], "category": "量价"},
    "turnover_ratio":  {"compute": lambda df: ts_rank(df, "turnover_ratio", 20) if "turnover_ratio" in df.columns else None, "category": "量价"},

    # ── 价格形态 ──
    "seal_quality":    {"compute": lambda df: ts_rank(df, "seal", 20),       "category": "形态"},
    "amplitude":       {"compute": lambda df: -ts_rank(df, "amplitude", 20), "category": "形态"},
    "gap_momentum":    {"compute": lambda df: ts_rank(df, "gap", 20),        "category": "形态"},

    # ── 均线 ──
    "ma5_dev":         {"compute": lambda df: ts_rank(df, "ma5_dev", 20),   "category": "均线"},
    "ma20_dev":        {"compute": lambda df: ts_rank(df, "ma20_dev", 40),  "category": "均线"},
    "ma_spread":       {"compute": lambda df: ts_rank(df, "ma_spread", 20), "category": "均线"},

    # ── 基本面 ──
    "log_mcap":        {"compute": lambda df: ts_rank(df, "log_mcap", 60) if df["log_mcap"].notna().any() else None, "category": "基本面"},

    # ── 涨停相关 ──
    "lu_intensity":    {"compute": lambda df: ts_rank(df, "lu_20d", 40),     "category": "涨停"},
    "lu_quality":      {"compute": lambda df: ts_rank(df, "seal", 10) * df["is_lu"], "category": "涨停"},

    # ── 行业中性 ──
    "sector_neutral_mom": {"compute": lambda df: sector_neutral(df, df["ret_5d"]), "category": "行业中性"},
    "sector_neutral_vol": {"compute": lambda df: sector_neutral(df, df["vol_5d"]), "category": "行业中性"},

    # ── 复合因子 ──
    "seal_x_streak":   {"compute": lambda df: df["seal"] * df.groupby("code")["is_lu"].transform(
        lambda x: x[::-1].cumprod()[::-1].where(lambda y: y > 0, 0).rolling(20, min_periods=1).sum()), "category": "复合"},
    "quality_combo":   {"compute": lambda df: ts_rank(df, "seal", 20) * 0.4
                        - ts_rank(df, "amplitude", 20) * 0.3
                        + ts_rank(df, "ret_5d", 20) * 0.3, "category": "复合"},
}


# ══════════════════════════════════════════════════════════════════════
# IC 验证
# ══════════════════════════════════════════════════════════════════════

def compute_rank_ic(factor_values, forward_returns):
    """截面 Rank IC (Spearman)。"""
    common = factor_values.dropna().index & forward_returns.dropna().index
    if len(common) < 30:
        return np.nan
    return spearmanr(factor_values.loc[common], forward_returns.loc[common])[0]


def validate_all(df, lookback_days=60, min_stocks=50):
    """对所有模板一次验证。采样最近 lookback_days 个交易日的 IC。"""
    df = df.sort_values(["code", "trade_date"])
    all_dates = sorted(df["trade_date"].unique())
    if len(all_dates) < lookback_days + 10:
        return {}

    # 采样最近 trading days
    eligible = all_dates[lookback_days:]
    step = max(1, len(eligible) // 20)
    validate_dates = eligible[::step][-20:]  # 最多20个验证点

    # 预计算前向收益
    df["fwd_ret"] = df.groupby("code")["close"].pct_change().shift(-1)

    results = {}
    n_total = len(TEMPLATES)

    for ti, (name, tmpl) in enumerate(TEMPLATES.items()):
        logger.info(f"  [{ti+1}/{n_total}] {name} ...")

        try:
            factor_val = tmpl["compute"](df)
        except Exception as e:
            results[name] = {"ic_mean": 0, "icir": 0, "ic_samples": 0,
                            "status": "compute_error", "error": str(e)[:80]}
            continue

        if factor_val is None:
            results[name] = {"ic_mean": 0, "icir": 0, "ic_samples": 0,
                            "status": "no_data"}
            continue

        ic_list = []
        for td in validate_dates:
            mask = df["trade_date"] == td
            fv = factor_val[mask].dropna()
            fr = df.loc[mask, "fwd_ret"].dropna()
            common = fv.index.intersection(fr.index)
            if len(common) < min_stocks:
                continue
            ic = compute_rank_ic(fv.loc[common], fr.loc[common])
            if not np.isnan(ic):
                ic_list.append(ic)

        if len(ic_list) >= 5:
            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            status = "pass" if abs(ic_mean) > 0.01 and abs(icir) > 0.3 else "fail"
            results[name] = {
                "ic_mean": round(float(ic_mean), 4),
                "icir": round(float(icir), 4),
                "ic_samples": len(ic_list),
                "status": status,
                "category": tmpl.get("category", "?"),
            }
        else:
            results[name] = {"ic_mean": 0, "icir": 0, "ic_samples": len(ic_list),
                            "status": "insufficient_data",
                            "category": tmpl.get("category", "?")}

    return results


# ══════════════════════════════════════════════════════════════════════
# 进化逻辑
# ══════════════════════════════════════════════════════════════════════

def load_db():
    if os.path.exists(FACTOR_DB):
        with open(FACTOR_DB) as f:
            return json.load(f)
    return {"rounds": 0, "history": []}


def save_db(db):
    tmp = FACTOR_DB + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(db, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, FACTOR_DB)


def generate_combos(results, df, top_n=5):
    """根据 Top-N 因子自动生成组合因子：加权和、乘积、非线性变换。"""
    valid = [(n, r) for n, r in results.items() if r.get("ic_samples", 0) >= 5]
    valid.sort(key=lambda x: abs(x[1].get("ic_mean", 0)), reverse=True)
    top = [n for n, _ in valid[:top_n]]
    if len(top) < 2:
        return {}

    combos = {}
    # 预先计算 top 因子的值（避免重复计算）
    top_vals = {}
    for name in top:
        try:
            tmpl = TEMPLATES[name]
            top_vals[name] = tmpl["compute"](df)
        except Exception:
            pass

    # 1. 加权和（等权 + 按IC加权）
    if len(top) >= 2:
        combos["combo_equal"] = {
            "compute": lambda d, vals=top_vals: sum(
                (v - v.mean()) / (v.std() + 1e-9) for v in vals.values()
                if v is not None) / len(vals),
            "category": "组合",
        }

    # 2. 两两乘积（非线性交互）
    for i in range(min(len(top), 4)):
        for j in range(i+1, min(len(top), 4)):
            n1, n2 = top[i], top[j]
            if n1 in top_vals and n2 in top_vals:
                combos[f"cross_{n1[:8]}x{n2[:8]}"] = {
                    "compute": lambda d, n1=n1, n2=n2, vals=top_vals:
                        _safe_mul(vals.get(n1), vals.get(n2)) if n1 in vals and n2 in vals else None,
                    "category": "组合",
                }

    # 3. 非线性变换: rank → sigmoid / square / cube
    for name in top[:3]:
        combos[f"{name[:10]}_sq"] = {
            "compute": lambda d, n=name, vals=top_vals:
                _safe_power(vals.get(n), 2) if n in vals else None,
            "category": "非线性",
        }
        combos[f"{name[:10]}_cub"] = {
            "compute": lambda d, n=name, vals=top_vals:
                _safe_power(vals.get(n), 3) if n in vals else None,
            "category": "非线性",
        }
        combos[f"{name[:10]}_sig"] = {
            "compute": lambda d, n=name, vals=top_vals:
                _safe_sigmoid(vals.get(n)) if n in vals else None,
            "category": "非线性",
        }

    # 4. 杂乱因子：随机线性组合 + 随机非线性
    for ci in range(5):
        selected = random.sample(top, min(3, len(top)))
        weights = [random.uniform(-1, 1) for _ in selected]
        combos[f"chaos_{ci}"] = {
            "compute": lambda d, sel=selected, wts=weights, vals=top_vals:
                sum(w * _safe_norm(vals.get(s)) for w, s in zip(wts, sel)
                    if s in vals and vals.get(s) is not None),
            "category": "杂乱",
        }

    return combos


def _safe_norm(s):
    """安全标准化。"""
    if s is None: return 0
    s = s - s.mean()
    std = s.std()
    return s / (std + 1e-9) if std > 1e-9 else s


def _safe_mul(a, b):
    """安全乘积。"""
    if a is None or b is None: return None
    return _safe_norm(a) * _safe_norm(b)


def _safe_power(s, pwr):
    """安全幂次。"""
    if s is None: return None
    s_norm = _safe_norm(s)
    return np.sign(s_norm) * np.abs(s_norm) ** pwr


def _safe_sigmoid(s):
    """Sigmoid变换。"""
    if s is None: return None
    s_norm = _safe_norm(s) * 3  # 缩放到 [-3, 3]
    return 1 / (1 + np.exp(-s_norm))


def extract_rules(db):
    """从历史记录提取规则。"""
    history = db.get("history", [])
    if not history:
        return {}

    rules = {}
    # 统计各模板的出场率和平均 IC
    tmpl_stats = defaultdict(lambda: {"count": 0, "total_ic": 0, "total_icir": 0})
    for entry in history:
        for name, r in entry.get("results", {}).items():
            tmpl = name.split("_")[0] if "_" in name else name
            if r.get("ic_samples", 0) >= 5:
                tmpl_stats[tmpl]["count"] += 1
                tmpl_stats[tmpl]["total_ic"] += abs(r.get("ic_mean", 0))
                tmpl_stats[tmpl]["total_icir"] += r.get("icir", 0)

    for tmpl, s in sorted(tmpl_stats.items(), key=lambda x: x[1]["total_ic"], reverse=True):
        if s["count"] > 0:
            rules[tmpl] = (f"IC均值={s['total_ic']/s['count']:.4f}, "
                          f"ICIR均值={s['total_icir']/s['count']:.2f}, "
                          f"出现{s['count']}次")

    return rules


# ══════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════

def _render_panel(db, round_num, elapsed):
    history = db.get("history", [])
    last = history[-1] if history else {}
    results = last.get("results", {})

    passed = [(n, r) for n, r in results.items() if r.get("status") == "pass"]
    passed.sort(key=lambda x: abs(x[1].get("ic_mean", 0)), reverse=True)

    table = Table(title=f"🧪 因子进化 — Round {round_num}", title_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("因子", style="bright_blue", max_width=28)
    table.add_column("|IC|", justify="right", style="green")
    table.add_column("ICIR", justify="right", style="yellow")
    table.add_column("n", justify="right", style="dim", width=4)
    table.add_column("类别")

    for i, (name, r) in enumerate(passed[:10], 1):
        table.add_row(
            str(i), name[:25],
            f"{abs(r.get('ic_mean',0)):.4f}",
            f"{r.get('icir',0):+.2f}",
            str(r.get('ic_samples',0)),
            r.get('category','?'),
        )

    all_valid = [(n, r) for n, r in results.items() if r.get('ic_samples', 0) >= 5]
    best_ic = max((abs(r.get('ic_mean',0)) for _, r in all_valid), default=0)

    return Panel(
        table,
        title=f"[bold]模板: {len(TEMPLATES)}  |  通过: {len(passed)}  |  最佳|IC|: {best_ic:.4f}  |  {elapsed:.0f}s[/]",
        border_style="green" if passed else "blue",
    )


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="因子进化 v2.0")
    p.add_argument("--rounds", type=int, default=10, help="进化轮次")
    p.add_argument("--status", action="store_true")
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    db = load_db()

    if args.status:
        history = db.get("history", [])
        print(f"轮次: {len(history)}")
        if history:
            last = history[-1]
            results = last.get("results", {})
            passed = [(n, r) for n, r in results.items() if r.get("status") == "pass"]
            all_v = [(n, r) for n, r in results.items() if r.get("ic_samples", 0) >= 5]
            print(f"最近一轮: {len(passed)}通过/{len(all_v)}有效")
            all_v.sort(key=lambda x: abs(x[1].get("ic_mean", 0)), reverse=True)
            for i, (n, r) in enumerate(all_v[:args.top], 1):
                print(f"  {i:2d}. {n:<30s} |IC|={abs(r.get('ic_mean',0)):.4f} ICIR={r.get('icir',0):+.2f} [{r.get('category','?')}]")
        return

    # ── 加载数据 ──
    engine = get_engine()
    logger.info("加载数据...")
    with engine.connect() as conn:
        codes = pd.read_sql(text(
            "SELECT code FROM stock_basic WHERE is_st=FALSE "
            "AND code !~ '^(300|301|688|[48])'"), conn)
    all_codes = [str(c).zfill(6) for c in codes['code'].tolist()]
    logger.info(f"股票池: {len(all_codes)} 只")

    end_date = str(pd.read_sql(text("SELECT MAX(trade_date) FROM stock_daily"), engine).iloc[0, 0])
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    logger.info(f"区间: {start_date} → {end_date}")

    daily = load_daily_data(engine, all_codes, start_date, end_date,
                            cols=["open", "high", "low", "close", "volume", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])

    # 加载市值+行业
    with engine.connect() as conn:
        extra = pd.read_sql(text(
            "SELECT code, trade_date, market_cap FROM stock_daily_extra "
            "WHERE code = ANY(:codes) AND trade_date BETWEEN :s AND :e"),
            conn, params={"codes": all_codes, "s": start_date, "e": end_date})
        sectors = pd.read_sql(text("SELECT code, industry FROM stock_basic"), conn)
    extra["code"] = extra["code"].astype(str).str.zfill(6)
    extra["trade_date"] = pd.to_datetime(extra["trade_date"])
    sector_map = dict(zip(sectors["code"].astype(str).str.zfill(6), sectors["industry"]))

    # 合并
    daily = daily.merge(extra[["code", "trade_date", "market_cap"]],
                        on=["code", "trade_date"], how="left")
    daily["mcap"] = daily["market_cap"]
    daily["sector"] = daily["code"].map(sector_map).fillna("其他")
    engine.dispose()

    logger.info(f"日线: {len(daily)} 行, 市值: {daily['mcap'].notna().sum()}")

    # 预计算基础因子
    df = precompute_base_factors(daily)

    # ── 进化循环 ──
    t0 = time.time()
    console = Console() if HAS_RICH else None

    for r in range(args.rounds):
        round_num = db["rounds"] + 1
        logger.info(f"═══ Round {round_num} ═══")

        results = validate_all(df)

        # 自动生成组合因子
        combos = generate_combos(results, df)
        if combos:
            # 临时加入模板列表
            for name, cfg in combos.items():
                TEMPLATES[name] = cfg
            combo_results = validate_all(df)  # 只验证新组合
            results.update(combo_results)
            # 清理
            for name in combos:
                del TEMPLATES[name]

        entry = {"round": round_num, "date": str(pd.Timestamp.now())[:19], "results": results}
        db["history"].append(entry)
        db["rounds"] = round_num

        passed = sum(1 for r in results.values() if r.get("status") == "pass")
        valid = sum(1 for r in results.values() if r.get("ic_samples", 0) >= 5)
        best_ic = max((abs(r.get("ic_mean", 0)) for r in results.values()
                       if r.get("ic_samples", 0) >= 5), default=0)

        # 提取规则
        rules = extract_rules(db)
        if rules:
            with open(RULES_FILE, "w") as f:
                for k, v in sorted(rules.items()):
                    f.write(f"- **{k}**: {v}\n")

        save_db(db)

        elapsed = time.time() - t0
        logger.info(f"  通过: {passed}/{valid}, 最佳|IC|: {best_ic:.4f}, {elapsed:.0f}s")

        if HAS_RICH and round_num % 5 == 0:
            panel = _render_panel(db, round_num, elapsed)
            console.print(panel)

    elapsed = time.time() - t0
    logger.success(f"完成: {args.rounds} 轮, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
