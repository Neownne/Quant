#!/usr/bin/env python
"""自进化因子挖掘 v4.0 — 3阶段GP表达式树 + LightGBM LambdaRank + 回测管线 + 分析师暂停。

Stage 1: RPN表达式树演化（expression_tree）
Stage 2: LightGBM LambdaRank 学习因子权重（ml_ranker）
Stage 3: 样本外回测验证（bt_yaogu backtest pipeline）

Analyst Pause: 每 N 轮自动生成分析报告、暂停并等待分析师建议注入。

用法:
  python scripts/evolve_factors.py --rounds 10
  python scripts/evolve_factors.py --rounds 10 --analyst-interval 5
  python scripts/evolve_factors.py --status
  python scripts/evolve_factors.py --status --top 20
"""

from __future__ import annotations
import sys, os, json, argparse, time, random, shutil
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factors.data_assembler import assemble_universe
from factors.expression_tree import (
    evaluate_rpn,
    random_tree,
    mutate_rpn,
    crossover_rpn,
    _OP_ARITY,
    _OPERATORS,
)
from factors.ml_ranker import train_lambdarank, predict_rank
from factors.analyst import (
    generate_analysis_report,
    load_suggestions,
    apply_suggestions,
    save_analysis_report,
    write_auto_suggestions,
)
from factors.viz import plot_live_dashboard, plot_terminal_summary, plot_evolution_dashboard, plot_round_detail
from scripts.validate_factors import compute_rank_ic
from scripts.bt_yaogu import run_backtest_on_signals

# ══════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════

FACTOR_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'factor_db.json')
RULES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'factor_rules.md')

RAW_LEAVES = [
    "open", "high", "low", "close", "volume", "amount", "turnover",
    "mcap", "float_mcap", "pe", "pb", "total_share", "float_share",
    "roe", "gross_margin", "net_margin", "bps", "eps",
    "cashflow_ps", "ocf_ps", "goodwill_ratio", "debt_ratio", "adjusted_profit",
    "north_net", "north_buy", "north_sell",
    "cn_10y", "us_10y", "spread_cn_us", "usd_cny",
]

_DEFAULT_OP_PROBS = {
    "rank": 0.12,
    "zscore": 0.08,
    "sector_rank": 0.05,
    "ts_delta": 0.06,
    "ts_pct": 0.06,
    "ts_mean": 0.10,
    "ts_std": 0.06,
    "ts_rank": 0.08,
    "ts_min": 0.04,
    "ts_max": 0.04,
    "ts_corr": 0.04,
    "add": 0.07,
    "sub": 0.05,
    "mul": 0.05,
    "div": 0.04,
    "log": 0.06,
}


def _get_op_probs():
    """Return operator probabilities filtered to currently available operators.

    Used so that if an analyst kills an operator, random_tree won't pick it.
    """
    return {op: prob for op, prob in _DEFAULT_OP_PROBS.items() if op in _OP_ARITY}


# ══════════════════════════════════════════════════════════════════════
# 持久化 (KEEP EXACTLY AS-IS)
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


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════

def _evaluate_one(df, tokens, forward_ret):
    """Evaluate a single RPN expression. Returns dict or None.

    IC computed as daily cross-sectional Spearman (factor vs fwd_ret),
    averaged across days. This correctly penalizes macro columns with no
    cross-sectional variation (they get ~0 IC instead of spurious IC from
    time-series trend overlap).
    """
    try:
        fv = evaluate_rpn(df, tokens)
        valid = fv.notna() & forward_ret.notna()
        if valid.sum() < 100:
            return None

        # 每日截面 IC，取均值
        tmp = pd.DataFrame({"fv": fv, "fwd": forward_ret, "td": df["trade_date"].values})
        tmp = tmp.dropna()
        daily_ics = []
        for td, grp in tmp.groupby("td"):
            if len(grp) >= 10:
                ic = compute_rank_ic(grp["fv"].values, grp["fwd"].values)
                if not np.isnan(ic):
                    daily_ics.append(ic)

        if len(daily_ics) < 5:
            return None

        ic = float(np.mean(daily_ics))
        if np.isnan(ic):
            return None

        name = ";".join(str(t) for t in tokens)
        if len(name) > 80:
            name = name[:77] + "..."
        return {"name": name, "tokens": tokens, "ic": ic, "series": fv}
    except Exception:
        return None


# ── 调仓参数优化 ──
_TRADING_PARAMS = {"top_n": 5, "rebalance_days": 1, "min_hold_days": 3,
                   "trailing_stop": 0.12}

def _optimize_trading_params(ml_result, bt_factor_df, bt_df, factor_cols, name_map):
    """每5轮对当前最佳因子做网格搜索，找最优调仓参数。

    测试12组(top_n, rebalance_days, min_hold_days, trailing_stop)组合。
    回测很快——信号已计算好，只是执行规则不同。
    """
    combos = [
        (3, 1, 2, 0.10), (3, 3, 5, 0.15), (3, 5, 7, 0.20),
        (5, 1, 3, 0.12), (5, 3, 5, 0.10), (5, 5, 2, 0.18),
        (8, 1, 5, 0.10), (8, 3, 3, 0.18), (8, 5, 7, 0.12),
        (5, 1, 2, 0.20), (5, 3, 7, 0.12), (5, 1, 5, 0.08),
    ]

    scores = predict_rank(ml_result, bt_factor_df)
    sig_df = bt_factor_df.copy()
    sig_df["score"] = scores.values
    sig_df["date"] = sig_df["trade_date"]

    best_fitness = -999
    best_params = dict(_TRADING_PARAMS)
    best_bt = None

    print(f"   [tune] 调仓网格搜索 ({len(combos)} 组)...")
    for tn, rd, mh, ts in combos:
        topn = sig_df.groupby("trade_date").apply(
            lambda g: g.nlargest(tn, "score")
        ).reset_index(drop=True)
        topn = topn.rename(columns={"trade_date": "date"})

        bt = run_backtest_on_signals(
            topn[["date", "code", "score"]], bt_df,
            name_map=name_map, top_n=tn,
            min_score=-999.0, rebalance_days=rd,
            min_hold_days=mh, trailing_stop=ts,
        )
        if "error" in bt:
            continue

        ann = bt.get("ret_annual", 0) or 0
        mdd = bt.get("max_dd", 1) or 1
        sp = bt.get("sharpe", 0) or 0
        fit = _compute_fitness(ann, mdd, sp)
        if fit > best_fitness:
            best_fitness = fit
            best_params = {"top_n": tn, "rebalance_days": rd,
                           "min_hold_days": mh, "trailing_stop": ts}
            best_bt = bt

    if best_bt:
        print(f"   [tune] 最优: top_n={best_params['top_n']} rebalance={best_params['rebalance_days']}d "
              f"hold={best_params['min_hold_days']}d stop={best_params['trailing_stop']:.0%} "
              f"→ ann={best_bt.get('ret_annual',0):+.1%} mdd={best_bt.get('max_dd',0):.1%}")
        _TRADING_PARAMS.update(best_params)
    return best_params


def _compute_fitness(ann, mdd, sharpe=0.0):
    """适应度：年化收益为主，回撤惩罚，夏普加成。

    关键改动：正收益策略必须比"不交易"(0,0)分数高。
    旧公式 ann-mdd*2 使 -10%/+10%(-0.3) 不如 0/0(0.0)，
    导致进化偏向保守、不出信号。
    """
    if ann > 0:
        # 正收益：年化主导 + 夏普加成 - 回撤惩罚
        return ann * 2.0 + max(sharpe, 0) * 0.3 - mdd * 1.5
    else:
        # 负收益：重罚
        return ann * 3.0 - mdd * 2.0


def _evolve_population(population, results, leaf_pool, max_depth=None,
                       kill_patterns=None, bt_fitness=0.0):
    """Generate next generation with diversity pressure.

    Key fixes vs old version:
    - Elite count: 10→3 (20%→6%), stops self-replicating oligarchy
    - IC ranking penalized by correlation redundancy
    - kill_patterns actually enforced
    - Random injection ↑ (40%→60%) with higher mutation rate
    - Backtest fitness used as tiebreaker for IC-ranked elites
    """
    if not results:
        depth = max_depth or 4
        return [random_tree(leaf_pool, max_depth=depth, op_probs=_get_op_probs())
                for _ in range(len(population))]

    # ── Phase 1: IC ranking with redundancy penalty ──
    kill_set = set(kill_patterns or [])
    scored = []
    for r in results:
        tokens = r.get("tokens", [])
        name = r.get("name", "")
        ic = abs(r.get("ic", 0))

        # Kill dominated patterns
        if kill_set:
            name_str = ";".join(str(t) for t in tokens)
            if any(pat in name_str for pat in kill_set):
                continue  # skip this individual

        # Redundancy penalty: penalize if similar to already-selected elites
        penalty = 0.0
        for prev in scored:
            prev_name = prev.get("name", "")
            # Simple heuristic: if name shares >50% tokens, penalize
            prev_tokens = set(str(t) for t in prev.get("tokens", []))
            cur_tokens = set(str(t) for t in tokens)
            if prev_tokens and cur_tokens:
                overlap = len(prev_tokens & cur_tokens) / max(len(prev_tokens | cur_tokens), 1)
                if overlap > 0.5:
                    penalty += 0.02 * overlap

        scored.append({**r, "diversity_score": ic - penalty})

    if not scored:
        # All killed — regenerate fully random
        depth = max_depth or 4
        return [random_tree(leaf_pool, max_depth=depth, op_probs=_get_op_probs())
                for _ in range(len(population))]

    # Sort by diversity-adjusted IC
    scored.sort(key=lambda x: x["diversity_score"], reverse=True)

    # ── Phase 2: Build next generation ──
    # Only 3 elites survive verbatim (6% of pop, down from 20%)
    ELITE_N = 3
    elites = scored[:ELITE_N]

    new_pop = [list(e["tokens"]) for e in elites]

    op_probs = _get_op_probs()

    # Mutate ALL valid individuals (not just top-5), each once
    # Wider genetic base = more diverse mutations
    mutation_pool = scored[:15]  # top 15 by diversity-adjusted IC
    for e in mutation_pool:
        child = mutate_rpn(list(e["tokens"]), leaf_pool, mutation_rate=0.5)
        if child:
            new_pop.append(child)

    # Crossover diverse pairs (not just top-5 with each other)
    import random as _random
    crossover_pool = scored[:12]
    _random.shuffle(crossover_pool)
    for i in range(0, len(crossover_pool) - 1, 2):
        child = crossover_rpn(list(crossover_pool[i]["tokens"]),
                              list(crossover_pool[i+1]["tokens"]))
        if child:
            new_pop.append(child)

    # Heavy random injection (60% of target size) — maintain exploration
    depth = max_depth or 4
    target = len(population)
    while len(new_pop) < target:
        new_pop.append(random_tree(leaf_pool, max_depth=depth, op_probs=op_probs))

    return new_pop[:target]


# ══════════════════════════════════════════════════════════════════════
# 状态面板
# ══════════════════════════════════════════════════════════════════════

def _render_panel(db, top_n=10):
    history = db.get("history", [])
    print(f"═══ 因子进化 v4.0 状态 ═══")
    print(f"轮次: {db['rounds']} | 历史记录: {len(history)}")

    if not history:
        print("  (无历史记录)")
        return

    latest = history[-1]

    # Backward-compat: old entries use 'results', new entries use 'n_valid'/'n_individuals'
    n_valid = latest.get('n_valid', None)
    if n_valid is None:
        results = latest.get("results", {})
        n_valid = sum(1 for r in results.values() if r.get("status") == "pass")
        n_individuals = len(results)
    else:
        n_individuals = latest.get('n_individuals', '?')
    print(f"最近一轮: {n_valid}通过/{n_individuals}有效")

    # Top factors across all rounds (new format: top_factors; old format: results)
    all_factors = {}
    for h in history:
        for f in h.get("top_factors", []):
            name = f.get("name", "?")[:60]
            ic = abs(f.get("ic", 0))
            if name not in all_factors or ic > abs(all_factors[name].get("ic", 0)):
                all_factors[name] = f
        # Also handle old-format entries
        for name, r in h.get("results", {}).items():
            if r.get("n_samples", 0) >= 50:
                ic = r.get("tail_ic", 0)
                if name not in all_factors or abs(ic) > abs(all_factors[name].get("ic", 0)):
                    all_factors[name] = {"name": name, "ic": ic, "category": r.get("category", "?")}

    top = sorted(all_factors.values(), key=lambda x: abs(x.get("ic", 0)), reverse=True)[:top_n]
    for i, f in enumerate(top):
        cat = f.get('category', '')
        cat_str = f" [{cat}]" if cat else ""
        print(f"  {i+1:2d}. {f['name'][:50]:50s} IC={f.get('ic', 0):+.4f}{cat_str}")

    # Best backtest result
    candidates = [h for h in history if h.get("bt_annual") is not None and abs(h.get("bt_annual", 0)) > 0.001]
    if candidates:
        best_bt = max(candidates, key=lambda h: h.get("bt_annual", -999))
        bt_ann = best_bt.get('bt_annual', 0)
        bt_mdd = best_bt.get('bt_mdd', 0)
        bt_sharpe = best_bt.get('bt_sharpe', 0)
        print(f"\n最佳回测: 年化={bt_ann:+.1%} MDD={bt_mdd:.1%} "
              f"夏普={bt_sharpe:+.2f} (第{best_bt['round']}轮)")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="因子进化 v4.0")
    p.add_argument("--rounds", type=int, default=10, help="进化轮次")
    p.add_argument("--status", action="store_true")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--analyst-interval", type=int, default=5,
                   help="每 N 轮暂停并生成分析报告（0=不暂停）")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.status:
        db = load_db()
        _render_panel(db, args.top)
        return

    # 抑制 bt_yaogu DEBUG 刷屏
    from loguru import logger
    logger.remove()
    logger.add(lambda _: None, level="WARNING")

    # ── Setup DB connection ──
    from config.settings import DBConfig
    engine = DBConfig.create_engine()

    # Date range: 6 years → 3年训练IC + 3年回测
    end_date = pd.read_sql("SELECT MAX(trade_date) FROM stock_daily", engine).iloc[0, 0]
    end_date = str(end_date)[:10]
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=365 * 6)).strftime("%Y-%m-%d")

    print(f"[v4.0] 数据: {start_date} → {end_date} (6年)")

    # ── Stage 1: Load data ──
    print("[1/4] 加载全维度数据...")
    df = assemble_universe(engine, start_date, end_date)
    print(f"   {len(df)} 行, {df['code'].nunique()} 只股票")

    # Split: 前3年训练IC + 后3年回测
    df = df.sort_values(["code", "trade_date"])
    bt_start = (pd.Timestamp(end_date) - pd.Timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    train_df = df[df["trade_date"] < bt_start].copy()
    bt_df = df[df["trade_date"] >= bt_start].copy()

    # 前向收益仅在训练集上计算（ML标签），回测引擎自己算PnL
    train_df["fwd_5d"] = train_df.groupby("code")["close"].transform(
        lambda x: x.pct_change(5).shift(-5)
    )
    print(f"   训练: {len(train_df)} 行, 回测: {len(bt_df)} 行")

    # ── Build leaf pool ──
    leaf_pool = list(RAW_LEAVES)

    # Precompute key @ALL_FACTORS as columns for GP leaves
    print("[2/4] 叶子池: 预计算因子...")
    try:
        from factors.engine import FactorEngine
        KEY_FACTORS = [
            "lu_streak", "lu_seal_quality", "lu_vol_intensity", "lu_amplitude",
            "lu_volume_climax", "lu_streak_quality",
            "lu_count_5d", "lu_count_20d", "lu_days_since_last",
            "log_mcap", "turnover_mom", "gap_ratio", "intra_vol",
            "rev_5", "rev_20", "mom_20", "mom_60",
            "rsi_14", "bb_position", "atr_14",
        ]
        engine = FactorEngine(KEY_FACTORS)
        # Compute once for train, once for backtest
        train_factors = engine.compute(train_df)
        bt_factors = engine.compute(bt_df)
        for fc in KEY_FACTORS:
            col_name = f"@{fc}"
            if fc in train_factors.columns:
                train_df[col_name] = train_factors[fc].values
            if fc in bt_factors.columns:
                bt_df[col_name] = bt_factors[fc].values
        leaf_pool.extend(["@" + f for f in KEY_FACTORS])
        print(f"   叶子池: {len(leaf_pool)} ({len(RAW_LEAVES)} 原始列 + {len(KEY_FACTORS)} @因子)")
    except Exception as e:
        print(f"   叶子池: {len(leaf_pool)} 原始列 (预计算失败: {e})")

    # ── Initialize population ──
    POP_SIZE = 50
    op_probs = _get_op_probs()
    population = [random_tree(leaf_pool, max_depth=4, op_probs=op_probs) for _ in range(POP_SIZE)]

    # ── Load DB ──
    db = load_db()
    start_round = db["rounds"]

    print(f"[3/4] 种群: {POP_SIZE} 个体, 起始轮次: {start_round + 1}")

    # ── Load suggestions if any ──
    analyst_max_depth = None
    kill_patterns_list = []  # persistent, updated by auto-analyst suggestions
    suggestions = load_suggestions()
    if suggestions:
        print(f"   [v] 加载分析师建议")
        leaf_pool, _ = apply_suggestions(suggestions, leaf_pool, {})
        if "cap_max_depth" in suggestions:
            analyst_max_depth = suggestions["cap_max_depth"]
        if "kill_operator" in suggestions:
            for op in suggestions["kill_operator"]:
                _OP_ARITY.pop(op, None)
                _OPERATORS.pop(op, None)
            print(f"   已禁用操作符: {suggestions['kill_operator']}")
        if "kill_patterns" in suggestions:
            kill_patterns_list = suggestions["kill_patterns"]
            print(f"   已禁用模式: {kill_patterns_list}")

    # ══════════════════════════════════════════════════════════════
    # EVOLUTION LOOP
    # ══════════════════════════════════════════════════════════════
    for gen in range(args.rounds):
        round_num = start_round + gen + 1
        t_round_start = time.time()

        print(f"\n{'=' * 60}")
        print(f"第 {round_num} 轮进化")
        print(f"{'=' * 60}")

        # Evaluate all individuals
        results = []
        t0_eval = time.time()
        for i, tokens in enumerate(population):
            r = _evaluate_one(train_df, tokens, train_df["fwd_5d"])
            if r is not None:
                results.append(r)
            # 进度条（每5个或最后一个）
            if (i + 1) % 5 == 0 or i == len(population) - 1:
                pct = (i + 1) / len(population) * 100
                bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
                elapsed = time.time() - t0_eval
                eta = elapsed / (i + 1) * (len(population) - i - 1) if i > 0 else 0
                print(f"\r  [{bar}] {pct:.0f}% {len(results)}有效 {elapsed:.0f}s ETA:{eta:.0f}s",
                      end="", flush=True, file=sys.stdout)
        print(file=sys.stdout)  # newline

        if len(results) < 3:
            print(f"   [!] 仅有 {len(results)} 个有效个体，跳过 ML/回测")
            # Save minimal round data
            round_data = {
                "round": round_num,
                "n_individuals": len(population),
                "n_valid": len(results),
                "best_ic": abs(results[0]["ic"]) if results else 0.0,
                "best_fitness": 0.0,
                "bt_annual": 0.0,
                "bt_mdd": 1.0,
                "bt_sharpe": 0.0,
                "bt_wr": 0.0,
                "bt_trades": 0,
                "ml_ndcg": 0.0,
                "top_factors": [],
            }
            db["rounds"] = round_num
            db["history"].append(round_data)
            save_db(db)

            if gen < args.rounds - 1:
                population = _evolve_population(population, results, leaf_pool, analyst_max_depth, kill_patterns_list)
            continue

        # Sort by |IC|, keep top-10 for ML (减少噪声因子干扰)
        results.sort(key=lambda x: abs(x["ic"]), reverse=True)
        top_factors = results[:10]

        # ── Stage 2: LightGBM LambdaRank ──
        # CRITICAL: 截面排名因子值 — 宏观列同一天所有股票值相同，不排名 qcut 会报错
        factor_df = train_df[["code", "trade_date"]].copy()
        for i, r in enumerate(top_factors):
            col = f"f_{i}"
            raw_vals = r["series"].values
            # 截面排名：每天内 rank(pct)，值域 [0, 1]，消除宏观列同值问题
            factor_df[col] = raw_vals
            r["col_name"] = col

        # 按天做截面排名
        factor_cols = [f"f_{i}" for i in range(len(top_factors))]
        for col in factor_cols:
            factor_df[col] = factor_df.groupby("trade_date")[col].transform(
                lambda g: g.rank(pct=True)
            )
        # 排名后填 NaN（rank 不会产生 NaN，但 transform 可能导致某些天只有 NaN）
        factor_df[factor_cols] = factor_df[factor_cols].fillna(0.5)

        ml_result = {"model": None, "ndcg_score": 0.0, "feature_importances": {}}
        try:
            ml_result = train_lambdarank(factor_df, train_df["fwd_5d"])
        except Exception as e:
            print(f"   [!] ML训练失败: {e}")

        if ml_result.get("model") is None:
            print(f"   [!] ML模型为None — 可能是因子截面变异不足或训练数据不够")

        # ── Stage 3: Backtest on out-of-sample period ──
        bt_annual = 0.0
        bt_mdd = 1.0
        bt_sharpe = 0.0
        bt_wr = 0.0
        bt_trades = 0

        if ml_result.get("model") is not None:
            # Predict on backtest period
            bt_factor_df = bt_df[["code", "trade_date"]].copy()
            for i, r in enumerate(top_factors):
                col = f"f_{i}"
                try:
                    bt_fv = evaluate_rpn(bt_df, r["tokens"])
                    bt_factor_df[col] = bt_fv.values
                except Exception:
                    bt_factor_df[col] = np.nan

            # 截面排名（与训练保持一致）
            for col in factor_cols:
                bt_factor_df[col] = bt_factor_df.groupby("trade_date")[col].transform(
                    lambda g: g.rank(pct=True)
                )
            bt_factor_df[factor_cols] = bt_factor_df[factor_cols].fillna(0.5)

            try:
                scores = predict_rank(ml_result, bt_factor_df)

                # Build signals: per day, top-5 by score
                sig_df = bt_factor_df.copy()
                sig_df["score"] = scores.values
                sig_df["date"] = sig_df["trade_date"]

                top5 = sig_df.groupby("trade_date").apply(
                    lambda g: g.nlargest(_TRADING_PARAMS["top_n"], "score")
                ).reset_index(drop=True)
                top5 = top5.rename(columns={"trade_date": "date"})

                # Simple name map (code -> code)
                name_map = dict(zip(bt_df["code"].unique(), bt_df["code"].unique()))

                bt = run_backtest_on_signals(
                    top5[["date", "code", "score"]], bt_df,
                    name_map=name_map,
                    top_n=_TRADING_PARAMS["top_n"],
                    min_score=-999.0,
                    rebalance_days=_TRADING_PARAMS["rebalance_days"],
                    min_hold_days=_TRADING_PARAMS["min_hold_days"],
                    trailing_stop=_TRADING_PARAMS["trailing_stop"],
                )

                if "error" not in bt:
                    bt_annual = bt.get("ret_annual", 0.0) or 0.0
                    bt_mdd = bt.get("max_dd", 1.0) or 1.0
                    bt_sharpe = bt.get("sharpe", 0.0) or 0.0
                    bt_wr = bt.get("win_rate", 0.0) or 0.0
                    bt_trades = bt.get("n_trades", 0) or 0
            except Exception as e:
                print(f"   [!] 回测失败: {e}")

        # Fitness
        fitness = _compute_fitness(bt_annual, bt_mdd, bt_sharpe)

        # Display results
        best_ic = abs(results[0]["ic"]) if results else 0
        print(f"   有效: {len(results)}/{len(population)} | 训练IC: {best_ic:+.4f} | 适应度: {fitness:+.4f}")
        if ml_result.get("model") is not None:
            print(f"   回测: 年化={bt_annual:+.1%} MDD={bt_mdd:.1%} 夏普={bt_sharpe:+.2f} WR={bt_wr:.1%} trades={bt_trades}")
            print(f"   ML NDCG: {ml_result.get('ndcg_score', 0):.4f}")

        # Show top-3 factors
        for i, r in enumerate(results[:3]):
            print(f"   #{i+1}: {r['name'][:60]}  IC={r['ic']:+.4f}")

        # Save round data to DB
        round_data = {
            "round": round_num,
            "n_individuals": len(population),
            "n_valid": len(results),
            "best_ic": best_ic,
            "best_fitness": fitness,
            "bt_annual": bt_annual,
            "bt_mdd": bt_mdd,
            "bt_sharpe": bt_sharpe,
            "bt_wr": bt_wr,
            "bt_trades": bt_trades,
            "ml_ndcg": ml_result.get("ndcg_score", 0),
            "top_factors": [
                {"name": r["name"][:60], "ic": r["ic"], "tokens": r["tokens"]}
                for r in results[:10]
            ],
        }
        db["rounds"] = round_num
        db["history"].append(round_data)
        save_db(db)

        # Terminal progress
        try:
            plot_live_dashboard(db)
        except Exception as e:
            print(f"   [!] 终端图表失败: {e}")

        t_elapsed = time.time() - t_round_start
        print(f"   耗时: {t_elapsed:.0f}s")

        # ── 每轮自动建议（代码做基础优化）──
        try:
            write_auto_suggestions(db, round_num)
            new_suggestions = load_suggestions()
            if new_suggestions:
                leaf_pool, _ = apply_suggestions(new_suggestions, leaf_pool, {})
                if "cap_max_depth" in new_suggestions:
                    analyst_max_depth = new_suggestions["cap_max_depth"]
                if "kill_patterns" in new_suggestions:
                    kill_patterns_list = new_suggestions["kill_patterns"]
                # Archive
                sp = "data/suggestions.json"
                if os.path.exists(sp):
                    shutil.move(sp, f"data/suggestions_round_{round_num:04d}.json")
        except Exception as e:
            print(f"   [!] 自动建议失败: {e}")

        # ── 每5轮调仓参数优化 ──
        if round_num % 5 == 0 and ml_result.get("model") is not None:
            try:
                # Rebuild bt_factor_df for tuning (same as backtest setup)
                tune_factor_df = bt_df[["code", "trade_date"]].copy()
                for i, r in enumerate(top_factors):
                    col = f"f_{i}"
                    try:
                        bt_fv = evaluate_rpn(bt_df, r["tokens"])
                        tune_factor_df[col] = bt_fv.values
                    except Exception:
                        tune_factor_df[col] = np.nan
                for col in factor_cols:
                    tune_factor_df[col] = tune_factor_df.groupby("trade_date")[col].transform(
                        lambda g: g.rank(pct=True))
                tune_factor_df[factor_cols] = tune_factor_df[factor_cols].fillna(0.5)
                _optimize_trading_params(ml_result, tune_factor_df, bt_df, factor_cols, name_map)
            except Exception as e:
                print(f"   [!] 调仓优化失败: {e}")

        # ── 每 N 轮 Claude 深度审查暂停 ──
        ANALYST_INTERVAL = args.analyst_interval
        if ANALYST_INTERVAL > 0 and round_num % ANALYST_INTERVAL == 0:
            _handle_analyst_pause(
                train_df, population, results, ml_result,
                bt_annual, bt_mdd, bt_sharpe, bt_wr, bt_trades,
                db, round_num, leaf_pool,
            )
            # 暂停后重新加载建议（Claude 可能覆盖了）
            new_suggestions = load_suggestions()
            if new_suggestions:
                print(f"   [v] 已加载 Claude 建议")
                leaf_pool, _ = apply_suggestions(new_suggestions, leaf_pool, {})
                if "cap_max_depth" in new_suggestions:
                    analyst_max_depth = new_suggestions["cap_max_depth"]
                if "kill_operator" in new_suggestions:
                    for op in new_suggestions["kill_operator"]:
                        _OP_ARITY.pop(op, None)
                        _OPERATORS.pop(op, None)
                    print(f"   已禁用操作符: {new_suggestions['kill_operator']}")
                if "kill_patterns" in new_suggestions:
                    kill_patterns_list = new_suggestions["kill_patterns"]
                    print(f"   已禁用模式: {kill_patterns_list}")
                # Archive suggestions file
                suggestions_path = "data/suggestions.json"
                if os.path.exists(suggestions_path):
                    archive_path = f"data/suggestions_round_{round_num:04d}.json"
                    shutil.move(suggestions_path, archive_path)
            else:
                print(f"   [-] 无建议文件，继续进化")

        # ── Evolve to next generation (unless last round) ──
        if gen < args.rounds - 1:
            population = _evolve_population(population, results, leaf_pool, analyst_max_depth, kill_patterns_list)

    print(f"\n[4/4] 完成。{args.rounds} 轮进化。")
    _render_panel(db, args.top)


def _handle_analyst_pause(train_df, population, results, ml_result,
                          bt_annual, bt_mdd, bt_sharpe, bt_wr, bt_trades,
                          db, round_num, leaf_pool):
    """Generate analysis report, charts, and pause for analyst suggestions."""
    suggestions_path = "data/suggestions.json"

    # Generate analysis report
    try:
        report = generate_analysis_report(
            train_df, population, results,
            {
                "feature_importances": ml_result.get("feature_importances", {}),
                "ndcg_score": ml_result.get("ndcg_score", 0),
                "bt_annual": bt_annual,
                "bt_mdd": bt_mdd,
                "bt_sharpe": bt_sharpe,
                "bt_wr": bt_wr,
                "bt_trades": bt_trades,
            },
            db, round_num,
        )
        save_analysis_report(report, round_num)
        print(f"   [v] 分析报告已保存")
    except Exception as e:
        print(f"   [!] 分析报告生成失败: {e}")

    # Generate charts
    try:
        os.makedirs("data/charts", exist_ok=True)
        plot_evolution_dashboard(db, save_path=f"data/charts/dashboard_round_{round_num:04d}.png")
        plot_round_detail(report if 'report' in dir() else {}, save_path=f"data/charts/detail_round_{round_num:04d}.png")
        print(f"   [v] 图表已保存到 data/charts/")
    except Exception as e:
        print(f"   [!] 图表生成失败: {e}")

    # Pause and wait for suggestions — poll file mtime instead of stdin
    # (works in background/non-interactive mode)
    import time as _time
    print(f"\n{'~' * 60}")
    print(f"  ⏸  Claude 审查点 — 第 {round_num} 轮")
    print(f"  分析报告: data/analysis_round_{round_num:04d}.json")
    print(f"  图表: data/charts/dashboard_round_{round_num:04d}.png")
    print(f"  编辑 data/suggestions.json 写入建议，自动检测继续。")
    print(f"  5 分钟超时后自动继续。")
    print(f"{'~' * 60}")

    # Record initial state
    old_mtime = os.path.getmtime(suggestions_path) if os.path.exists(suggestions_path) else None
    waited = 0
    while waited < 300:  # 5 min for Claude review
        _time.sleep(3)
        waited += 3
        if os.path.exists(suggestions_path):
            new_mtime = os.path.getmtime(suggestions_path)
            if old_mtime is None or new_mtime > old_mtime:
                print(f"   [v] 检测到建议文件更新 (waited {waited}s)")
                return
        else:
            if old_mtime is not None:
                print(f"   [-] 建议文件已删除，跳过分析 (waited {waited}s)")
                return

    print(f"   [-] 超时 ({waited}s)，无建议更新，继续进化")


if __name__ == "__main__":
    main()
