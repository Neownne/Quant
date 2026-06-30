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


def _compute_fitness(ann, mdd):
    """ACGR >= 40% & MDD <= 15% -> 高分，否则惩罚。"""
    if ann >= 0.40 and mdd <= 0.15:
        return ann * 2 - mdd
    else:
        return ann - mdd * 2


def _evolve_population(population, results, leaf_pool, max_depth=None):
    """Generate next generation from current results."""
    if not results:
        depth = max_depth or 4
        return [random_tree(leaf_pool, max_depth=depth, op_probs=_get_op_probs())
                for _ in range(len(population))]

    # Sort by |IC|
    ranked = sorted(results, key=lambda x: abs(x["ic"]), reverse=True)
    elites = ranked[:10]

    new_pop = [list(e["tokens"]) for e in elites]

    op_probs = _get_op_probs()

    # Mutate top 5 elites x 2 children each
    for e in elites[:5]:
        for _ in range(2):
            child = mutate_rpn(list(e["tokens"]), leaf_pool)
            if child:
                new_pop.append(child)

    # Crossover top 5
    for i in range(min(5, len(elites))):
        for j in range(i + 1, min(5, len(elites))):
            child = crossover_rpn(list(elites[i]["tokens"]), list(elites[j]["tokens"]))
            if child:
                new_pop.append(child)

    # Random injection to fill
    depth = max_depth or 4
    while len(new_pop) < len(population):
        new_pop.append(random_tree(leaf_pool, max_depth=depth, op_probs=op_probs))

    return new_pop[:len(population)]


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

    # ── Setup DB connection ──
    from config.settings import DBConfig
    engine = DBConfig.create_engine()

    # Date range: 3 years
    end_date = pd.read_sql("SELECT MAX(trade_date) FROM stock_daily", engine).iloc[0, 0]
    end_date = str(end_date)[:10]
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    print(f"[v4.0] 数据范围: {start_date} -> {end_date}")

    # ── Stage 1: Load data ──
    print("[1/4] 加载全维度数据...")
    df = assemble_universe(engine, start_date, end_date)
    print(f"   {len(df)} 行, {df['code'].nunique()} 只股票")

    # Compute forward returns (5-day)
    df = df.sort_values(["code", "trade_date"])
    df["fwd_5d"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change(5).shift(-5)
    )

    # Train/backtest split: last 6 months for backtest
    bt_start = (pd.Timestamp(end_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    train_df = df[df["trade_date"] < bt_start].copy()
    bt_df = df[df["trade_date"] >= bt_start].copy()
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
        for tokens in population:
            r = _evaluate_one(train_df, tokens, train_df["fwd_5d"])
            if r is not None:
                results.append(r)

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
                population = _evolve_population(population, results, leaf_pool, analyst_max_depth)
            continue

        # Sort by |IC|, keep top-20 for ML
        results.sort(key=lambda x: abs(x["ic"]), reverse=True)
        top_factors = results[:20]

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
                    lambda g: g.nlargest(5, "score")
                ).reset_index(drop=True)
                top5 = top5.rename(columns={"trade_date": "date"})

                # Simple name map (code -> code)
                name_map = dict(zip(bt_df["code"].unique(), bt_df["code"].unique()))

                bt = run_backtest_on_signals(
                    top5[["date", "code", "score"]], bt_df,
                    name_map=name_map, top_n=5,
                    min_score=-999.0,  # ML分数是0-1，yaogu默认min_score=3会滤掉全部
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
        fitness = _compute_fitness(bt_annual, bt_mdd)

        # Display results
        best_ic = abs(results[0]["ic"]) if results else 0
        print(f"   有效: {len(results)}/{len(population)} | 最佳IC: {best_ic:+.4f} | 适应度: {fitness:+.4f}")
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

        # ── ANALYST PAUSE (Task 8) ──
        ANALYST_INTERVAL = args.analyst_interval
        if ANALYST_INTERVAL > 0 and round_num % ANALYST_INTERVAL == 0:
            _handle_analyst_pause(
                train_df, population, results, ml_result,
                bt_annual, bt_mdd, bt_sharpe, bt_wr, bt_trades,
                db, round_num, leaf_pool,
            )
            # Reload suggestions after pause and apply
            new_suggestions = load_suggestions()
            if new_suggestions:
                print(f"   [v] 已加载建议")
                leaf_pool, _ = apply_suggestions(new_suggestions, leaf_pool, {})
                if "cap_max_depth" in new_suggestions:
                    analyst_max_depth = new_suggestions["cap_max_depth"]
                if "kill_operator" in new_suggestions:
                    for op in new_suggestions["kill_operator"]:
                        _OP_ARITY.pop(op, None)
                        _OPERATORS.pop(op, None)
                    print(f"   已禁用操作符: {new_suggestions['kill_operator']}")
                # Archive suggestions file
                suggestions_path = "data/suggestions.json"
                if os.path.exists(suggestions_path):
                    archive_path = f"data/suggestions_round_{round_num:04d}.json"
                    shutil.move(suggestions_path, archive_path)
            else:
                print(f"   [-] 无建议文件，继续进化")

        # ── Evolve to next generation (unless last round) ──
        if gen < args.rounds - 1:
            population = _evolve_population(population, results, leaf_pool, analyst_max_depth)

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

        # 自动分析 → 生成建议 → 写入 suggestions.json
        try:
            write_auto_suggestions(db, round_num)
        except Exception as e:
            print(f"   [!] 自动建议生成失败: {e}")
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
    print(f"  Paused at 第 {round_num} 轮。Waiting for analyst suggestions...")
    print(f"  Edit data/suggestions.json, save it, and I'll detect the change.")
    print(f"  Delete data/suggestions.json to skip (continue without suggestions).")
    print(f"  Timeout in 10 minutes if no changes detected.")
    print(f"{'~' * 60}")

    # Record initial state
    old_mtime = os.path.getmtime(suggestions_path) if os.path.exists(suggestions_path) else None
    waited = 0
    while waited < 600:  # 10 min timeout
        _time.sleep(10)
        waited += 10
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
