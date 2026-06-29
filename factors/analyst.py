"""Deep Analysis Report Generator — 7-dimension diagnostics at evolutionary pause points.

Usage:
    from factors.analyst import generate_analysis_report, load_suggestions, apply_suggestions

    report = generate_analysis_report(df, population, results, ml_result, db, round_num)
    suggestions = load_suggestions()
    if suggestions:
        pool, probs = apply_suggestions(suggestions, leaf_pool, operator_probs)
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Import operator arity from the expression tree engine
from factors.expression_tree import _OP_ARITY, _OP_INT_PARAMS, _OP_NUM_SUBTREES

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data Source Classification
# ═══════════════════════════════════════════════════════════════════════════════

LEAF_DATA_SOURCE: dict[str, str] = {
    "open": "price_volume",
    "high": "price_volume",
    "low": "price_volume",
    "close": "price_volume",
    "volume": "price_volume",
    "amount": "price_volume",
    "turnover": "price_volume",
    "mcap": "valuation",
    "float_mcap": "valuation",
    "pe": "valuation",
    "pb": "valuation",
    "total_share": "valuation",
    "float_share": "valuation",
    "roe": "financial",
    "gross_margin": "financial",
    "net_margin": "financial",
    "bps": "financial",
    "eps": "financial",
    "cashflow_ps": "financial",
    "ocf_ps": "financial",
    "goodwill_ratio": "financial",
    "debt_ratio": "financial",
    "adjusted_profit": "financial",
    "north_net": "macro",
    "north_buy": "macro",
    "north_sell": "macro",
    "cn_10y": "macro",
    "us_10y": "macro",
    "spread_cn_us": "macro",
    "usd_cny": "macro",
    "bar0_ret": "intraday",
    "bar3_ret": "intraday",
    "intra_vol": "intraday",
}


def _classify_leaf(leaf_name: str) -> str:
    """Classify a leaf token by its data source category."""
    if leaf_name.startswith("@"):
        return "prebuilt_factor"
    return LEAF_DATA_SOURCE.get(leaf_name, "other")


# Recognise parameter tokens: integers used as window sizes etc.
def _is_param(token: str) -> bool:
    """Check if a token is an integer parameter (window size, etc.)."""
    try:
        int(token)
        return True
    except ValueError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Analysis Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _leaf_usage(population: list[list[str]]) -> dict:
    """Count how often each leaf token appears across the population.

    Leaf = column name, integer parameter, or prebuilt factor token.
    Operators are excluded.

    Returns:
        dict: {leaf_name: count} sorted descending by count.
    """
    operator_set = set(_OP_ARITY.keys())
    counter: Counter = Counter()
    for tokens in population:
        for tok in tokens:
            if tok not in operator_set:
                counter[tok] += 1
    return dict(counter.most_common())


def _operator_usage(population: list[list[str]]) -> dict:
    """Count how often each operator appears across the population.

    Returns:
        dict: {operator_name: count} sorted descending by count.
    """
    operator_set = set(_OP_ARITY.keys())
    counter: Counter = Counter()
    for tokens in population:
        for tok in tokens:
            if tok in operator_set:
                counter[tok] += 1
    return dict(counter.most_common())


def _tree_depth(tokens: list[str]) -> int:
    """Compute RPN expression tree depth by simulating stack.

    Leaves (column names, integer params, prebuilt factors) push depth 0.
    Operators pop *arity* items, push max(child_depths) + 1.

    Returns:
        int: tree depth.  Returns 0 for empty or trivial trees.
    """
    if not tokens:
        return 0

    arity = _OP_ARITY
    stack: list[int] = []

    for tok in tokens:
        if tok in arity:
            ar = arity[tok]
            if len(stack) < ar:
                # Malformed expression — treat as leaf
                stack = [0]
                break
            children = [stack.pop() for _ in range(ar)]
            stack.append(max(children) + 1)
        else:
            stack.append(0)

    return max(stack) if stack else 0


def _factor_correlation(results: list[dict]) -> dict:
    """Compute pairwise Spearman correlation between top factors.

    Args:
        results: List of result dicts, each with "name", "series", "ic" keys.

    Returns:
        dict: {"redundant_pairs": [...], "high_corr_count": N}
    """
    # Filter to factors that have a valid series
    valid = [r for r in results if "series" in r and isinstance(r["series"], pd.Series)]
    if len(valid) < 2:
        return {"redundant_pairs": [], "high_corr_count": 0}

    # Take top 20 by abs IC
    sorted_valid = sorted(valid, key=lambda r: abs(r.get("ic", 0)), reverse=True)
    top20 = sorted_valid[:20]

    redundant_pairs: list[dict] = []
    high_corr_count = 0
    n = len(top20)

    for i in range(n):
        for j in range(i + 1, n):
            s_i = top20[i]["series"].dropna()
            s_j = top20[j]["series"].dropna()
            # Align indices
            common = s_i.index.intersection(s_j.index)
            if len(common) < 10:
                continue
            corr, _ = spearmanr(s_i.loc[common], s_j.loc[common])
            if np.isnan(corr):
                continue
            abs_corr = abs(corr)
            if abs_corr > 0.7:
                high_corr_count += 1
                redundant_pairs.append({
                    "f1": top20[i]["name"],
                    "f2": top20[j]["name"],
                    "spearman_corr": round(float(corr), 4),
                })

    return {
        "redundant_pairs": sorted(redundant_pairs, key=lambda p: -abs(p["spearman_corr"])),
        "high_corr_count": high_corr_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Main Report Generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_analysis_report(
    df: pd.DataFrame,
    population: list[list[str]],
    results: list[dict],
    ml_result: dict,
    db: dict,
    round_num: int,
) -> dict:
    """Generate 7-dimension deep analysis report as a JSON-serializable dict.

    Dimensions:
        1. data_source_coverage  — leaf source distribution, unused leaves, top leaves
        2. factor_structure      — depth distribution, operator usage, avg depth, avg nodes, depth vs IC
        3. ic_decay              — IC trend over rounds, current avg IC
        4. regime_sensitivity    — industry & market-cap IC breakdown
        5. factor_redundancy     — pairwise correlations, redundant pairs
        6. ml_feature_importance — LightGBM feature importances, NDCG score
        7. backtest_diagnostics  — annual return, max DD, Sharpe, win rate, n_trades, fitness

    Args:
        df: Panel dataframe with at least [code, trade_date, close, fwd_5d].
        population: Current factor population, list of RPN token lists.
        results: List of per-factor result dicts (each with name, tokens, ic, series).
        ml_result: Dict from train_lambdarank with model, feature_importances, ndcg_score, etc.
        db: Factor DB dict (with history list).
        round_num: Current evolution round number.

    Returns:
        dict: JSON-serializable report with all 7 dimensions.
    """
    report: dict = {}

    # ── Dimension 1: Data Source Coverage ──
    report["data_source_coverage"] = _dim_data_source(population, df)

    # ── Dimension 2: Factor Structure ──
    report["factor_structure"] = _dim_factor_structure(population, results)

    # ── Dimension 3: IC Decay ──
    report["ic_decay"] = _dim_ic_decay(db, results)

    # ── Dimension 4: Regime Sensitivity ──
    report["regime_sensitivity"] = _dim_regime_sensitivity(df)

    # ── Dimension 5: Factor Redundancy ──
    report["factor_redundancy"] = _factor_correlation(results)

    # ── Dimension 6: ML Feature Importance ──
    report["ml_feature_importance"] = _dim_ml_importance(ml_result)

    # ── Dimension 7: Backtest Diagnostics ──
    report["backtest_diagnostics"] = _dim_backtest(ml_result)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# 3a. Dimension helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _dim_data_source(population: list[list[str]], df: pd.DataFrame) -> dict:
    """Dimension 1: Data source coverage."""
    usage = _leaf_usage(population)
    op_set = set(_OP_ARITY.keys())

    # Classify each leaf
    source_counts: Counter = Counter()
    leaf_details: dict[str, int] = {}
    for leaf, count in usage.items():
        if leaf in op_set:
            continue
        src = _classify_leaf(leaf)
        source_counts[src] += count
        leaf_details[leaf] = count

    total = sum(source_counts.values())
    source_pct = {}
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        source_pct[src] = round(cnt / total * 100, 2) if total > 0 else 0.0

    # Unused leaves: leaves in LEAF_DATA_SOURCE not appearing in population
    all_known_leaves = set(LEAF_DATA_SOURCE.keys())
    used_leaves = {leaf for leaf in leaf_details if leaf in all_known_leaves}
    unused_leaves = sorted(all_known_leaves - used_leaves)

    # Also consider columns in df that are known leaf names
    df_leaf_cols = set(df.columns) & all_known_leaves
    unused_from_df = sorted(df_leaf_cols - used_leaves)

    # Top leaves
    top_leaves = dict(sorted(leaf_details.items(), key=lambda x: -x[1])[:20])

    return {
        "source_pct": source_pct,
        "unused_leaves": unused_leaves[:30],
        "unused_from_df": unused_from_df[:30],
        "top_leaves": top_leaves,
    }


def _dim_factor_structure(population: list[list[str]], results: list[dict]) -> dict:
    """Dimension 2: Factor structure analysis."""
    op_usage = _operator_usage(population)

    depths = [_tree_depth(tokens) for tokens in population]
    node_counts = [len([t for t in tokens if t not in _OP_ARITY]) for tokens in population]

    if depths:
        avg_depth = round(np.mean(depths), 2)
        depth_dist = dict(sorted(Counter(depths).items()))
    else:
        avg_depth = 0.0
        depth_dist = {}

    if node_counts:
        avg_nodes = round(np.mean(node_counts), 2)
    else:
        avg_nodes = 0.0

    # Depth vs IC
    depth_vs_ic: list[dict] = []
    for r in results:
        if "tokens" in r and "ic" in r:
            d = _tree_depth(r["tokens"])
            depth_vs_ic.append({
                "name": r.get("name", "?"),
                "depth": d,
                "ic": round(float(r["ic"]), 6),
            })

    return {
        "depth_distribution": depth_dist,
        "operator_usage": op_usage,
        "avg_depth": avg_depth,
        "avg_nodes": avg_nodes,
        "depth_vs_ic": sorted(depth_vs_ic, key=lambda x: abs(x["ic"]), reverse=True)[:20],
    }


def _dim_ic_decay(db: dict, results: list[dict]) -> dict:
    """Dimension 3: IC decay trend."""
    trend: list[dict] = []
    history = db.get("history", [])
    if history:
        for entry in history:
            rnd = entry.get("round", 0)
            entry_results = entry.get("results", [])
            if entry_results:
                avg_abs_ic = round(
                    float(np.mean([abs(r.get("ic", 0)) for r in entry_results])), 6
                )
            else:
                avg_abs_ic = 0.0
            trend.append({"round": rnd, "avg_abs_ic": avg_abs_ic})

    # Current round avg IC
    current_avg_ic = 0.0
    if results:
        current_avg_ic = round(
            float(np.mean([abs(r.get("ic", 0)) for r in results])), 6
        )

    return {
        "trend": trend[-50:],  # last 50 rounds max
        "current_avg_ic": current_avg_ic,
    }


def _dim_regime_sensitivity(df: pd.DataFrame) -> dict:
    """Dimension 4: Regime sensitivity — industry & market cap IC.

    Computes Spearman rank IC (fwd_5d vs close return) within each industry
    and market cap bucket.
    """
    result: dict = {"industry_ic": {}, "market_cap_ic": {}}

    # Industry IC
    if "industry_sw1" in df.columns and "fwd_5d" in df.columns and "close" in df.columns:
        industry_ic = {}
        df_work = df.dropna(subset=["industry_sw1", "fwd_5d", "close"])
        if len(df_work) > 0:
            # Use close return as proxy factor
            df_work = df_work.copy()
            df_work["_ret"] = df_work.groupby("code")["close"].pct_change()
            df_work = df_work.dropna(subset=["_ret"])
            for ind, grp in df_work.groupby("industry_sw1"):
                if len(grp) >= 10:
                    ic, _ = spearmanr(grp["_ret"], grp["fwd_5d"])
                    if not np.isnan(ic):
                        industry_ic[str(ind)] = round(float(ic), 6)
            result["industry_ic"] = dict(
                sorted(industry_ic.items(), key=lambda x: -abs(x[1]))
            )
    else:
        result["industry_ic"] = {"note": "industry_sw1 column not available"}

    # Market cap IC
    if "mcap" in df.columns and "fwd_5d" in df.columns and "close" in df.columns:
        mcap_ic = {}
        df_work = df.dropna(subset=["mcap", "fwd_5d", "close"])
        if len(df_work) > 0:
            df_work = df_work.copy()
            df_work["_ret"] = df_work.groupby("code")["close"].pct_change()
            df_work = df_work.dropna(subset=["_ret"])
            # Bucket by mcap quintile
            df_work["_mcap_q"] = pd.qcut(df_work["mcap"], 5, labels=False, duplicates="drop")
            for q, grp in df_work.groupby("_mcap_q"):
                if len(grp) >= 10:
                    ic, _ = spearmanr(grp["_ret"], grp["fwd_5d"])
                    if not np.isnan(ic):
                        label = f"Q{int(q) + 1}"
                        mcap_ic[label] = round(float(ic), 6)
            result["market_cap_ic"] = mcap_ic
    else:
        result["market_cap_ic"] = {"note": "mcap column not available"}

    return result


def _dim_ml_importance(ml_result: dict) -> dict:
    """Dimension 6: ML feature importance."""
    if ml_result is None or not isinstance(ml_result, dict):
        return {"importances": {}, "ndcg_score": 0.0}

    importances = ml_result.get("feature_importances", {})
    if importances and isinstance(importances, dict):
        sorted_imp = dict(sorted(importances.items(), key=lambda x: -x[1]))
    else:
        sorted_imp = {}

    ndcg = ml_result.get("ndcg_score", 0.0)
    ndcg = round(float(ndcg), 6) if ndcg is not None else 0.0

    return {
        "importances": sorted_imp,
        "ndcg_score": ndcg,
    }


def _dim_backtest(ml_result: dict) -> dict:
    """Dimension 7: Backtest diagnostics."""
    defaults = {
        "annual_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "win_rate": 0.0,
        "n_trades": 0,
        "fitness_score": 0.0,
    }
    if ml_result is None or not isinstance(ml_result, dict):
        return defaults

    bt = ml_result.get("backtest", {})
    if not isinstance(bt, dict):
        bt = {}

    return {
        "annual_return": round(float(bt.get("bt_annual", bt.get("annual_return", 0) or 0)), 4),
        "max_drawdown": round(float(bt.get("bt_max_dd", bt.get("max_drawdown", 0) or 0)), 4),
        "sharpe": round(float(bt.get("bt_sharpe", bt.get("sharpe", 0) or 0)), 4),
        "win_rate": round(float(bt.get("bt_win_rate", bt.get("win_rate", 0) or 0)), 4),
        "n_trades": int(bt.get("bt_n_trades", bt.get("n_trades", 0) or 0)),
        "fitness_score": round(float(bt.get("bt_fitness", bt.get("fitness_score", 0) or 0)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Suggestion Loading & Application
# ═══════════════════════════════════════════════════════════════════════════════

def load_suggestions(path: str = "data/suggestions.json") -> Optional[dict]:
    """Load analyst suggestions JSON.

    Args:
        path: Path to the suggestions JSON file.

    Returns:
        dict or None: The suggestions dict, or None if file missing or invalid.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def apply_suggestions(
    suggestions: dict,
    leaf_pool: list,
    operator_probs: dict,
) -> tuple[list, dict]:
    """Apply analyst suggestions to bias evolution parameters.

    Supports:
        - force_data_source: ["financial", "macro"] -> duplicate matching leaves 4x
        - boost_leaf_prob: {"roe": 0.3} -> duplicate leaf N times
        - boost_operator: ["sector_rank"] -> double probability
        - kill_operator: ["ts_corr"] -> remove from operator_probs
        - cap_max_depth: int -> not returned in probs; caller should handle

    Args:
        suggestions: Dict loaded from load_suggestions().
        leaf_pool: List of leaf token strings.
        operator_probs: Dict of {operator_name: probability}.

    Returns:
        tuple: (modified_leaf_pool, modified_operator_probs)
    """
    new_pool = list(leaf_pool)  # copy
    new_probs = dict(operator_probs)

    # force_data_source: duplicate matching leaves 4x
    if "force_data_source" in suggestions:
        targets = set(suggestions["force_data_source"])
        boosted = [l for l in leaf_pool if _classify_leaf(l) in targets]
        new_pool.extend(boosted * 3)  # 3 extra copies = 4x total

    # boost_leaf_prob: duplicate specific leaves N times
    if "boost_leaf_prob" in suggestions:
        for leaf_name, boost_n in suggestions["boost_leaf_prob"].items():
            if leaf_name in leaf_pool:
                # Add (boost_n - 1) copies; if boost_n is a probability-like
                # fraction, treat it as multiplier on base count
                existing_count = leaf_pool.count(leaf_name)
                add_count = int(existing_count * max(0, float(boost_n)))
                if add_count > 0:
                    new_pool.extend([leaf_name] * add_count)

    # boost_operator: double probability of specified operators
    if "boost_operator" in suggestions:
        for op_name in suggestions["boost_operator"]:
            if op_name in new_probs:
                new_probs[op_name] = new_probs[op_name] * 2.0

    # kill_operator: remove from operator_probs and renormalize
    if "kill_operator" in suggestions:
        for op_name in suggestions["kill_operator"]:
            new_probs.pop(op_name, None)
        total = sum(new_probs.values())
        if total > 0:
            new_probs = {k: v / total for k, v in new_probs.items()}

    return new_pool, new_probs


def save_analysis_report(report: dict, round_num: int) -> str:
    """Save report to data/analysis_round_NNNN.json.

    Args:
        report: The analysis report dict.
        round_num: Current round number.

    Returns:
        str: Path to the saved file.
    """
    os.makedirs("data", exist_ok=True)
    filename = f"data/analysis_round_{round_num:04d}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return filename
