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
import warnings
from loguru import logger
from scipy.stats import spearmanr
from sqlalchemy import text

warnings.filterwarnings('ignore', category=RuntimeWarning)

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

# 可排序的基础列（ts_rank 的参数）
RANKABLE_COLS = ["ret_1d", "ret_5d", "ret_20d", "vol_5d", "vol_20d",
                 "vol_ratio", "amplitude", "seal", "gap", "ma5_dev",
                 "ma20_dev", "ma_spread", "log_mcap", "lu_20d"]
PERIODS = [10, 20, 40, 60, 120]
CATEGORIES = ["动量", "反转", "波动率", "量价", "形态", "均线", "基本面", "涨停", "行业中性", "复合", "杂乱", "非线性", "组合"]


def make_template(name, col, period, sign=1, category="进化"):
    """创建参数化的 ts_rank 模板。"""
    return {
        "name": name,
        "col": col,
        "period": period,
        "sign": sign,
        "category": category,
        "compute": lambda df, c=col, p=period, s=sign: ts_rank(df, c, p) * s,
    }


# 种子模板池
BASE_TEMPLATES = [
    make_template("momentum_5d", "ret_5d", 20, 1, "动量"),
    make_template("momentum_20d", "ret_20d", 40, 1, "动量"),
    make_template("reversal_1d", "ret_1d", 10, -1, "反转"),
    make_template("reversal_5d", "ret_5d", 20, -1, "反转"),
    make_template("vol_5d", "vol_5d", 20, -1, "波动率"),
    make_template("vol_20d", "vol_20d", 40, -1, "波动率"),
    make_template("vol_ratio", "vol_ratio", 20, 1, "量价"),
    make_template("seal_quality", "seal", 20, 1, "形态"),
    make_template("amplitude", "amplitude", 20, -1, "形态"),
    make_template("gap_momentum", "gap", 20, 1, "形态"),
    make_template("ma5_dev", "ma5_dev", 20, 1, "均线"),
    make_template("ma20_dev", "ma20_dev", 40, 1, "均线"),
    make_template("ma_spread", "ma_spread", 20, 1, "均线"),
    make_template("log_mcap", "log_mcap", 60, 1, "基本面"),
    make_template("lu_intensity", "lu_20d", 40, 1, "涨停"),
]

# 特殊模板（非 ts_rank）
SPECIAL_TEMPLATES = {
    "volume_shock":   {"compute": lambda df: df["volume"] / df["volume_ma20"], "category": "量价"},
    "turnover_ratio": {"compute": lambda df: ts_rank(df, "turnover_ratio", 20) if "turnover_ratio" in df.columns else None, "category": "量价"},
    "lu_quality":     {"compute": lambda df: ts_rank(df, "seal", 10) * df["is_lu"], "category": "涨停"},
    "sector_n_mom":   {"compute": lambda df: sector_neutral(df, df["ret_5d"]), "category": "行业中性"},
    "sector_n_vol":   {"compute": lambda df: sector_neutral(df, df["vol_5d"]), "category": "行业中性"},
}


def get_active_templates(pool):
    """获取当前活跃的模板列表（基础+特殊+pool中的进化模板）。"""
    result = {}
    # 基础
    for t in BASE_TEMPLATES:
        result[t["name"]] = t
    # 特殊
    result.update(SPECIAL_TEMPLATES)
    # 进化池
    for t in pool:
        result[t["name"]] = t
    return result


# ══════════════════════════════════════════════════════════════════════
# 头部效应评价（替代全市场 IC）
# ══════════════════════════════════════════════════════════════════════

def compute_top_n_performance(factor_val, df, forward_ret, top_n=5,
                               quality_filter=True):
    """评价因子在头部 N 只股票上的实际表现。

    返回:
      - mean_ret: 头部平均前向收益
      - win_rate: 正收益比例
      - sharpe: 头部收益的 Sharpe (截面)
      - top_ret: 头部收益序列
      - tail_ic: 仅在前20%股票中计算的IC
    """
    dates = sorted(df["trade_date"].unique())
    all_rets = []
    all_wins = []
    ic_top20 = []

    for td in dates[-60:]:  # 最近60个交易日
        mask = df["trade_date"] == td
        if mask.sum() < 100:
            continue

        fv = factor_val[mask].dropna()
        fr = forward_ret[mask].dropna()
        common = fv.index.intersection(fr.index)
        if len(common) < 100:
            continue
        fv = fv.loc[common]
        fr = fr.loc[common]

        # 质量底线：剔除市值最小20% + 成交额最低20%
        if quality_filter:
            mcap_td = df.loc[common, "mcap"] if "mcap" in df.columns else None
            vol_td = df.loc[common, "volume"] if "volume" in df.columns else None
            keep = pd.Series(True, index=common)
            if mcap_td is not None and mcap_td.notna().sum() > 50:
                mcap_cut = mcap_td.quantile(0.2)
                keep &= mcap_td >= mcap_cut
            if vol_td is not None and vol_td.notna().sum() > 50:
                vol_cut = vol_td.quantile(0.2)
                keep &= vol_td >= vol_cut
            fv = fv[keep[common]]
            fr = fr[keep[common]]
            common = fv.index.intersection(fr.index)
            if len(common) < 50:
                continue

        # 因子排序取 top-N
        fv_sorted = fv.sort_values(ascending=False)
        top_idx = fv_sorted.head(top_n).index
        top_ret = fr.loc[top_idx]

        all_rets.extend(top_ret.values)
        all_wins.extend((top_ret > 0).values)

        # Tail IC: 只在 top 20% 中算
        top20_cut = int(len(fv) * 0.2)
        if top20_cut >= 20:
            top20_idx = fv_sorted.head(top20_cut).index
            try:
                with np.errstate(invalid='ignore'):
                    ic, _ = spearmanr(fv.loc[top20_idx], fr.loc[top20_idx])
                if not np.isnan(ic):
                    ic_top20.append(ic)
            except Exception:
                pass

    if len(all_rets) < 20:
        return {"mean_ret": 0, "win_rate": 0, "sharpe": 0,
                "n_samples": len(all_rets), "tail_ic": 0, "status": "insufficient"}

    arr = np.array(all_rets)
    mean_ret = float(np.mean(arr))
    win_rate = float(np.mean(np.array(all_wins)))
    sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252)) if np.std(arr) > 0 else 0
    tail_ic = float(np.mean(ic_top20)) if ic_top20 else 0

    return {
        "mean_ret": round(mean_ret, 4),
        "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 4),
        "n_samples": len(all_rets),
        "tail_ic": round(tail_ic, 4),
        "status": "pass" if len(all_rets) >= 50 and win_rate > 0.45 else "fail",
    }


def compute_multi_factor_intersection(factor_dict, df, forward_ret, top_n=5,
                                       quality_filter=True):
    """多因子交集法：A∩B 交集内取 top-N。

    factor_dict: {name: factor_series}
    先用因子A选前20%→因子B选前20%→交集内用因子C排序取top-N
    """
    if len(factor_dict) < 2:
        return {}

    dates = sorted(df["trade_date"].unique())
    factor_names = list(factor_dict.keys())
    # 取前3个因子做交集
    use_factors = factor_names[:3]
    all_rets = []

    for td in dates[-60:]:
        mask = df["trade_date"] == td
        fr = forward_ret[mask].dropna()
        common = fr.index

        # 质量底线
        if quality_filter:
            mcap_td = df.loc[common, "mcap"] if "mcap" in df.columns else None
            if mcap_td is not None and mcap_td.notna().sum() > 50:
                common = common.intersection(mcap_td[mcap_td >= mcap_td.quantile(0.2)].index)
            if len(common) < 50:
                continue

        # 逐层交集
        pool = set(common)
        for fn in use_factors:
            fv = factor_dict[fn][mask].dropna()
            valid = set(fv.index) & pool
            if len(valid) < 50:
                break
            fv_valid = fv[list(valid)].sort_values(ascending=False)
            cut20 = int(len(fv_valid) * 0.2)
            pool = set(fv_valid.head(cut20).index)

        if len(pool) < top_n:
            continue

        # 在最终交集中用第一个因子排序取 top-N
        final_fv = factor_dict[use_factors[0]][mask].dropna()
        final_pool = list(set(final_fv.index) & pool)
        if len(final_pool) < top_n:
            continue
        top_idx = final_fv[final_pool].sort_values(ascending=False).head(top_n).index
        top_ret = fr.loc[fr.index.intersection(top_idx)]
        all_rets.extend(top_ret.values)

    if len(all_rets) < 20:
        return {"mean_ret": 0, "win_rate": 0, "n_samples": len(all_rets)}

    arr = np.array(all_rets)
    return {
        "mean_ret": round(float(np.mean(arr)), 4),
        "win_rate": round(float(np.mean(arr > 0)), 4),
        "sharpe": round(float(np.mean(arr) / np.std(arr) * np.sqrt(252)), 4)
        if np.std(arr) > 0 else 0,
        "n_samples": len(all_rets),
    }


def mcap_neutral(df, col):
    """市值中性化：因子值 - 同市值分位截面均值。"""
    if "mcap" not in df.columns or df["mcap"].isna().all():
        return df[col] if col in df.columns else None
    # 分5个市值组
    mcap_rank = df.groupby("trade_date")["mcap"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop"))
    group_mean = df.groupby(["trade_date", mcap_rank])[col].transform("mean")
    return df[col] - group_mean


def compute_quantile_spread(factor_val, df, forward_ret, n_quantiles=20):
    """分位数颗粒度：Top组 vs Top-2组收益差。正值=有顶部效应，负值=极值反转。"""
    dates = sorted(df["trade_date"].unique())
    spreads = []
    for td in dates[-60:]:
        mask = df["trade_date"] == td
        fv = factor_val[mask].dropna()
        fr = forward_ret[mask].dropna()
        common = fv.index.intersection(fr.index)
        if len(common) < n_quantiles * 5: continue
        fv_c = fv.loc[common]
        try:
            labels = pd.qcut(fv_c.rank(method="first"), n_quantiles, labels=False, duplicates="drop")
            q1_ret = fr.loc[common][labels == labels.max()].mean()
            q2_ret = fr.loc[common][labels == labels.max() - 1].mean()
            if pd.notna(q1_ret) and pd.notna(q2_ret):
                spreads.append(q1_ret - q2_ret)
        except Exception:
            continue
    return np.mean(spreads) if spreads else 0


def generate_nonlinear_combos(df, factor_dict, top_n=6):
    """从 Top 因子生成非线性组合：乘积/平方/Sigmoid。"""
    scored = []
    for n, fv in factor_dict.items():
        s = fv.dropna()
        scored.append((len(s), n))
    scored.sort(reverse=True)
    top = [n for _, n in scored[:top_n]]

    combos = {}
    for i in range(min(len(top), 5)):
        for j in range(i + 1, min(len(top), 5)):
            n1, n2 = top[i], top[j]
            if n1 in factor_dict and n2 in factor_dict:
                v1 = _safe_norm(factor_dict[n1])
                v2 = _safe_norm(factor_dict[n2])
                name = f"cross_{n1[:8]}x{n2[:8]}"
                combos[name] = {"compute": lambda df, a=v1, b=v2: a * b, "category": "非线性"}

    for n in top[:4]:
        if n in factor_dict:
            v = _safe_norm(factor_dict[n])
            combos[f"{n[:10]}_sq"] = {"compute": lambda df, x=v: x ** 2, "category": "非线性"}
            combos[f"{n[:10]}_sig"] = {
                "compute": lambda df, x=v: 1 / (1 + np.exp(-x * 3)), "category": "非线性"}

    return combos


def train_ml_ranker(df, factor_dict, top_n=5):
    """XGBoost 学习头部特征：用因子截面 rank 预测 5 日涨 > 2%。"""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, {}

    feature_cols = list(factor_dict.keys())
    if len(feature_cols) < 3:
        return None, {}

    # 构建训练集：每只股票每天的特征 = 各因子的截面 rank
    df["fwd_5d_up"] = (df.groupby("code")["close"].pct_change(5).shift(-5) > 0.02).astype(int)
    dates = sorted(df["trade_date"].unique())

    X_list, y_list = [], []
    for td in dates[-90:]:
        mask = df["trade_date"] == td
        if mask.sum() < 200: continue
        feats = {}
        for fn in feature_cols:
            if fn in factor_dict:
                fv = factor_dict[fn][mask].dropna()
                feats[fn] = fv.rank(pct=True)
        if len(feats) < 3: continue
        feat_df = pd.DataFrame(feats)
        feat_df = feat_df.dropna()
        if len(feat_df) < 100: continue
        y_td = df.loc[feat_df.index, "fwd_5d_up"].dropna()
        common = feat_df.index.intersection(y_td.index)
        if len(common) < 100: continue
        X_list.append(feat_df.loc[common])
        y_list.append(y_td.loc[common])

    if len(X_list) < 10:
        return None, {}

    X = pd.concat(X_list)
    y = pd.concat(y_list)
    if len(X) < 500:
        return None, {}

    model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, verbosity=0)
    model.fit(X, y)

    # 在最近 20 天评估胜率
    ml_rets, ml_wins = [], []
    for td in dates[-20:]:
        mask = df["trade_date"] == td
        feats = {}
        for fn in feature_cols:
            if fn in factor_dict:
                fv = factor_dict[fn][mask].dropna()
                feats[fn] = fv.rank(pct=True)
        if len(feats) < 3: continue
        feat_df = pd.DataFrame(feats).dropna()
        if len(feat_df) < 100: continue
        proba = model.predict_proba(feat_df[feature_cols])[:, 1]
        top_idx = pd.Series(proba, index=feat_df.index).sort_values(ascending=False).head(top_n).index
        fr = df.loc[mask, "fwd_5d_up"].dropna()
        r = fr.loc[fr.index.intersection(top_idx)]
        ml_rets.extend(r.values)  # 这里是二元标签
        ml_wins.extend((r > 0).values if len(r) > 0 else [])

    ml_wr = np.mean(ml_wins) if ml_wins else 0
    return model, {"ml_win_rate": round(float(ml_wr), 4), "ml_samples": len(ml_wins)}


def validate_all_v3(df, pool, top_n=5):
    """v3.0 主评价循环：单因子头部效应 + 分位数 + 中性化 + 非线性 + 交集 + ML。"""
    df = df.sort_values(["code", "trade_date"])
    df["fwd_5d"] = df.groupby("code")["close"].pct_change(5).shift(-5)
    df["fwd_10d"] = df.groupby("code")["close"].pct_change(10).shift(-10)

    templates = get_active_templates(pool)
    results = {}
    factor_values = {}

    # ── Phase 1: 单因子评价 ──
    for ti, (name, tmpl) in enumerate(templates.items()):
        logger.info(f"  [{ti+1}/{len(templates)}] {name} ...")
        try:
            fv_raw = tmpl["compute"](df)
        except Exception as e:
            results[name] = {"win_rate": 0, "mean_ret": 0, "status": "compute_error",
                            "category": tmpl.get("category", "?")}
            continue
        if fv_raw is None:
            results[name] = {"win_rate": 0, "mean_ret": 0, "status": "no_data",
                            "category": tmpl.get("category", "?")}
            continue

        fv = fv_raw
        factor_values[name] = fv

        # 头部效应
        perf = compute_top_n_performance(fv, df, df["fwd_5d"], top_n, quality_filter=True)
        perf_10 = compute_top_n_performance(fv, df, df["fwd_10d"], top_n, quality_filter=True)
        q_spread = compute_quantile_spread(fv, df, df["fwd_5d"])

        n = perf.get("n_samples", 0)
        wr = perf.get("win_rate", 0)
        mr = perf.get("mean_ret", 0)
        status = "pass" if (n >= 50 and wr >= 0.50) else "fail"

        results[name] = {
            "win_rate": round(wr, 4),
            "mean_ret": round(mr, 4),
            "win_rate_10d": round(perf_10.get("win_rate", 0), 4),
            "mean_ret_10d": round(perf_10.get("mean_ret", 0), 4),
            "sharpe": perf.get("sharpe", 0),
            "tail_ic": perf.get("tail_ic", 0),
            "quantile_spread": round(float(q_spread), 4),
            "n_samples": n,
            "status": status,
            "category": tmpl.get("category", "?"),
        }

    # ── Phase 2: 非线性因子 ──
    passed = {n: r for n, r in results.items() if r.get("n_samples", 0) >= 50}
    if len(passed) >= 3:
        combos = generate_nonlinear_combos(df, factor_values)
        for cname, ctmpl in combos.items():
            try:
                fv = ctmpl["compute"](df)
                if fv is None: continue
                factor_values[cname] = fv
                perf = compute_top_n_performance(fv, df, df["fwd_5d"], top_n, quality_filter=True)
                wr = perf.get("win_rate", 0)
                results[cname] = {
                    "win_rate": round(wr, 4),
                    "mean_ret": round(perf.get("mean_ret", 0), 4),
                    "n_samples": perf.get("n_samples", 0),
                    "status": "pass" if (perf.get("n_samples", 0) >= 50 and wr >= 0.50) else "fail",
                    "category": "非线性",
                }
            except Exception:
                pass

    # ── Phase 3: 多因子交集 ──
    scored = [(n, r["win_rate"] * r.get("mean_ret", 0))
              for n, r in results.items() if r.get("n_samples", 0) >= 50 and r["win_rate"] > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_names = [n for n, _ in scored[:6]]
    if len(top_names) >= 3:
        inter = compute_multi_factor_intersection(
            {n: factor_values[n] for n in top_names[:3] if n in factor_values},
            df, df["fwd_5d"], top_n, quality_filter=True)
        if inter.get("n_samples", 0) >= 20:
            wr_i = inter.get("win_rate", 0)
            results["multi_intersect"] = {
                "win_rate": round(wr_i, 4),
                "mean_ret": round(inter.get("mean_ret", 0), 4),
                "sharpe": inter.get("sharpe", 0),
                "n_samples": inter.get("n_samples", 0),
                "status": "pass" if wr_i >= 0.50 else "fail",
                "category": "多因子交集",
                "factors": top_names[:3],
            }
            logger.info(f"  交集: wr={wr_i:.1%} ret={inter.get('mean_ret',0):+.2%}")

    # ── Phase 4: ML 评分 ──
    model, ml_result = train_ml_ranker(df, factor_values, top_n)
    if ml_result:
        ml_wr = ml_result.get("ml_win_rate", 0)
        results["xgb_ranker"] = {
            "win_rate": round(ml_wr, 4),
            "mean_ret": 0,
            "n_samples": ml_result.get("ml_samples", 0),
            "status": "pass" if ml_wr >= 0.50 else "fail",
            "category": "ML",
        }
        logger.info(f"  XGBoost: wr={ml_wr:.1%}")

    return results, pool

def mutate_template(tmpl):
    """变异一个模板的参数。"""
    new = dict(tmpl)
    if "col" in new:
        if random.random() < 0.4:
            new["col"] = random.choice(RANKABLE_COLS)
        if random.random() < 0.5:
            delta = random.choice([-1, 1, -2, 2]) * random.choice([10, 20, 40])
            new["period"] = max(5, min(240, new.get("period", 20) + delta))
    if random.random() < 0.2 and "sign" in new:
        new["sign"] = -new.get("sign", 1)
    if random.random() < 0.3:
        new["category"] = random.choice(CATEGORIES[:6])
    new["name"] = f"evo_{new.get('col','?')[:8]}_p{new.get('period',20)}_s{new.get('sign',1)}_g{random.randint(0,999)}"
    if "compute" in new:
        del new["compute"]  # 重新生成
    new["compute"] = lambda df, c=new["col"], p=new["period"], s=new.get("sign", 1): ts_rank(df, c, p) * s
    return new


def crossover_templates(t1, t2):
    """两个模板交叉。"""
    child = {}
    for key in ["col", "period", "sign", "category"]:
        parent = random.choice([t1, t2])
        if key in parent:
            child[key] = parent[key]
    child["name"] = f"evo_x_{child.get('col','?')[:6]}_p{child.get('period',20)}_g{random.randint(0,999)}"
    child["compute"] = lambda df, c=child["col"], p=child["period"], s=child.get("sign", 1): ts_rank(df, c, p) * s
    return child


def evolve_pool(pool, results, max_pool=30):
    """根据 IC 结果进化模板池：保留高分 + 变异 + 交叉 + 随机注入。"""
    scored = []
    for t in pool:
        r = results.get(t["name"], {})
        # 适应度 = win_rate × (1 + mean_ret)，胜率≥50%才有正分
        wr = r.get("win_rate", 0)
        mr = r.get("mean_ret", 0)
        score = wr * (1 + mr) if wr >= 0.45 else 0
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 保留 top 10
    new_pool = [t for _, t in scored[:10]]

    # 变异 top 5
    for _, t in scored[:5]:
        for _ in range(2):
            new_pool.append(mutate_template(t))

    # 交叉 top 5
    top5 = [t for _, t in scored[:5]]
    for i in range(min(3, len(top5))):
        for j in range(i+1, min(5, len(top5))):
            new_pool.append(crossover_templates(top5[i], top5[j]))

    # 随机注入新模板
    for _ in range(5):
        col = random.choice(RANKABLE_COLS)
        period = random.choice(PERIODS)
        sign = random.choice([1, -1])
        new_t = make_template(
            f"rnd_{col[:6]}_p{period}_s{sign}_g{random.randint(0,999)}",
            col, period, sign, random.choice(CATEGORIES[:6]))
        new_pool.append(new_t)

    # 去重 + 限制数量
    seen = set()
    unique = []
    for t in new_pool:
        h = f"{t.get('col','')}_{t.get('period','')}_{t.get('sign','')}"
        if h not in seen:
            seen.add(h)
            unique.append(t)

    logger.info(f"  模板池: {len(pool)}→{len(unique)} (存{len(scored[:10])}+变异+交叉+随机)")
    return unique[:max_pool]


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

    valid = [(n, r) for n, r in results.items() if r.get("n_samples", 0) >= 50]
    valid.sort(key=lambda x: x[1].get("win_rate", 0), reverse=True)

    table = Table(title=f"🧪 因子进化 — Round {round_num}", title_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("因子", style="bright_blue", max_width=28)
    table.add_column("胜率", justify="right", style="green")
    table.add_column("均收益", justify="right", style="yellow")
    table.add_column("n", justify="right", style="dim", width=5)
    table.add_column("类别")

    for i, (name, r) in enumerate(valid[:10], 1):
        table.add_row(
            str(i), name[:25],
            f"{r.get('win_rate',0):.1%}",
            f"{r.get('mean_ret',0):+.2%}",
            str(r.get('n_samples',0)),
            r.get('category','?'),
        )

    best_wr = max((r.get('win_rate',0) for _, r in valid), default=0)

    return Panel(
        table,
        title=f"[bold]通过: {sum(1 for _,r in valid if r.get('status')=='pass')}/{len(valid)}  |  最佳胜率: {best_wr:.1%}  |  {elapsed:.0f}s[/]",
        border_style="green" if best_wr >= 0.50 else "blue",
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
            all_v = [(n, r) for n, r in results.items() if r.get("n_samples", 0) >= 50]
            print(f"最近一轮: {len(passed)}通过/{len(all_v)}有效")
            all_v.sort(key=lambda x: x[1].get("win_rate", 0), reverse=True)
            for i, (n, r) in enumerate(all_v[:args.top], 1):
                print(f"  {i:2d}. {n:<30s} WR={r.get('win_rate',0):.1%} ret={r.get('mean_ret',0):+.2%} "
                      f"n={r.get('n_samples',0)} [{r.get('category','?')}]")
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
    top_n = 5  # 头部选股数
    t0 = time.time()
    console = Console() if HAS_RICH else None

    pool = []  # 进化模板池

    for r in range(args.rounds):
        round_num = db["rounds"] + 1
        logger.info(f"═══ Round {round_num} ═══")

        results, pool = validate_all_v3(df, pool, top_n=top_n)

        # 进化模板池
        if round_num >= 1:
            pool = evolve_pool(pool, results)

        entry = {"round": round_num, "date": str(pd.Timestamp.now())[:19],
                 "results": results, "pool_size": len(pool)}
        db["history"].append(entry)
        db["rounds"] = round_num

        passed = sum(1 for r in results.values() if r.get("status") == "pass")
        valid_cnt = sum(1 for r in results.values() if r.get("n_samples", 0) >= 50)
        best_wr = max((r.get("win_rate", 0) for r in results.values()
                       if r.get("n_samples", 0) >= 50), default=0)
        best_ret = max((r.get("mean_ret", 0) for r in results.values()
                        if r.get("n_samples", 0) >= 50), default=0)

        if round_num % 5 == 0:
            rules = extract_rules(db)
            if rules:
                with open(RULES_FILE, "w") as f:
                    for k, v in sorted(rules.items()):
                        f.write(f"- **{k}**: {v}\n")
        save_db(db)

        elapsed = time.time() - t0
        logger.info(f"  通过: {passed}/{valid_cnt}, 最佳WR: {best_wr:.1%} ret: {best_ret:+.2%}, 池: {len(pool)}, {elapsed:.0f}s")

        if HAS_RICH and round_num % 10 == 0:
            panel = _render_panel(db, round_num, elapsed)
            console.print(panel)

    elapsed = time.time() - t0
    logger.success(f"完成: {args.rounds} 轮, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
