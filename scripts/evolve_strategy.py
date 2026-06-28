#!/usr/bin/env python
"""策略自进化引擎 v3.0 —— 对接 bt_yaogu.py 真实回测。

每轮:
  1. 加载全量日线 + 妖股信号
  2. 对每个基因组: 过滤信号 → 跑 bt_yaogu 真实回测 → 得年化/Sharpe/MDD
  3. 按年化收益排序，精英保留，变异+交叉 → 下一轮

用法:
  python scripts/evolve_strategy.py --target 0.50    # 目标年化 50%
  python scripts/evolve_strategy.py --status          # 查看进度
"""

from __future__ import annotations

import sys, os, json, argparse, time, hashlib, random
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data
from config.settings import TradingConfig

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── 路径 ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_DB = os.path.join(BASE, 'data', 'strategy_db.json')
RULES_FILE = os.path.join(BASE, 'data', 'strategy_rules.md')

# ── 缓存 ──
_CACHE = None


def _load_cache(force=False):
    """加载全量日线 + 妖股信号（全局缓存）。"""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    engine = get_engine()
    logger.info("加载全量日线数据...")
    with engine.connect() as conn:
        codes = pd.read_sql(text(
            "SELECT code FROM stock_basic WHERE is_st=FALSE "
            "AND code !~ '^(300|301|688|[48])'"), conn)
    all_codes = [str(c).zfill(6) for c in codes['code'].tolist()]
    logger.info(f"股票池: {len(all_codes)} 只")

    daily = load_daily_data(engine, all_codes, "2019-01-01", str(date.today()),
                            cols=["open", "high", "low", "close", "volume", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["code", "trade_date"])
    daily["ret"] = daily.groupby("code")["close"].pct_change()

    # 加载妖股全量信号
    yaogu = pd.read_csv(os.path.join(BASE, 'data/signals/bt_signals_yaogu_full.csv'))
    yaogu["date"] = pd.to_datetime(yaogu["date"])
    yaogu["code"] = yaogu["code"].astype(str).str.zfill(6)
    # 仅主板
    yaogu = yaogu[~yaogu["code"].str.startswith(('300', '301', '688', '4', '8'))]
    logger.info(f"妖股信号: {len(yaogu)} 条, 日线: {len(daily)} 行")

    engine.dispose()
    _CACHE = {"daily": daily, "yaogu": yaogu}
    return _CACHE


# ══════════════════════════════════════════════════════════════════════
# 基因组
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StrategyGenome:
    """策略变体 —— 所有参数都是信号过滤条件。"""
    yaogu_score_min: int = 3
    low_vol_streak_min: int = 0
    lu_20d_min: int = 0
    lu_20d_max: int = 999
    lu_60d_max: int = 999
    ma_bullish: bool = False
    require_lu_day: bool = False
    amplitude_max: float = 0.20
    seal_quality_min: float = 0.0

    # ── 调仓参数 ──
    top_n: int = 5
    rebalance_days: int = 5
    trailing_stop: float = 0.12
    min_hold_days: int = 7

    # ── 复合特征 ──
    entry_quality_min: float = 0.0     # 封板×缩量/振幅
    seal_streak_min: float = 0.0       # 封板×缩量
    lu_efficiency_max: float = 99.0    # 涨停效率上限
    lu_streak_min: int = 0            # 连板天数

    generation: int = 0
    parent_hash: str = ""

    _DEFAULTS = {
        'yaogu_score_min': 3, 'low_vol_streak_min': 0,
        'lu_20d_min': 0, 'lu_20d_max': 999, 'lu_60d_max': 999,
        'ma_bullish': False, 'require_lu_day': False,
        'amplitude_max': 0.20, 'seal_quality_min': 0.0,
        'top_n': 5, 'rebalance_days': 5, 'trailing_stop': 0.12, 'min_hold_days': 7,
        'entry_quality_min': 0.0, 'seal_streak_min': 0.0,
        'lu_efficiency_max': 99.0, 'lu_streak_min': 0,
    }

    def active_params(self) -> dict:
        active = {}
        for k, default in self._DEFAULTS.items():
            val = getattr(self, k)
            if isinstance(default, float) and isinstance(val, float):
                if abs(val - default) > 0.001: active[k] = val
            elif val != default: active[k] = val
        return active

    def genome_hash(self) -> str:
        s = json.dumps(self.active_params(), sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:10]

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        for k in d:
            if d[k] == float('inf'): d[k] = "inf"
            elif d[k] == float('-inf'): d[k] = "-inf"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'StrategyGenome':
        d2 = d.copy()
        for k in d2:
            if d2[k] == "inf": d2[k] = float('inf')
            elif d2[k] == "-inf": d2[k] = float('-inf')
        g = cls()
        for k, v in d2.items():
            if hasattr(g, k) and not callable(getattr(g, k)):
                setattr(g, k, v)
        return g

    def condition_desc(self) -> str:
        parts = []
        ap = self.active_params()
        if 'yaogu_score_min' in ap: parts.append(f"妖股≥{self.yaogu_score_min}")
        if 'low_vol_streak_min' in ap: parts.append(f"缩量≥{self.low_vol_streak_min}d")
        if 'lu_20d_min' in ap: parts.append(f"涨停≥{self.lu_20d_min}")
        if 'lu_20d_max' in ap: parts.append(f"涨停≤{self.lu_20d_max}")
        if 'lu_60d_max' in ap: parts.append(f"60日≤{self.lu_60d_max}")
        if 'ma_bullish' in ap: parts.append("MA多头")
        if 'require_lu_day' in ap: parts.append("当日涨停")
        if 'amplitude_max' in ap: parts.append(f"振幅≤{self.amplitude_max:.0%}")
        if 'seal_quality_min' in ap: parts.append(f"封板≥{self.seal_quality_min:.0%}")
        # 调仓参数
        if 'top_n' in ap: parts.append(f"持仓{self.top_n}")
        if 'rebalance_days' in ap: parts.append(f"调仓{self.rebalance_days}d")
        if 'trailing_stop' in ap: parts.append(f"止盈{self.trailing_stop:.0%}")
        if 'min_hold_days' in ap: parts.append(f"持≥{self.min_hold_days}d")
        if 'entry_quality_min' in ap: parts.append(f"入场质量≥{self.entry_quality_min:.0f}")
        if 'seal_streak_min' in ap: parts.append(f"封板×缩量≥{self.seal_streak_min:.0f}")
        if 'lu_efficiency_max' in ap: parts.append(f"涨停效率≤{self.lu_efficiency_max:.1f}")
        if 'lu_streak_min' in ap: parts.append(f"连板≥{self.lu_streak_min}")
        return " + ".join(parts) if parts else "基线"


# ══════════════════════════════════════════════════════════════════════
# 特征计算（在信号上附加过滤列）
# ══════════════════════════════════════════════════════════════════════

def attach_features(sig_df, daily_df):
    """给信号 DataFrame 附加特征列，用于基因组过滤。"""
    daily = daily_df.sort_values(["code", "trade_date"]).copy()

    # 预计算
    daily["prev_close"] = daily.groupby("code")["close"].shift(1)
    daily["is_lu"] = daily.apply(
        lambda r: 1 if (pd.notna(r["close"]) and pd.notna(r["prev_close"])
                        and r["prev_close"] > 0
                        and TradingConfig.is_at_limit_up(r["close"], r["prev_close"],
                                                         str(r["code"]), tolerance=0.98)) else 0,
        axis=1)
    daily["lu_20d"] = daily.groupby("code")["is_lu"].transform(
        lambda x: x.rolling(20, min_periods=10).sum())
    daily["lu_60d"] = daily.groupby("code")["is_lu"].transform(
        lambda x: x.rolling(60, min_periods=30).sum())
    daily["ma5"] = daily.groupby("code")["close"].transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    daily["ma10"] = daily.groupby("code")["close"].transform(
        lambda x: x.rolling(10, min_periods=10).mean())
    daily["ma20"] = daily.groupby("code")["close"].transform(
        lambda x: x.rolling(20, min_periods=20).mean())
    daily["vol_ma20"] = daily.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=5).mean())

    daily_by_code_date = daily.set_index(["code", "trade_date"])

    total = len(sig_df)
    features = []
    for si, (_, sig) in enumerate(sig_df.iterrows()):
        if si > 0 and si % 5000 == 0:
            logger.info(f"  特征进度: {si}/{total} ({si/total*100:.0f}%)")
        code = sig["code"]
        sig_date = sig["date"]

        try:
            today = daily_by_code_date.loc[(code, sig_date)]
        except KeyError:
            continue

        pre = daily[(daily["code"] == code) & (daily["trade_date"] < sig_date)].tail(60)
        if len(pre) < 5: continue

        # 缩量
        vol_ma = pre["volume"].tail(20).mean() if len(pre) >= 20 else pre["volume"].mean()
        low_vol_streak = 0
        if vol_ma > 0:
            for _, pr in pre.iterrows():
                if pr["volume"] < vol_ma * 0.7: low_vol_streak += 1
                else: low_vol_streak = 0

        lu_20d = int(today.get("lu_20d", 0)) if pd.notna(today.get("lu_20d")) else 0
        lu_60d = int(today.get("lu_60d", 0)) if pd.notna(today.get("lu_60d")) else 0
        ma5 = today.get("ma5", np.nan)
        ma10 = today.get("ma10", np.nan)
        ma20 = today.get("ma20", np.nan)
        ma_bullish = (pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20)
                      and ma5 > ma10 > ma20)
        sig_ret = float(today.get("ret", 0)) if pd.notna(today.get("ret")) else 0
        is_lu_day = sig_ret >= 0.095
        amplitude = float((today["high"] - today["low"]) /
                          (today["prev_close"] if pd.notna(today.get("prev_close"))
                           and today["prev_close"] > 0 else today["close"] or 1)
                          if pd.notna(today.get("high")) and pd.notna(today.get("low")) else 0.50)
        seal = float(today["close"] / today["high"]) if (pd.notna(today.get("high"))
                     and today["high"] > 0) else 0.0

        # 连板天数
        lu_streak_val = 0
        for _, pr in pre.iloc[::-1].iterrows():
            if pd.notna(pr.get('ret')) and pr['ret'] >= 0.095: lu_streak_val += 1
            else: break
        if is_lu_day: lu_streak_val += 1

        # 复合特征
        entry_quality = seal * low_vol_streak / (amplitude + 0.001)
        seal_streak = seal * low_vol_streak
        lu_efficiency = lu_20d / max(lu_streak_val, 1) if lu_20d > 0 else 0

        features.append({
            "code": code, "date": sig_date,
            "score": float(sig.get("score", 0)) if pd.notna(sig.get("score")) else 0,
            "low_vol_streak": low_vol_streak,
            "lu_20d": lu_20d, "lu_60d": lu_60d,
            "ma_bullish": ma_bullish,
            "is_lu_day": is_lu_day, "sig_ret": sig_ret,
            "amplitude": amplitude, "seal_quality": seal,
            "lu_streak": lu_streak_val,
            "entry_quality": round(entry_quality, 1),
            "seal_streak": round(seal_streak, 1),
            "lu_efficiency": round(lu_efficiency, 2),
        })

    return pd.DataFrame(features)


def filter_signals(sig_df, genome, feats_df):
    """按基因组条件过滤特征，返回通过过滤的信号(code,date,score)。"""
    df = feats_df.copy()  # feats_df 已有 score 列
    m = pd.Series(True, index=df.index)

    ap = genome.active_params()
    if 'yaogu_score_min' in ap:
        m &= df["score"] >= genome.yaogu_score_min
    if 'low_vol_streak_min' in ap:
        m &= df["low_vol_streak"] >= genome.low_vol_streak_min
    if 'lu_20d_min' in ap:
        m &= df["lu_20d"] >= genome.lu_20d_min
    if 'lu_20d_max' in ap:
        m &= df["lu_20d"] <= genome.lu_20d_max
    if 'lu_60d_max' in ap:
        m &= df["lu_60d"] <= genome.lu_60d_max
    if 'ma_bullish' in ap:
        m &= df["ma_bullish"] == True
    if 'require_lu_day' in ap:
        m &= df["is_lu_day"] == True
    if 'amplitude_max' in ap:
        m &= df["amplitude"] <= genome.amplitude_max
    if 'seal_quality_min' in ap:
        m &= df["seal_quality"] >= genome.seal_quality_min
    if 'entry_quality_min' in ap:
        m &= df["entry_quality"] >= genome.entry_quality_min
    if 'seal_streak_min' in ap:
        m &= df["seal_streak"] >= genome.seal_streak_min
    if 'lu_efficiency_max' in ap:
        m &= df["lu_efficiency"] <= genome.lu_efficiency_max
    if 'lu_streak_min' in ap:
        m &= df["lu_streak"] >= genome.lu_streak_min

    return df[m][["code", "date", "score"]].copy()


# ══════════════════════════════════════════════════════════════════════
# 适应度评估
# ══════════════════════════════════════════════════════════════════════

def evaluate_genome(genome, feats_df, daily_df, cash=1_000_000):
    """跑真实回测，返回指标。使用基因组的调仓参数。"""
    from scripts.bt_yaogu import run_backtest_on_signals

    filtered = filter_signals(None, genome, feats_df)
    n_signals = len(filtered)

    if n_signals < 20:
        return {"n_signals": n_signals, "error": "信号不足", "fitness": -1}

    result = run_backtest_on_signals(
        filtered, daily_df, name_map={},
        top_n=genome.top_n, cash=cash, min_score=0,
        trailing_stop=genome.trailing_stop,
        min_hold_days=genome.min_hold_days,
        rebalance_days=genome.rebalance_days,
    )

    if "error" in result:
        result["n_signals"] = n_signals
        result["fitness"] = -1
        return result

    # 适应度 = 年化收益为主 + Sharpe奖金 + 样本量惩罚
    ann = result.get("ret_annual", -1)
    sharpe = result.get("sharpe", 0)
    n_trades = result.get("n_trades", 0)

    fitness = ann * 0.7 + max(sharpe, 0) * 0.15 + min(n_trades / 50, 1.0) * 0.15

    return {
        "n_signals": n_signals,
        "n_trades": result.get("n_trades", 0),
        "ret_annual": ann,
        "sharpe": result.get("sharpe", 0),
        "max_dd": result.get("max_dd", 0),
        "win_rate": result.get("win_rate", 0),
        "ret_total": result.get("ret_total", 0),
        "fitness": round(float(fitness), 4),
    }


# ══════════════════════════════════════════════════════════════════════
# 变异 / 交叉 / 种子
# ══════════════════════════════════════════════════════════════════════

def mutate(genome, generation, temp=1.0):
    g = deepcopy(genome)
    g.generation = generation
    g.parent_hash = genome.genome_hash()

    int_fields = {'yaogu_score_min': (0, 9), 'low_vol_streak_min': (0, 15),
                  'lu_20d_min': (0, 10), 'lu_20d_max': (1, 999), 'lu_60d_max': (1, 999),
                  'top_n': (1, 15), 'rebalance_days': (1, 20), 'min_hold_days': (3, 20),
                  'lu_streak_min': (0, 15)}
    for f, (lo, hi) in int_fields.items():
        if random.random() < 0.4 * temp:
            delta = random.choice([-1, 1, -2, 2])
            setattr(g, f, max(lo, min(hi, getattr(g, f) + delta)))

    float_fields = {'amplitude_max': (0.03, 0.20), 'seal_quality_min': (0.0, 1.0),
                    'trailing_stop': (0.05, 0.30),
                    'entry_quality_min': (0, 500), 'seal_streak_min': (0, 50),
                    'lu_efficiency_max': (0.5, 99)}
    for f, (lo, hi) in float_fields.items():
        if random.random() < 0.4 * temp:
            delta = random.choice([-0.02, 0.02, -0.05, 0.05])
            setattr(g, f, max(lo, min(hi, round(getattr(g, f) + delta, 3))))

    bool_fields = ['ma_bullish', 'require_lu_day']
    for f in bool_fields:
        if random.random() < 0.15 * temp:
            setattr(g, f, not getattr(g, f))

    if random.random() < 0.1 * temp:
        f = random.choice(list(int_fields.keys()))
        lo, hi = int_fields[f]
        setattr(g, f, random.randint(lo, hi))

    return g


def crossover(g1, g2, generation):
    child = StrategyGenome()
    child.generation = generation
    child.parent_hash = g1.genome_hash()[:5] + "_" + g2.genome_hash()[:5]
    fields = ['yaogu_score_min', 'low_vol_streak_min', 'lu_20d_min', 'lu_20d_max',
              'lu_60d_max', 'ma_bullish', 'require_lu_day',
              'amplitude_max', 'seal_quality_min',
              'top_n', 'rebalance_days', 'trailing_stop', 'min_hold_days',
              'entry_quality_min', 'seal_streak_min', 'lu_efficiency_max', 'lu_streak_min']
    for f in fields:
        setattr(child, f, getattr(random.choice([g1, g2]), f))
    return child


def seed_genomes():
    seeds = []
    for score in [3, 5, 6]:
        for streak in [0, 1, 3, 6]:
            seeds.append(StrategyGenome(yaogu_score_min=score, low_vol_streak_min=streak))
    for ma in [True, False]:
        seeds.append(StrategyGenome(yaogu_score_min=5, low_vol_streak_min=3, ma_bullish=ma))
    for lu in [True, False]:
        seeds.append(StrategyGenome(yaogu_score_min=6, low_vol_streak_min=3,
                                    ma_bullish=True, require_lu_day=lu))
    for amp in [0.05, 0.08, 0.12]:
        seeds.append(StrategyGenome(yaogu_score_min=5, amplitude_max=amp, low_vol_streak_min=3))
    for seal in [0.8, 0.9, 0.95]:
        seeds.append(StrategyGenome(yaogu_score_min=5, seal_quality_min=seal))
    # 调仓参数变体
    for top_n in [3, 5, 8, 10]:
        for reb in [3, 5, 10]:
            seeds.append(StrategyGenome(yaogu_score_min=5, low_vol_streak_min=3,
                                        top_n=top_n, rebalance_days=reb))
    for ts in [0.08, 0.12, 0.18, 0.25]:
        for mh in [5, 7, 10]:
            seeds.append(StrategyGenome(yaogu_score_min=5, low_vol_streak_min=3,
                                        trailing_stop=ts, min_hold_days=mh))

    for _ in range(10):
        seeds.append(StrategyGenome(
            yaogu_score_min=random.randint(0, 7),
            low_vol_streak_min=random.randint(0, 10),
            lu_20d_min=random.randint(0, 5),
            ma_bullish=random.random() < 0.5,
            require_lu_day=random.random() < 0.3,
            top_n=random.choice([3, 5, 8, 10]),
            rebalance_days=random.choice([3, 5, 10, 15]),
            trailing_stop=random.choice([0.08, 0.12, 0.18, 0.25]),
            min_hold_days=random.choice([5, 7, 10, 15]),
        ))
    return seeds


# ══════════════════════════════════════════════════════════════════════
# DB / 可视化
# ══════════════════════════════════════════════════════════════════════

def load_db():
    if os.path.exists(STRATEGY_DB):
        with open(STRATEGY_DB) as f:
            return json.load(f)
    return {"variants": [], "rounds": 0, "best_ann": 0}


def save_db(db):
    variants = db.get('variants', [])
    if len(variants) > 200:
        variants.sort(key=lambda v: v.get('fitness', 0), reverse=True)
        db['variants'] = variants[:200]
    tmp = STRATEGY_DB + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(db, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, STRATEGY_DB)


def _render_panel(db, round_num, best_ann, target, elapsed, baseline):
    valid = [v for v in db.get('variants', []) if v.get('n_trades', 0) >= 5]
    valid.sort(key=lambda v: v.get('fitness', 0), reverse=True)
    stag = db.get('stagnation_counter', 0)
    total = len(db.get('variants', []))

    pct = min(best_ann / target, 1.0) if target > 0 else 0
    bar = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))

    table = Table(title=f"策略进化 — Round {round_num}", title_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("年化", justify="right", style="green")
    table.add_column("Sharpe", justify="right", style="yellow")
    table.add_column("MDD", justify="right", style="red")
    table.add_column("胜率", justify="right")
    table.add_column("交易", justify="right", style="dim")
    table.add_column("条件", style="bright_blue", max_width=45)

    for i, v in enumerate(valid[:8], 1):
        ann = v.get('ret_annual', 0)
        table.add_row(
            str(i), f"{ann:+.1%}", f"{v.get('sharpe',0):.2f}",
            f"{v.get('max_dd',0):.1%}", f"{v.get('win_rate',0):.1%}",
            str(v.get('n_trades',0)),
            v.get('condition_desc', '?')[:42],
        )

    return Panel(
        table,
        title=f"[bold]目标年化: {target:.0%}  |  基线: {baseline:.1%}  |  "
              f"最佳: {best_ann:.1%}  |  停滞: {stag}轮  |  {elapsed:.0f}s[/]",
        subtitle=f"[{bar}] {best_ann:.1%}",
        border_style="green" if best_ann >= target else "blue",
    )


# ══════════════════════════════════════════════════════════════════════
# 进化主循环
# ══════════════════════════════════════════════════════════════════════

def evolve_round(db, feats_df, daily_df, round_num, elite_count=5):
    logger.info(f"═══ Round {round_num} ═══")
    variants = db.get("variants", [])
    valid_variants = [v for v in variants if v.get('n_trades', 0) >= 5]

    # 收敛检测
    if len(valid_variants) >= 5:
        prev_best = db.get('prev_best_fitness', 0)
        best_now = valid_variants[0].get('fitness', 0) if valid_variants else 0
        sc = db.get('stagnation_counter', 0)
        if abs(best_now - prev_best) < 0.001:
            sc += 1
        else:
            sc = 0
        db['stagnation_counter'] = sc
        db['prev_best_fitness'] = best_now

        if sc >= 10:
            logger.warning(f"停滞 {sc} 轮，注入随机变体")
            for _ in range(15):
                g = StrategyGenome(
                    yaogu_score_min=random.randint(0, 8),
                    low_vol_streak_min=random.randint(0, 12),
                    lu_20d_min=random.randint(0, 6),
                    ma_bullish=random.random() < 0.5,
                    require_lu_day=random.random() < 0.4,
                    amplitude_max=random.choice([0.05, 0.08, 0.12, 0.20]),
                    seal_quality_min=random.choice([0, 0.8, 0.9]),
                    generation=round_num,
                )
                result = evaluate_genome(g, feats_df, daily_df)
                entry = g.to_dict()
                entry['genome_hash'] = g.genome_hash()
                entry.update(result)
                entry['condition_desc'] = g.condition_desc()
                entry['round'] = round_num
                variants.append(entry)
            db['stagnation_counter'] = 0

    variants.sort(key=lambda v: v.get('fitness', 0), reverse=True)
    valid_variants = [v for v in variants if v.get('n_trades', 0) >= 5]
    elites = [StrategyGenome.from_dict(v) for v in valid_variants[:elite_count]]

    if not elites:
        population = seed_genomes()
    else:
        best_v = valid_variants[0]
        logger.info(f"精英: {len(elites)} 个, 最佳: ann={best_v.get('ret_annual',0):.1%} "
                    f"Sharpe={best_v.get('sharpe',0):.2f} trades={best_v.get('n_trades',0)}")
        population = list(elites)
        temp = 1.0 + db.get('stagnation_counter', 0) * 0.3
        for e in elites:
            for _ in range(3):
                population.append(mutate(e, round_num, temp))
        for i in range(min(len(elites), 3)):
            for j in range(i + 1, min(len(elites), 5)):
                population.append(crossover(elites[i], elites[j], round_num))
        for _ in range(max(3, 15 - len(population))):
            population.append(mutate(random.choice(seed_genomes()), round_num, temp=2.0))

    # 去重
    seen = {v.get('genome_hash') for v in variants}
    unique = []
    for g in population:
        h = g.genome_hash()
        if h not in seen:
            seen.add(h)
            unique.append(g)

    logger.info(f"种群: {len(unique)} 个")

    if not unique:
        return db

    new_variants = []
    for gi, g in enumerate(unique):
        result = evaluate_genome(g, feats_df, daily_df)
        entry = g.to_dict()
        entry['genome_hash'] = g.genome_hash()
        entry.update(result)
        entry['condition_desc'] = g.condition_desc()
        entry['round'] = round_num
        new_variants.append(entry)
        if (gi + 1) % 10 == 0:
            logger.info(f"  评估 {gi+1}/{len(unique)} ...")

    all_v = variants + new_variants
    by_hash = {}
    for v in all_v:
        h = v.get('genome_hash', '')
        if h not in by_hash or v.get('fitness', 0) > by_hash[h].get('fitness', 0):
            by_hash[h] = v
    db['variants'] = sorted(by_hash.values(), key=lambda v: v.get('fitness', 0), reverse=True)
    db['rounds'] = round_num

    valid = [v for v in new_variants if v.get('n_trades', 0) >= 5]
    best_ann = max((v.get('ret_annual', 0) for v in db['variants'] if v.get('n_trades', 0) >= 5), default=0)
    db['best_ann'] = best_ann

    logger.info(f"新通过: {len(valid)}, 最佳年化: {best_ann:.1%}")
    if round_num % 3 == 0:
        save_db(db)

    return db


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="策略自进化引擎 v3.0")
    p.add_argument("--target", type=float, default=0.50, help="目标年化收益")
    p.add_argument("--rounds", type=int, default=0, help="轮次上限(0=不到目标不停)")
    p.add_argument("--status", action="store_true")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--top-n", type=int, default=5, help="回测持仓数")
    return p.parse_args()


def main():
    args = parse_args()

    if args.reset:
        for f in [STRATEGY_DB, STRATEGY_DB + '.tmp', RULES_FILE]:
            if os.path.exists(f): os.remove(f)
        logger.info("已重置")

    db = load_db()

    if args.status:
        valid = [v for v in db.get('variants', []) if v.get('n_trades', 0) >= 5]
        valid.sort(key=lambda v: v.get('ret_annual', 0), reverse=True)
        print(f"\n═══ 策略进化状态 ═══")
        print(f"轮次: {db.get('rounds', 0)} | 变体: {len(db.get('variants',[]))} | 有效: {len(valid)}")
        print(f"最佳年化: {db.get('best_ann', 0):.1%}")
        print(f"\nTop-{args.top}:")
        for i, v in enumerate(valid[:args.top], 1):
            print(f"  {i:2d}. ann={v.get('ret_annual',0):+.1%} Sharpe={v.get('sharpe',0):.2f} "
                  f"MDD={v.get('max_dd',0):.1%} WR={v.get('win_rate',0):.1%} trades={v.get('n_trades',0)}")
            print(f"       {v.get('condition_desc', '?')}")
        return

    # ── 加载数据 ──
    cache = _load_cache()
    daily = cache["daily"]
    sig_df = cache["yaogu"]

    logger.info("附加信号特征...")
    feats = attach_features(sig_df, daily)
    logger.info(f"特征: {len(feats)} 行, {len(feats.columns)} 列")

    # 基线
    logger.info("运行基线回测...")
    base_genome = StrategyGenome(top_n=args.top_n)
    baseline = evaluate_genome(base_genome, feats, daily)
    logger.info(f"基线: ann={baseline.get('ret_annual',0):.1%} Sharpe={baseline.get('sharpe',0):.2f} "
                f"trades={baseline.get('n_trades',0)}")

    # ── 进化 ──
    t0 = time.time()
    target = args.target
    max_rounds = args.rounds if args.rounds > 0 else 100000
    round_num = db.get('rounds', 0)
    best_ann = db.get('best_ann', 0)
    baseline_ann = baseline.get('ret_annual', 0)

    console = Console() if HAS_RICH else None
    if HAS_RICH:
        live = Live(console=console, refresh_per_second=2, screen=True)
        live.start()

    try:
        while best_ann < target and round_num < max_rounds:
            round_num += 1
            db = evolve_round(db, feats, daily, round_num)
            best_ann = db.get('best_ann', 0)
            elapsed = time.time() - t0

            if HAS_RICH:
                live.update(_render_panel(db, round_num, best_ann, target, elapsed, baseline_ann))
            elif round_num % 5 == 0:
                logger.info(f"  Round {round_num}: best_ann={best_ann:.1%}, {elapsed:.0f}s")
    finally:
        if HAS_RICH:
            live.stop()

    elapsed = time.time() - t0
    if best_ann >= target:
        logger.success(f"🎯 达到目标年化 {target:.0%}！Round {round_num}, {elapsed:.0f}s")
    else:
        logger.warning(f"达到轮次上限, 最佳年化 {best_ann:.1%}, {elapsed:.0f}s")

    save_db(db)


if __name__ == "__main__":
    main()
