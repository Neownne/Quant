#!/usr/bin/env python
"""策略自进化引擎 —— 自动迭代选股策略，向目标胜率逼近。

核心思路:
  1. 每个"策略变体" = 一组过滤条件 + 阈值
  2. 适应度 = 历史信号上的 10日胜率（ret_10d > 5%）
  3. 每轮: 生成变体 → 验证 → 选择 → 变异 → 下一轮
  4. 结果存入 strategy_db.json，规则写入 strategy_rules.md

用法:
  python scripts/evolve_strategy.py --rounds 5        # 5 轮进化
  python scripts/evolve_strategy.py --status           # 查看进度
  python scripts/evolve_strategy.py --top 10           # 最佳策略
  python scripts/evolve_strategy.py --target 0.80      # 目标胜率(默认0.80)
"""

from __future__ import annotations

import sys, os, json, argparse, time, hashlib, random
from copy import deepcopy
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine

# ── 可视化（可选）──
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── 路径 ──
STRATEGY_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'strategy_db.json')
RULES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'strategy_rules.md')

# ══════════════════════════════════════════════════════════════════════
# 信号数据加载与缓存
# ══════════════════════════════════════════════════════════════════════

_SIGNAL_CACHE = None  # 全局缓存

def load_all_signals(engine=None, force_reload=False):
    """加载妖股 + 涨停全量信号，合并为一个 DataFrame。"""
    global _SIGNAL_CACHE
    if _SIGNAL_CACHE is not None and not force_reload:
        return _SIGNAL_CACHE.copy()

    close_engine = False
    if engine is None:
        engine = get_engine()
        close_engine = True

    yaogu = pd.read_csv('data/signals/bt_signals_yaogu_full.csv')
    limit_up = pd.read_csv('data/signals/bt_signals_limit_up_full.csv')

    yaogu['date'] = pd.to_datetime(yaogu['date'])
    yaogu['code'] = yaogu['code'].astype(str).str.zfill(6)
    yaogu['source'] = 'yaogu'

    limit_up['date'] = pd.to_datetime(limit_up['date'])
    limit_up['code'] = limit_up['code'].astype(str).str.zfill(6)
    limit_up['source'] = 'limit_up'

    # 合并，去重
    all_sigs = pd.concat([yaogu, limit_up], ignore_index=True)
    all_sigs = all_sigs.drop_duplicates(subset=['date', 'code'], keep='first')
    all_sigs = all_sigs[all_sigs['date'] >= '2024-01-01']  # 只看近两年

    # 加载日线数据用于计算因子
    codes = all_sigs['code'].unique().tolist()
    all_dates = sorted(all_sigs['date'].unique())
    min_date = str(pd.Timestamp(min(all_dates)) - pd.Timedelta(days=90))[:10]
    max_date = str(pd.Timestamp(max(all_dates)) + pd.Timedelta(days=30))[:10]

    with engine.connect() as conn:
        daily = pd.read_sql(text("""
            SELECT code, trade_date, open, high, low, close, volume,
                LAG(close) OVER (PARTITION BY code ORDER BY trade_date) as prev_close,
                (close - LAG(close) OVER (PARTITION BY code ORDER BY trade_date))
                / NULLIF(LAG(close) OVER (PARTITION BY code ORDER BY trade_date), 0) as ret
            FROM stock_daily WHERE code = ANY(:codes)
            AND trade_date BETWEEN :s AND :e ORDER BY code, trade_date
        """), conn, params={"codes": codes, "s": min_date, "e": max_date})
        extra = pd.read_sql(text("""
            SELECT code, trade_date, market_cap FROM stock_daily_extra
            WHERE code = ANY(:codes) AND trade_date BETWEEN :s AND :e
        """), conn, params={"codes": codes, "s": min_date, "e": max_date})

    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily['code'] = daily['code'].astype(str).str.zfill(6)
    extra['trade_date'] = pd.to_datetime(extra['trade_date'])
    extra['code'] = extra['code'].astype(str).str.zfill(6)

    if close_engine:
        engine.dispose()

    _SIGNAL_CACHE = {
        'signals': all_sigs,
        'daily': daily,
        'extra': extra,
    }
    logger.info(f"信号缓存: {len(all_sigs)} 条, {len(codes)} 只, 日线 {len(daily)} 行")
    return deepcopy(_SIGNAL_CACHE)


# ══════════════════════════════════════════════════════════════════════
# 策略基因组
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StrategyGenome:
    """一个策略变体的完整基因组 —— 18 个可进化特征。"""
    # ── 核心评分 ──
    yaogu_score_min: int = 3        # 妖股评分下限 (0-9)
    lu_20d_min: int = 1             # 前20日涨停次数下限 (0-20)
    lu_20d_max: int = 999           # 前20日涨停次数上限 (1-999)
    lu_60d_max: int = 999           # 前60日涨停次数上限 (1-999)

    # ── 缩量结构 ──
    low_vol_streak_min: int = 1     # 涨停前缩量天数下限 (0-30)
    vol_ratio_max: float = 99.0     # 涨停日量比上限 (vol/ma20)

    # ── 价格结构 ──
    amplitude_max: float = 0.20     # 涨停日振幅上限 (high-low)/prev_close
    seal_quality_min: float = 0.0   # 封板质量下限 close/high (0-1)
    gap_up_min: float = -0.10       # 开盘跳空下限 (open-prev_close)/prev_close

    # ── MA 结构 ──
    ma_bullish: bool = False        # MA5 > MA10 > MA20
    ma_converge: bool = False       # |MA5/MA10 - 1| < 2%
    ma_deviation_max: float = 0.50  # close偏离MA20上限 (close/ma20 - 1)

    # ── 市值 ──
    mcap_min: float = 10.0          # 市值下限（亿）
    mcap_max: float = float('inf')  # 市值上限（亿）

    # ── 信号日状态 ──
    require_lu_day: bool = False    # 必须当日涨停
    sig_ret_min: float = -0.10      # 信号日涨幅下限
    sig_ret_max: float = 0.15       # 信号日涨幅上限

    # ── 连板质量 ──
    lu_streak_min: int = 1          # 连板天数下限

    # ── 元信息 ──
    generation: int = 0
    parent_hash: str = ""

    def genome_hash(self) -> str:
        """基因组哈希，用于去重。"""
        s = json.dumps({
            k: v for k, v in self.__dict__.items()
            if k not in ('generation', 'parent_hash', 'genome_hash')
        }, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:10]

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        # float('inf') → "inf" for JSON
        for k in d:
            if d[k] == float('inf'):
                d[k] = "inf"
            elif d[k] == float('-inf'):
                d[k] = "-inf"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'StrategyGenome':
        d2 = d.copy()
        for k in d2:
            if d2[k] == "inf":
                d2[k] = float('inf')
            elif d2[k] == "-inf":
                d2[k] = float('-inf')
        g = cls()
        for k, v in d2.items():
            if hasattr(g, k) and not callable(getattr(g, k)):
                setattr(g, k, v)
        return g

    def condition_desc(self) -> str:
        """人类可读的条件描述。"""
        parts = []
        if self.yaogu_score_min > 0:
            parts.append(f"妖股≥{self.yaogu_score_min}")
        if self.low_vol_streak_min > 0:
            parts.append(f"缩量≥{self.low_vol_streak_min}d")
        if self.vol_ratio_max < 99:
            parts.append(f"量比≤{self.vol_ratio_max:.1f}")
        if self.lu_20d_min > 0:
            parts.append(f"涨停≥{self.lu_20d_min}")
        if self.lu_20d_max < 999:
            parts.append(f"涨停≤{self.lu_20d_max}")
        if self.lu_60d_max < 999:
            parts.append(f"60日≤{self.lu_60d_max}")
        if self.lu_streak_min > 1:
            parts.append(f"连板≥{self.lu_streak_min}")
        if self.amplitude_max < 0.20:
            parts.append(f"振幅≤{self.amplitude_max:.0%}")
        if self.seal_quality_min > 0:
            parts.append(f"封板≥{self.seal_quality_min:.0%}")
        if self.gap_up_min > -0.10:
            parts.append(f"跳空≥{self.gap_up_min:.0%}")
        if self.ma_bullish:
            parts.append("MA多头")
        if self.ma_converge:
            parts.append("MA收敛")
        if self.ma_deviation_max < 0.50:
            parts.append(f"偏离≤{self.ma_deviation_max:.0%}")
        if self.mcap_min > 0:
            parts.append(f"市值≥{self.mcap_min:.0f}亿")
        if self.mcap_max < float('inf'):
            parts.append(f"市值≤{self.mcap_max:.0f}亿")
        if self.require_lu_day:
            parts.append("当日涨停")
        if self.sig_ret_min > -0.10:
            parts.append(f"涨幅≥{self.sig_ret_min:.0%}")
        if self.sig_ret_max < 0.15:
            parts.append(f"涨幅≤{self.sig_ret_max:.0%}")
        return " + ".join(parts) if parts else "全通过"


# ══════════════════════════════════════════════════════════════════════
# 适应度评估
# ══════════════════════════════════════════════════════════════════════

def compute_features_for_signals(signals_df, daily_df, extra_df=None):
    """为每个信号行预计算所有可能用到的特征。"""
    daily_df = daily_df.sort_values(['code', 'trade_date']).copy()

    # 合并市值数据
    if extra_df is not None and not extra_df.empty:
        daily_df = daily_df.merge(
            extra_df[['code', 'trade_date', 'market_cap']],
            on=['code', 'trade_date'], how='left')
        daily_df['mcap'] = daily_df['market_cap'].fillna(0)
    else:
        daily_df['mcap'] = 0.0

    # 分组预计算
    daily_df['ret'] = daily_df.groupby('code')['close'].pct_change()
    daily_df['ma5'] = daily_df.groupby('code')['close'].transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    daily_df['ma10'] = daily_df.groupby('code')['close'].transform(
        lambda x: x.rolling(10, min_periods=10).mean())
    daily_df['ma20'] = daily_df.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=20).mean())
    daily_df['vol_ma20'] = daily_df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean())

    # is_lu（9.5% 涨幅 = 涨停）
    daily_df['is_lu'] = (daily_df['ret'] >= 0.095).astype(int)

    # lu_20d
    daily_df['lu_20d'] = daily_df.groupby('code')['is_lu'].transform(
        lambda x: x.rolling(20, min_periods=10).sum())
    # lu_60d
    daily_df['lu_60d'] = daily_df.groupby('code')['is_lu'].transform(
        lambda x: x.rolling(60, min_periods=30).sum())

    features = []
    daily_by_code_date = daily_df.set_index(['code', 'trade_date'])
    total_sigs = len(signals_df)

    for si, (_, sig) in enumerate(signals_df.iterrows()):
        if si > 0 and si % 3000 == 0:
            logger.info(f"  特征进度: {si}/{total_sigs} ({si/total_sigs*100:.0f}%)")
        code = sig['code']
        sig_date = sig['date']
        score = sig.get('score', 0)

        # 信号日的数据
        try:
            today = daily_by_code_date.loc[(code, sig_date)]
        except KeyError:
            continue

        # 信号日前 N 日数据
        pre = daily_df[(daily_df['code'] == code) & (daily_df['trade_date'] < sig_date)].tail(60)
        if len(pre) < 5:
            continue

        # 缩量整理（不包含信号日）
        low_vol_streak = 0
        vol_ma = pre['volume'].tail(20).mean() if len(pre) >= 20 else pre['volume'].mean()
        if vol_ma > 0:
            for _, pr in pre.iterrows():
                if pr['volume'] < vol_ma * 0.7:
                    low_vol_streak += 1
                else:
                    low_vol_streak = 0

        # 前20日涨停次数
        lu_20d = int(today.get('lu_20d', 0)) if pd.notna(today.get('lu_20d')) else 0
        lu_60d = int(today.get('lu_60d', 0)) if pd.notna(today.get('lu_60d')) else 0

        # MA 条件
        ma5 = today.get('ma5', np.nan)
        ma10 = today.get('ma10', np.nan)
        ma20 = today.get('ma20', np.nan)
        ma_bullish = (pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20)
                      and ma5 > ma10 > ma20)
        ma_converge = (pd.notna(ma5) and pd.notna(ma10)
                       and abs(ma5 / ma10 - 1) < 0.02)

        # 信号日涨幅
        sig_ret = float(today.get('ret', 0)) if pd.notna(today.get('ret')) else 0
        is_lu_day = sig_ret >= 0.095

        # 新特征
        prev_c = today.get('prev_close', np.nan)
        if pd.isna(prev_c) or prev_c <= 0:
            prev_c = today['close'] if pd.notna(today['close']) and today['close'] > 0 else 1.0
        amplitude = float((today['high'] - today['low']) / prev_c
                         ) if pd.notna(today.get('high')) and pd.notna(today.get('low')) else 0.50
        seal_quality = float(today['close'] / today['high']) if (pd.notna(today.get('high'))
                         and today['high'] > 0) else 0.0
        gap_up = float(today['open'] / pre.iloc[-1]['close'] - 1) if (len(pre) > 0
                     and pd.notna(today.get('open')) and pre.iloc[-1]['close'] > 0) else 0.0
        vol_ratio = float(today['volume'] / vol_ma) if (pd.notna(today.get('volume'))
                     and vol_ma > 0) else 1.0
        ma_deviation = float(today['close'] / ma20 - 1) if (pd.notna(today.get('close'))
                          and pd.notna(ma20) and ma20 > 0) else 0.0

        # 连板天数
        lu_streak = 0
        for _, pr in pre.iloc[::-1].iterrows():
            if pd.notna(pr.get('ret')) and pr['ret'] >= 0.095:
                lu_streak += 1
            else:
                break
        if is_lu_day:
            lu_streak += 1

        # 前向收益：如果信号日是涨停日，实际 T+1 才能买入，用次日收盘为入场价
        fwd = daily_df[(daily_df['code'] == code) & (daily_df['trade_date'] > sig_date)].head(21)
        if len(fwd) < 2:
            continue

        if is_lu_day:
            # 涨停日买不到，T+1 入场
            entry_price = fwd.iloc[0]['close']  # 次日收盘
            fwd = fwd.iloc[1:]  # 前向从 T+2 开始
        else:
            entry_price = today['close']

        if pd.isna(entry_price) or entry_price <= 0:
            continue
        ret_5d = fwd.iloc[min(4, len(fwd)-1)]['close'] / entry_price - 1 if len(fwd) >= 1 else 0
        ret_10d = fwd.iloc[min(9, len(fwd)-1)]['close'] / entry_price - 1 if len(fwd) >= 1 else 0
        ret_20d = fwd.iloc[min(19, len(fwd)-1)]['close'] / entry_price - 1 if len(fwd) >= 1 else 0

        mcap_val = float(today.get('market_cap', 0)) if pd.notna(today.get('market_cap')) else 0.0
        features.append({
            'code': code, 'date': sig_date, 'score': score,
            'low_vol_streak': low_vol_streak,
            'lu_20d': lu_20d, 'lu_60d': lu_60d, 'lu_streak': lu_streak,
            'ma_bullish': ma_bullish, 'ma_converge': ma_converge,
            'ma_deviation': ma_deviation,
            'is_lu_day': is_lu_day, 'sig_ret': sig_ret,
            'amplitude': amplitude, 'seal_quality': seal_quality,
            'gap_up': gap_up, 'vol_ratio': vol_ratio, 'mcap': mcap_val,
            'ret_5d': ret_5d, 'ret_10d': ret_10d, 'ret_20d': ret_20d,
        })

    return pd.DataFrame(features)


def evaluate_genome(genome: StrategyGenome, feats_df: pd.DataFrame) -> dict:
    """在历史信号上评估策略变体的表现。"""
    df = feats_df.copy()

    # 应用基因组过滤
    mask = pd.Series(True, index=df.index)

    if genome.yaogu_score_min > 0:
        mask &= df['score'] >= genome.yaogu_score_min
    if genome.low_vol_streak_min > 0:
        mask &= df['low_vol_streak'] >= genome.low_vol_streak_min
    if genome.vol_ratio_max < 99:
        mask &= df['vol_ratio'] <= genome.vol_ratio_max
    if genome.lu_20d_min > 0:
        mask &= df['lu_20d'] >= genome.lu_20d_min
    if genome.lu_20d_max < 999:
        mask &= df['lu_20d'] <= genome.lu_20d_max
    if genome.lu_60d_max < 999:
        mask &= df['lu_60d'] <= genome.lu_60d_max
    if genome.lu_streak_min > 1:
        mask &= df['lu_streak'] >= genome.lu_streak_min
    if genome.amplitude_max < 0.20:
        mask &= df['amplitude'] <= genome.amplitude_max
    if genome.seal_quality_min > 0:
        mask &= df['seal_quality'] >= genome.seal_quality_min
    if genome.gap_up_min > -0.10:
        mask &= df['gap_up'] >= genome.gap_up_min
    if genome.ma_bullish:
        mask &= df['ma_bullish'] == True
    if genome.ma_converge:
        mask &= df['ma_converge'] == True
    if genome.ma_deviation_max < 0.50:
        mask &= df['ma_deviation'] <= genome.ma_deviation_max
    if genome.mcap_min > 0:
        mask &= df['mcap'] >= genome.mcap_min
    if genome.mcap_max < float('inf'):
        mask &= df['mcap'] <= genome.mcap_max
    if genome.require_lu_day:
        mask &= df['is_lu_day'] == True
    if genome.sig_ret_min > -0.10:
        mask &= df['sig_ret'] >= genome.sig_ret_min
    if genome.sig_ret_max < 0.15:
        mask &= df['sig_ret'] <= genome.sig_ret_max

    sub = df[mask]
    n = len(sub)

    if n < 30:
        return {'n': n, 'win_rate_10d': 0, 'avg_ret_10d': 0,
                'win_rate_20d': 0, 'avg_ret_20d': 0, 'fitness': -1}

    wr_10d = sub['ret_10d'].gt(0.05).mean()
    avg_10d = sub['ret_10d'].mean()
    wr_20d = sub['ret_20d'].gt(0.10).mean()
    avg_20d = sub['ret_20d'].mean()

    # 适应度 = 胜率为主 + 收益为辅 + 样本惩罚
    sample_bonus = min(n / 200, 1.0) * 0.05  # 样本≥200得满分
    fitness = wr_10d * 0.7 + wr_20d * 0.2 + min(avg_10d / 0.3, 0.1) + sample_bonus

    return {
        'n': n,
        'win_rate_10d': round(float(wr_10d), 4),
        'avg_ret_10d': round(float(avg_10d), 4),
        'win_rate_20d': round(float(wr_20d), 4),
        'avg_ret_20d': round(float(avg_20d), 4),
        'fitness': round(float(fitness), 4),
    }


# ══════════════════════════════════════════════════════════════════════
# 变异与交叉
# ══════════════════════════════════════════════════════════════════════

def mutate(genome: StrategyGenome, generation: int, temp: float = 1.0) -> StrategyGenome:
    """随机变异。temp 高温→大步变异，低温→微调。"""
    g = deepcopy(genome)
    g.generation = generation
    g.parent_hash = genome.genome_hash()

    # 整数阈值：概率变异 ±1~3 步
    int_fields = {
        'yaogu_score_min': (0, 9, 1),
        'low_vol_streak_min': (0, 20, 2),
        'lu_20d_min': (0, 10, 1),
        'lu_streak_min': (1, 8, 1),
    }
    for field, (lo, hi, step) in int_fields.items():
        if random.random() < 0.4 * temp:
            delta = random.choice([-step, step, -step*2, step*2])
            setattr(g, field, max(lo, min(hi, getattr(g, field) + delta)))

    # 整数上限
    upper_fields = {'lu_20d_max': (1, 999), 'lu_60d_max': (1, 999)}
    for field, (lo, hi) in upper_fields.items():
        if random.random() < 0.3 * temp:
            delta = random.choice([-1, 1, -2, 2])
            setattr(g, field, max(lo, min(hi, getattr(g, field) + delta)))

    # 浮点阈值
    float_fields = {
        'vol_ratio_max': (0.5, 99, 0.5),
        'amplitude_max': (0.02, 0.20, 0.02),
        'seal_quality_min': (0.0, 1.0, 0.05),
        'gap_up_min': (-0.10, 0.10, 0.02),
        'ma_deviation_max': (0.02, 0.50, 0.05),
        'mcap_min': (0, 500, 10),
        'sig_ret_min': (-0.10, 0.10, 0.02),
        'sig_ret_max': (0.02, 0.15, 0.02),
    }
    for field, (lo, hi, step) in float_fields.items():
        if random.random() < 0.4 * temp:
            delta = random.choice([-step, step, -step*2, step*2, 0])
            new_val = getattr(g, field) + delta
            setattr(g, field, max(lo, min(hi, round(new_val, 3))))

    # 布尔翻转
    bool_fields = ['ma_bullish', 'ma_converge', 'require_lu_day']
    for field in bool_fields:
        if random.random() < 0.15 * temp:
            setattr(g, field, not getattr(g, field))

    # 高通量变异：随机大幅跳变
    if random.random() < 0.1 * temp:
        field = random.choice(list(int_fields.keys()) + list(float_fields.keys()))
        lo = int_fields.get(field, (0, 1, 1))[0] if field in int_fields else float_fields[field][0]
        hi = int_fields.get(field, (0, 1, 1))[1] if field in int_fields else float_fields[field][1]
        if isinstance(getattr(g, field), bool):
            setattr(g, field, not getattr(g, field))
        elif isinstance(getattr(g, field), int):
            setattr(g, field, random.randint(int(lo), int(hi)))
        else:
            setattr(g, field, round(random.uniform(lo, hi), 3))

    return g


def crossover(g1: StrategyGenome, g2: StrategyGenome, generation: int) -> StrategyGenome:
    """两个基因组交叉。每个基因随机从父母之一继承。"""
    child = StrategyGenome()
    child.generation = generation
    child.parent_hash = g1.genome_hash()[:5] + "_" + g2.genome_hash()[:5]

    fields = ['yaogu_score_min', 'low_vol_streak_min', 'vol_ratio_max',
              'lu_20d_min', 'lu_20d_max', 'lu_60d_max', 'lu_streak_min',
              'amplitude_max', 'seal_quality_min', 'gap_up_min',
              'ma_bullish', 'ma_converge', 'ma_deviation_max',
              'mcap_min', 'mcap_max', 'require_lu_day',
              'sig_ret_min', 'sig_ret_max']

    for f in fields:
        setattr(child, f, getattr(random.choice([g1, g2]), f))

    return child


# ══════════════════════════════════════════════════════════════════════
# DB 管理
# ══════════════════════════════════════════════════════════════════════

def load_db():
    if os.path.exists(STRATEGY_DB):
        with open(STRATEGY_DB) as f:
            return json.load(f)
    return {"variants": [], "rounds": 0, "best_fitness": 0, "target": 0.80}


def save_db(db):
    os.makedirs(os.path.dirname(STRATEGY_DB), exist_ok=True)
    with open(STRATEGY_DB, 'w') as f:
        json.dump(db, f, ensure_ascii=False, indent=2, default=str)


def save_rules(db):
    """从 DB 中提取规则。"""
    variants = db.get("variants", [])
    if not variants:
        return

    passed = [v for v in variants if v.get('n', 0) >= 50
              and v.get('win_rate_10d', 0) >= 0.35]
    passed.sort(key=lambda v: v.get('win_rate_10d', 0), reverse=True)

    lines = [
        "# 策略进化规则",
        f"# 进化轮次: {db.get('rounds', 0)}",
        f"# 目标胜率: {db.get('target', 0.80):.0%}",
        f"# 当前最佳: {db.get('best_fitness', 0):.4f}",
        "",
        "## 已验证有效的过滤条件",
        "",
    ]

    # 统计每个条件的边际贡献
    all_conds = {}
    for v in passed:
        conds = v.get('condition_desc', '')
        wr = v.get('win_rate_10d', 0)
        n = v.get('n', 0)
        if wr > 0:
            all_conds[conds] = (wr, n)

    for conds, (wr, n) in sorted(all_conds.items(), key=lambda x: x[1][0], reverse=True)[:20]:
        lines.append(f"- **{wr:.1%}** (n={n}) — `{conds}`")

    lines.append("")
    lines.append("## 进化历史")
    for v in passed[:10]:
        h = v.get('genome_hash', '?')[:8]
        g = v.get('generation', '?')
        wr = v.get('win_rate_10d', 0)
        n = v.get('n', 0)
        lines.append(f"- Gen{g} {h}: {wr:.1%} (n={n}) — `{v.get('condition_desc', '?')}`")

    with open(RULES_FILE, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"规则已保存: {RULES_FILE}")


# ══════════════════════════════════════════════════════════════════════
# 种子策略
# ══════════════════════════════════════════════════════════════════════

def seed_genomes() -> list[StrategyGenome]:
    """生成初始种子种群——覆盖多种已知有效特征组合。"""
    seeds = []

    # 基线
    seeds.append(StrategyGenome(yaogu_score_min=3, generation=0))
    seeds.append(StrategyGenome(lu_20d_min=2, ma_bullish=True, generation=0))

    # 妖股+缩量系列
    for score in [3, 5, 6]:
        for streak in [1, 3, 6]:
            seeds.append(StrategyGenome(yaogu_score_min=score, low_vol_streak_min=streak, generation=0))

    # MA条件系列
    for ma_bull in [True, False]:
        for ma_conv in [True, False]:
            seeds.append(StrategyGenome(yaogu_score_min=5, low_vol_streak_min=3,
                                        ma_bullish=ma_bull, ma_converge=ma_conv, generation=0))

    # 涨停日限定系列
    for score in [5, 6]:
        for streak in [3, 6]:
            seeds.append(StrategyGenome(yaogu_score_min=score, low_vol_streak_min=streak,
                                        ma_bullish=True, require_lu_day=True, generation=0))

    # 新特征系列：封板+振幅+量比
    for seal in [0.8, 0.9, 0.95]:
        seeds.append(StrategyGenome(yaogu_score_min=5, seal_quality_min=seal, generation=0))

    for amp in [0.05, 0.08, 0.12]:
        seeds.append(StrategyGenome(yaogu_score_min=5, amplitude_max=amp, low_vol_streak_min=3, generation=0))

    for vr in [2.0, 3.0, 5.0]:
        seeds.append(StrategyGenome(yaogu_score_min=5, vol_ratio_max=vr, low_vol_streak_min=3, generation=0))

    # 连板限定
    for streak_lu in [2, 3, 4]:
        seeds.append(StrategyGenome(yaogu_score_min=5, lu_streak_min=streak_lu, low_vol_streak_min=3, generation=0))

    # 随机注入多样性
    for _ in range(20):
        g = StrategyGenome(
            yaogu_score_min=random.randint(0, 7),
            low_vol_streak_min=random.randint(0, 10),
            lu_20d_min=random.randint(0, 5),
            lu_20d_max=random.choice([2, 4, 6, 999]),
            ma_bullish=random.random() < 0.5,
            ma_converge=random.random() < 0.5,
            require_lu_day=random.random() < 0.3,
            seal_quality_min=random.choice([0, 0.8, 0.9]),
            amplitude_max=random.choice([0.05, 0.08, 0.12, 0.20]),
            vol_ratio_max=random.choice([2.0, 3.0, 5.0, 99.0]),
            generation=0,
        )
        seeds.append(g)

    return seeds


# ══════════════════════════════════════════════════════════════════════
# 进化主循环
# ══════════════════════════════════════════════════════════════════════

def evolve_round(db: dict, feats_df: pd.DataFrame, round_num: int,
                  elite_count: int = 5, stagnation_limit: int = 15):
    """执行一轮进化。检测收敛时注入随机多样性，高温重启。"""
    logger.info(f"═══ Round {round_num} ═══")
    variants = db.get("variants", [])
    valid_variants = [v for v in variants if v.get('n', 0) >= 30]

    # 收敛检测
    if len(valid_variants) >= 10:
        prev_best = db.get('prev_best_fitness', 0)
        best_now = valid_variants[0].get('fitness', 0) if valid_variants else 0

        db['stagnation_counter'] = db.get('stagnation_counter', 0)
        if abs(best_now - prev_best) < 0.001:
            db['stagnation_counter'] += 1
        else:
            db['stagnation_counter'] = 0
        db['prev_best_fitness'] = best_now

        if db['stagnation_counter'] >= stagnation_limit:
            logger.warning(f"收敛停滞 {stagnation_limit} 轮，高温重启！注入 30 个随机变体")
            for _ in range(30):
                g = StrategyGenome(
                    yaogu_score_min=random.randint(0, 8),
                    low_vol_streak_min=random.randint(0, 15),
                    lu_20d_min=random.randint(0, 6),
                    lu_20d_max=random.choice([2, 4, 6, 8, 999]),
                    lu_60d_max=random.choice([5, 10, 20, 999]),
                    lu_streak_min=random.randint(1, 6),
                    amplitude_max=random.choice([0.03, 0.05, 0.08, 0.12, 0.20]),
                    seal_quality_min=random.choice([0, 0.7, 0.8, 0.9, 0.95]),
                    gap_up_min=random.choice([-0.05, 0, 0.02, 0.05]),
                    vol_ratio_max=random.choice([1.5, 2.0, 3.0, 5.0, 99.0]),
                    ma_bullish=random.random() < 0.5,
                    ma_converge=random.random() < 0.5,
                    require_lu_day=random.random() < 0.4,
                    ma_deviation_max=random.choice([0.1, 0.2, 0.3, 0.50]),
                    generation=round_num,
                )
                entry = g.to_dict()
                entry['genome_hash'] = g.genome_hash()
                result = evaluate_genome(g, feats_df)
                entry.update(result)
                entry['condition_desc'] = g.condition_desc()
                entry['round'] = round_num
                entry['date'] = str(date.today())
                variants.append(entry)
            db['stagnation_counter'] = 0

    # 精英选择
    variants.sort(key=lambda v: v.get('fitness', 0), reverse=True)
    valid_variants = [v for v in variants if v.get('n', 0) >= 30]
    elites = [StrategyGenome.from_dict(v) for v in valid_variants[:elite_count]]

    if not elites:
        logger.warning("无有效精英，重新播种")
        population = seed_genomes()
        round_num = 0
    else:
        population = list(elites)
        best_wr = valid_variants[0].get('win_rate_10d', 0)
        logger.info(f"精英: {len(elites)} 个, 最佳胜率: {best_wr:.1%}, "
                    f"适应度: {valid_variants[0].get('fitness', 0):.4f}")

        # 变异：每个精英产生变异后代，温度随停滞递增
        temp = 1.0 + db.get('stagnation_counter', 0) * 0.3
        for elite in elites:
            for _ in range(4):
                population.append(mutate(elite, round_num, temp=temp))

        # 交叉
        for i in range(min(len(elites), 3)):
            for j in range(i + 1, min(len(elites), 5)):
                population.append(crossover(elites[i], elites[j], round_num))

        # 随机注入：与搜索空间大小成比例
        for _ in range(max(5, 20 - len(population))):
            population.append(mutate(random.choice(seed_genomes()), round_num, temp=2.0))

    # 去重
    seen_hashes = {v.get('genome_hash') for v in variants}
    unique_pop = []
    for g in population:
        h = g.genome_hash()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_pop.append(g)

    logger.info(f"种群: {len(unique_pop)} 个")

    if not unique_pop:
        return db

    # 评估新变体
    new_variants = []
    for gi, g in enumerate(unique_pop):
        result = evaluate_genome(g, feats_df)
        entry = g.to_dict()
        entry['genome_hash'] = g.genome_hash()
        entry.update(result)
        entry['condition_desc'] = g.condition_desc()
        entry['round'] = round_num
        entry['date'] = str(date.today())
        new_variants.append(entry)
        if (gi + 1) % 20 == 0:
            logger.info(f"  评估 {gi+1}/{len(unique_pop)} ...")

    # 合并
    all_variants = variants + new_variants
    by_hash = {}
    for v in all_variants:
        h = v.get('genome_hash', '')
        if h not in by_hash or v.get('fitness', 0) > by_hash[h].get('fitness', 0):
            by_hash[h] = v
    all_variants = sorted(by_hash.values(), key=lambda v: v.get('fitness', 0), reverse=True)

    db['variants'] = all_variants
    db['rounds'] = round_num

    passed = [v for v in new_variants if v.get('n', 0) >= 30]
    best_wr = max((v.get('win_rate_10d', 0) for v in all_variants if v.get('n', 0) >= 30), default=0)
    db['best_win_rate'] = best_wr
    db['best_fitness'] = all_variants[0].get('fitness', 0) if all_variants else 0

    top5 = [v for v in all_variants[:5] if v.get('n', 0) >= 30]
    avg_wr = np.mean([v.get('win_rate_10d', 0) for v in top5]) if top5 else 0
    db['top5_avg_wr'] = avg_wr

    logger.info(f"新通过: {len(passed)}, 最佳胜率: {best_wr:.1%}, 停滞: {db.get('stagnation_counter', 0)}轮")

    if round_num % 5 == 0 or len(passed) > 0:
        save_rules(db)
        save_db(db)

    return db


# ══════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════

def _render_panel(db, round_num, best_wr, target_wr, elapsed, baseline_wr):
    """返回一个 Rich Panel（用于 Live.update）。"""
    valid = [v for v in db.get('variants', []) if v.get('n', 0) >= 30]
    valid = sorted(valid, key=lambda v: v.get('win_rate_10d', 0), reverse=True)[:10]
    stag = db.get('stagnation_counter', 0)
    total = len(db.get('variants', []))

    progress_pct = min(best_wr / target_wr, 1.0) if target_wr > 0 else 0
    bar_width = 30
    filled = int(bar_width * progress_pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    table = Table(title=f"🧬 策略进化 — Round {round_num}", title_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("10日胜率", justify="right", style="green")
    table.add_column("20日胜率", justify="right", style="yellow")
    table.add_column("均收益", justify="right")
    table.add_column("n", justify="right", style="dim", width=4)
    table.add_column("条件", style="bright_blue", max_width=55)

    for i, v in enumerate(valid[:8], 1):
        table.add_row(
            str(i),
            f"{v.get('win_rate_10d',0):.1%}",
            f"{v.get('win_rate_20d',0):.1%}",
            f"{v.get('avg_ret_10d',0):+.1%}",
            str(v.get('n',0)),
            v.get('condition_desc','?')[:52],
        )

    return Panel(
        table,
        title=f"[bold]目标胜率: {target_wr:.0%}  |  基线: {baseline_wr:.1%}  |  "
              f"最佳: {best_wr:.1%}  |  停滞: {stag}轮  |  变体: {total}  |  {elapsed:.0f}s[/]",
        subtitle=f"[{bar}] {best_wr:.1%}",
        border_style="green" if best_wr >= target_wr else "blue",
    )


def _render_status(console, db, round_num, best_wr, target_wr, elapsed, baseline_wr, final=False):
    """最终输出。"""
    if console is None:
        return
    valid = [v for v in db.get('variants', []) if v.get('n', 0) >= 30]
    console.print(_render_panel(db, round_num, best_wr, target_wr, elapsed, baseline_wr))
    if final:
        console.print(f"\n[bold green]✅ 进化完成！总变体: {len(db.get('variants',[]))}, "
                      f"有效策略: {len(valid)}[/]")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="策略自进化引擎")
    p.add_argument("--rounds", type=int, default=5, help="进化轮次")
    p.add_argument("--status", action="store_true", help="查看进化状态")
    p.add_argument("--top", type=int, default=10, help="显示最佳策略数")
    p.add_argument("--target", type=float, default=0.80, help="目标胜率")
    p.add_argument("--reset", action="store_true", help="重置进化DB")
    return p.parse_args()


def main():
    args = parse_args()

    if args.reset:
        if os.path.exists(STRATEGY_DB):
            os.remove(STRATEGY_DB)
        if os.path.exists(RULES_FILE):
            os.remove(RULES_FILE)
        logger.info("已重置进化 DB")

    db = load_db()
    db['target'] = args.target

    if args.status:
        variants = db.get('variants', [])
        valid = [v for v in variants if v.get('n', 0) >= 30]
        valid.sort(key=lambda v: v.get('win_rate_10d', 0), reverse=True)

        print(f"\n═══ 策略进化状态 ═══")
        print(f"轮次: {db.get('rounds', 0)}")
        print(f"目标胜率: {args.target:.0%}")
        print(f"变体总数: {len(variants)} (有效: {len(valid)})")
        print(f"最佳适应度: {db.get('best_fitness', 0):.4f}")
        print(f"\nTop-{args.top} 策略:")
        for i, v in enumerate(valid[:args.top], 1):
            print(f"  {i:2d}. Gen{v.get('generation','?')} wr_10d={v.get('win_rate_10d',0):.1%} "
                  f"wr_20d={v.get('win_rate_20d',0):.1%} avg10d={v.get('avg_ret_10d',0):+.1%} "
                  f"n={v.get('n',0):4d}")
            print(f"       {v.get('condition_desc', '?')}")
        return

    # ── 加载信号特征 ──
    logger.info("加载信号数据...")
    cache = load_all_signals()
    signals = cache['signals']
    daily = cache['daily']

    logger.info("预计算特征...")
    feats = compute_features_for_signals(signals, daily, cache.get('extra'))
    logger.info(f"特征: {len(feats)} 行, {len(feats.columns)} 列")
    logger.info(f"基线胜率: {feats['ret_10d'].gt(0.05).mean():.1%} (n={len(feats)})")

    # ── 进化 ──
    t0 = time.time()
    target_wr = args.target
    baseline_wr = feats['ret_10d'].gt(0.05).mean()
    logger.info(f"目标胜率: {target_wr:.0%}, 基线: {baseline_wr:.1%}")

    round_num = db.get('rounds', 0)
    best_wr = db.get('best_win_rate', 0)
    max_rounds = 10000

    # 可视化
    console = Console() if HAS_RICH else None

    if HAS_RICH:
        live = Live(console=console, refresh_per_second=2, screen=True)
        live.start()

    try:
        while best_wr < target_wr and round_num < max_rounds:
            round_num += 1
            db = evolve_round(db, feats, round_num)
            best_wr = db.get('best_win_rate', 0)
            elapsed = time.time() - t0

            if HAS_RICH:
                live.update(_render_panel(db, round_num, best_wr, target_wr, elapsed, baseline_wr))
            elif round_num % 10 == 0:
                logger.info(f"  ⏱ Round {round_num}: best_wr={best_wr:.1%}, "
                            f"变体={len(db.get('variants',[]))}, {elapsed:.0f}s")
    finally:
        if HAS_RICH:
            live.stop()
            _render_status(console, db, round_num, best_wr, target_wr, elapsed, baseline_wr, final=True)

    elapsed = time.time() - t0
    if best_wr >= target_wr:
        logger.success(f"🎯 达到目标 {target_wr:.0%}！Round {round_num}, {elapsed:.0f}s")
    else:
        logger.warning(f"达到轮次上限 {max_rounds}, 最佳 {best_wr:.1%}, {elapsed:.0f}s")

    if HAS_RICH:
        _render_status(console, db, round_num, best_wr, target_wr, elapsed, baseline_wr, final=True)

    save_rules(db)
    save_db(db)


if __name__ == "__main__":
    main()
