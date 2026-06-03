"""市场状态特征构建器（缓存优化版）。

构建 ~50 维特征向量供 RL 策略网络使用：
  - 市场宽度 (8维): 涨跌比、涨停数、跌停数、资金流等
  - 板块动量 (5维): 5板块20日收益
  - 波动率 (3维): 指数20/60日波动 + 20日收益
  - 恐贪指标 (5维): 简化恐贪分量
  - 因子IC (N维): 各因子近20日RankIC
"""
import numpy as np
import pandas as pd
from config.sector_map import classify_stock

FEATURE_NAMES = [
    "mkt_adv_dec", "mkt_limit_up", "mkt_limit_down", "mkt_vol_ratio",
    "mkt_ret_mean", "mkt_ret_std", "mkt_turnover", "mkt_active_pct",
    "sec_mom_kc", "sec_mom_large", "sec_mom_small", "sec_mom_div", "sec_mom_rest",
    "idx_vol_20", "idx_vol_60", "idx_ret_20",
    "fg_vol", "fg_money", "fg_mom", "fg_nh", "fg_ad",
]


class StateBuilder:
    """从 OHLCV + 指数数据构建 RL 状态向量。支持缓存避免重复IO。"""

    def __init__(self, n_factors: int = 10):
        self.n_factors = n_factors
        self.ic_names = [f"ic_{i}" for i in range(n_factors)]
        self.feature_names = FEATURE_NAMES + self.ic_names

    @property
    def state_dim(self) -> int:
        return len(self.feature_names)

    def build(self, ohlcv, index_df, factor_ic, date):
        return self.build_cached(ohlcv, index_df, factor_ic, date, None, None)

    def build_cached(self, ohlcv, index_df, factor_ic, date, sector_map=None, breadth_cache=None):
        date = pd.Timestamp(date)
        f = []  # features

        # --- 市场宽度 (8维) ---
        day = ohlcv[ohlcv["trade_date"] == date]
        if not day.empty and "close" in day.columns:
            prev = ohlcv[ohlcv["trade_date"] < date].groupby("code")["close"].last()
            cur = day.groupby("code")["close"].last()
            common = prev.index.intersection(cur.index)
            if len(common) > 0:
                rets = (cur[common] - prev[common]) / prev[common]
                adv, dec = int((rets > 0).sum()), int((rets < 0).sum())
                lu = int((rets > 0.099).sum())
                ld = int((rets < -0.099).sum())
                up_amt = day[day["code"].isin(common[rets > 0])]["amount"].sum() if "amount" in day.columns else 0
                tot_amt = day["amount"].sum() if "amount" in day.columns else 1
                f.extend([adv / max(dec, 1), lu, ld, up_amt / max(tot_amt, 1),
                          float(rets.mean()), float(rets.std()) if len(rets) > 1 else 0.0,
                          float(day["turnover"].mean()) if "turnover" in day.columns else 0.0,
                          (adv + dec) / max(len(day), 1)])
            else:
                f.extend([1.0, 0, 0, 0.5, 0.0, 0.0, 0.0, 0.0])
        else:
            f.extend([1.0, 0, 0, 0.5, 0.0, 0.0, 0.0, 0.0])

        # --- 板块动量 (5维, 优先用缓存) ---
        try:
            if breadth_cache is not None:
                breadth = {}
                for cd in sorted(breadth_cache.keys(), reverse=True):
                    if cd <= date:
                        breadth = breadth_cache[cd]
                        break
            else:
                from factors.sector_breadth import compute_breadth_features
                if sector_map is None:
                    codes = ohlcv["code"].unique()
                    sector_map = {c: classify_stock(c, set(), set()) for c in codes}
                breadth = compute_breadth_features(ohlcv, sector_map, date, lookback_days=20)
            for sec in ["科创", "主板大盘", "主板小盘", "红利", "北证"]:
                f.append(breadth.get(sec, {}).get("sector_mom_20", 0.0))
        except Exception:
            f.extend([0.0] * 5)

        # --- 波动率 (3维) ---
        idx = index_df[index_df["trade_date"] <= date].copy()
        if len(idx) > 20 and "close" in idx.columns:
            idx["ret"] = idx["close"].pct_change()
            r = idx["ret"].dropna()
            f.append(float(r.tail(20).std()) if len(r) >= 20 else 0.0)
            f.append(float(r.tail(60).std()) if len(r) >= 60 else (float(r.std()) if len(r) > 0 else 0.0))
            f.append(float(r.tail(20).mean()) if len(r) >= 20 else 0.0)
        else:
            f.extend([0.0, 0.0, 0.0])

        # --- 恐贪 (5维, 简化: 用波动率推算) ---
        try:
            vol20 = f[13] if len(f) > 13 else 0.02
            mom = f[15] if len(f) > 15 else 0.0
            f.extend([
                max(0, min(100, 50 - vol20 * 200)),   # 低波=贪婪
                max(0, min(100, 50 + mom * 500)),      # 正动量=贪婪
                50.0, 50.0, 50.0,
            ])
        except Exception:
            f.extend([50.0] * 5)

        # --- 因子IC (N维) ---
        for i in range(self.n_factors):
            f.append(float(factor_ic.get(i, 0.0)))

        return np.array(f, dtype=np.float32)
