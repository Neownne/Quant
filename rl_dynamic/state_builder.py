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
    # 概念板块特征 (8维) — 替换原恐贪占位符
    "cb_hot1_5d", "cb_hot2_5d", "cb_hot3_5d", "cb_hot4_5d", "cb_hot5_5d",
    "cb_breadth", "cb_vol", "cb_rotation",
]


class StateBuilder:
    """从 OHLCV + 指数数据构建 RL 状态向量。支持缓存避免重复IO。"""

    def __init__(self, n_factors: int = 10, concept_features=None):
        self.n_factors = n_factors
        self.concept_features = concept_features  # ConceptBoardFeatures 或 None
        self.ic_names = [f"ic_{i}" for i in range(n_factors)]
        self.feature_names = FEATURE_NAMES + self.ic_names

    @property
    def state_dim(self) -> int:
        return len(self.feature_names)

    def build(self, ohlcv, index_df, factor_ic, date):
        return self.build_cached(ohlcv, index_df, factor_ic, date, None, None)

    def build_all(self, ohlcv, index_df, ic_maps: dict, cache_5d_rets: bool = True
                  ) -> tuple[dict, dict | None]:
        """预计算所有日期的状态 + 5日收益（向量化，比逐个 build() 快 10x+）。

        Returns:
            states: {pd.Timestamp: np.array}
            rets_5d: {pd.Timestamp: pd.Series} or None
        """
        import numpy as np

        dates = sorted(ohlcv["trade_date"].unique())
        ohlcv = ohlcv.sort_values(["code", "trade_date"]).copy()

        # ── 预计算 per-date aggregates ──
        close_pivot = ohlcv.pivot_table(index="trade_date", columns="code",
                                         values="close", aggfunc="last").sort_index()
        amount_pivot = ohlcv.pivot_table(index="trade_date", columns="code",
                                          values="amount", aggfunc="sum").sort_index()
        turnover_pivot = ohlcv.pivot_table(index="trade_date", columns="code",
                                            values="turnover", aggfunc="last").sort_index()

        # Daily returns (close-based)
        ret_pivot = close_pivot.pct_change()

        # ── 预计算概念板块特征 ──
        concept_cache = None
        if self.concept_features is not None:
            concept_cache = self.concept_features._cache

        # ── 预计算波动率 ──
        idx = index_df.sort_values("trade_date").copy()
        idx["idx_ret"] = idx["close"].pct_change()
        idx_vol_20 = idx["idx_ret"].rolling(20).std()
        idx_vol_60 = idx["idx_ret"].rolling(60).std()
        idx_ret_20 = idx["idx_ret"].rolling(20).mean()

        # ── 预计算 5 日收益（可选）──
        rets_5d = {}
        if cache_5d_rets:
            close_shifted = close_pivot.shift(5)
            ret_5d_pivot = (close_pivot - close_shifted) / close_shifted
            for dt in dates:
                if dt in ret_5d_pivot.index:
                    s = ret_5d_pivot.loc[dt].dropna()
                    if len(s) > 0:
                        rets_5d[pd.Timestamp(dt.date())] = s.values.astype(float)

        # ── 预计算板块动量 ──
        # 用 classify_stock 做 sector_map
        from config.sector_map import classify_stock
        codes_all = ohlcv["code"].unique()
        sector_map = {c: classify_stock(c, set(), set()) for c in codes_all}

        # 预计算板块每日等权收益
        sec_to_codes = {}
        for c, sec in sector_map.items():
            sec_to_codes.setdefault(sec, []).append(c)
        sec_daily_ret = {}
        for sec, codes in sec_to_codes.items():
            cols = [c for c in codes if c in ret_pivot.columns]
            if cols:
                sec_daily_ret[sec] = ret_pivot[cols].mean(axis=1)

        # ── 批量构建状态 ──
        states = {}
        nf = self.n_factors

        for di, dt in enumerate(dates):
            f = []
            # Date key
            idx_row = idx[idx["trade_date"] == dt]

            # -- 市场宽度 (8d) --
            if dt in ret_pivot.index:
                day_rets = ret_pivot.loc[dt].dropna()
                if len(day_rets) > 1:
                    adv, dec = int((day_rets > 0).sum()), int((day_rets < 0).sum())
                    lu = int((day_rets > 0.099).sum())
                    ld = int((day_rets < -0.099).sum())

                    # 成交额比
                    if dt in amount_pivot.index:
                        day_amt = amount_pivot.loc[dt].dropna()
                        up_codes = set(day_rets[day_rets > 0].index)
                        up_amt = day_amt[day_amt.index.isin(up_codes)].sum()
                        tot_amt = day_amt.sum()
                        amt_ratio = up_amt / max(tot_amt, 1)
                    else:
                        amt_ratio = 0.5

                    f.extend([
                        adv / max(dec, 1), lu, ld, amt_ratio,
                        float(day_rets.mean()), float(day_rets.std()),
                        float(turnover_pivot.loc[dt].mean()) if dt in turnover_pivot.index else 0.0,
                        (adv + dec) / max(len(day_rets), 1),
                    ])
                else:
                    f.extend([1.0, 0, 0, 0.5, 0.0, 0.0, 0.0, 0.0])
            else:
                f.extend([1.0, 0, 0, 0.5, 0.0, 0.0, 0.0, 0.0])

            # -- 板块动量 (5d) --
            for sec in ["科创", "主板大盘", "主板小盘", "红利", "北证"]:
                if sec in sec_daily_ret and dt in sec_daily_ret[sec].index:
                    s = sec_daily_ret[sec]
                    pos = s.index.get_loc(dt)
                    if pos >= 20:
                        mom_20 = (1 + s.iloc[pos-20:pos+1]).prod() - 1
                    else:
                        mom_20 = 0.0
                    f.append(float(mom_20) if not np.isnan(mom_20) else 0.0)
                else:
                    f.append(0.0)

            # -- 波动率 (3d) --
            if not idx_row.empty:
                f.extend([
                    float(idx_vol_20.loc[dt]) if not np.isnan(idx_vol_20.get(dt, np.nan)) else 0.0,
                    float(idx_vol_60.loc[dt]) if not np.isnan(idx_vol_60.get(dt, np.nan)) else 0.0,
                    float(idx_ret_20.loc[dt]) if not np.isnan(idx_ret_20.get(dt, np.nan)) else 0.0,
                ])
            else:
                f.extend([0.0, 0.0, 0.0])

            # -- 概念板块 (8d) --
            dt_key = pd.Timestamp(dt.date())
            if concept_cache is not None:
                cb = concept_cache.get(dt_key)
                if cb is None:
                    prev = sorted(d for d in concept_cache.keys() if d < dt_key)
                    cb = concept_cache[prev[-1]] if prev else {}
                f.extend([
                    cb.get("cb_hot1_5d", 0.0), cb.get("cb_hot2_5d", 0.0),
                    cb.get("cb_hot3_5d", 0.0), cb.get("cb_hot4_5d", 0.0),
                    cb.get("cb_hot5_5d", 0.0), cb.get("cb_breadth", 0.5),
                    cb.get("cb_vol", 0.0), cb.get("cb_rotation", 0.0),
                ])
            else:
                f.extend([0.0] * 8)

            # -- 因子 IC (N维) --
            ic_map = ic_maps.get(dt, {})
            for i in range(nf):
                f.append(float(ic_map.get(i, 0.0)))

            states[dt] = np.array(f, dtype=np.float32)

        return states, rets_5d

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

        # --- 概念板块特征 (8维) — 替换原恐贪占位符 ---
        if self.concept_features is not None:
            try:
                cb = self.concept_features.get_features(date)
                f.extend([
                    cb.get("cb_hot1_5d", 0.0),
                    cb.get("cb_hot2_5d", 0.0),
                    cb.get("cb_hot3_5d", 0.0),
                    cb.get("cb_hot4_5d", 0.0),
                    cb.get("cb_hot5_5d", 0.0),
                    cb.get("cb_breadth", 0.5),
                    cb.get("cb_vol", 0.0),
                    cb.get("cb_rotation", 0.0),
                ])
            except Exception:
                f.extend([0.0] * 8)
        else:
            # 向后兼容：无概念特征时保留原恐贪简化逻辑
            try:
                vol20 = f[13] if len(f) > 13 else 0.02
                mom = f[15] if len(f) > 15 else 0.0
                f.extend([
                    max(0, min(100, 50 - vol20 * 200)),
                    max(0, min(100, 50 + mom * 500)),
                    50.0, 50.0, 50.0,
                    0.0, 0.0, 0.0,  # 补齐 8 维占位
                ])
            except Exception:
                f.extend([50.0] * 5 + [0.0] * 3)

        # --- 因子IC (N维) ---
        for i in range(self.n_factors):
            f.append(float(factor_ic.get(i, 0.0)))

        return np.array(f, dtype=np.float32)
