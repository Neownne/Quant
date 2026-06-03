"""市场状态特征构建器。

构建 ~50 维特征向量供 RL 策略网络使用：
  - 市场宽度 (8维): 涨跌比、涨停数、跌停数、资金流等
  - 板块动量 (5维): 5板块20日收益
  - 波动率 (3维): 指数20/60日波动 + 20日收益
  - 恐贪指标 (5维): 综合恐贪分量
  - 因子IC (N维): 各因子近20日RankIC
"""
import numpy as np
import pandas as pd
from config.sector_map import classify_stock

FEATURE_NAMES = [
    # 市场宽度 (8)
    "mkt_adv_dec", "mkt_limit_up", "mkt_limit_down", "mkt_vol_ratio",
    "mkt_ret_mean", "mkt_ret_std", "mkt_turnover", "mkt_active_pct",
    # 板块动量 (5)
    "sec_mom_kc", "sec_mom_large", "sec_mom_small", "sec_mom_div", "sec_mom_rest",
    # 波动率 (3)
    "idx_vol_20", "idx_vol_60", "idx_ret_20",
    # 恐贪 (5)
    "fg_vol", "fg_money", "fg_mom", "fg_nh", "fg_ad",
]


class StateBuilder:
    """从 OHLCV + 指数数据构建 RL 状态向量。"""

    def __init__(self, n_factors: int = 10):
        self.n_factors = n_factors
        # 动态添加因子IC维度
        self.ic_names = [f"ic_{i}" for i in range(n_factors)]
        self.feature_names = FEATURE_NAMES + self.ic_names

    @property
    def state_dim(self) -> int:
        return len(self.feature_names)

    def build(self, ohlcv: pd.DataFrame, index_df: pd.DataFrame,
              factor_ic: dict[int, float], date) -> np.ndarray:
        """从原始数据构建完整状态向量。

        Args:
            ohlcv: 全量OHLCV数据
            index_df: 指数日线数据
            factor_ic: {factor_index: recent_IC_value}
            date: 目标日期

        Returns:
            np.ndarray: state_dim 维 float32 向量
        """
        date = pd.Timestamp(date)
        features = []

        # --- 市场宽度 ---
        day = ohlcv[ohlcv["trade_date"] == date]
        if not day.empty and "close" in day.columns:
            # 计算日收益率
            prev = ohlcv[ohlcv["trade_date"] < date].groupby("code")["close"].last()
            cur = day.groupby("code")["close"].last()
            common = prev.index.intersection(cur.index)
            if len(common) > 0:
                rets = (cur[common] - prev[common]) / prev[common]
                adv = int((rets > 0).sum())
                dec = int((rets < 0).sum())
                limit_up = int((rets > 0.099).sum())
                limit_down = int((rets < -0.099).sum())
                up_amt = day[day["code"].isin(common[rets > 0])]["amount"].sum() if "amount" in day.columns else 0
                total_amt = day["amount"].sum() if "amount" in day.columns else 1
                features.extend([
                    adv / max(dec, 1),
                    limit_up, limit_down,
                    up_amt / max(total_amt, 1),
                    float(rets.mean()), float(rets.std()) if len(rets) > 1 else 0.0,
                    float(day["turnover"].mean()) if "turnover" in day.columns else 0.0,
                    (adv + dec) / max(len(day), 1),
                ])
            else:
                features.extend([1.0, 0, 0, 0.5, 0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([1.0, 0, 0, 0.5, 0.0, 0.0, 0.0, 0.0])

        # --- 板块动量 (使用 sector_breadth 已有函数) ---
        try:
            from factors.sector_breadth import compute_breadth_features
            csi300 = set()
            try:
                from data.db import get_engine
                from sqlalchemy import text
                e = get_engine()
                with e.connect() as c:
                    rows = c.execute(text("SELECT con_code FROM index_constituent WHERE idx_code='000300'")).fetchall()
                    csi300 = {r[0] for r in rows}
                e.dispose()
            except Exception:
                pass

            codes = ohlcv["code"].unique()
            sector_map = {c: classify_stock(c, csi300, set()) for c in codes}
            breadth = compute_breadth_features(ohlcv, sector_map, date, lookback_days=20)
            for sec in ["科创", "主板大盘", "主板小盘", "红利", "北证"]:
                if sec in breadth:
                    features.append(breadth[sec].get("sector_mom_20", 0.0))
                else:
                    features.append(0.0)
        except Exception:
            features.extend([0.0] * 5)

        # --- 波动率 ---
        idx = index_df[index_df["trade_date"] <= date].copy()
        if len(idx) > 20 and "close" in idx.columns:
            idx["ret"] = idx["close"].pct_change()
            rets = idx["ret"].dropna()
            features.append(float(rets.tail(20).std()) if len(rets) >= 20 else 0.0)
            features.append(float(rets.tail(60).std()) if len(rets) >= 60 else float(rets.std()) if len(rets) > 0 else 0.0)
            features.append(float(rets.tail(20).mean()) if len(rets) >= 20 else 0.0)
        else:
            features.extend([0.0, 0.0, 0.0])

        # --- 恐贪指标 ---
        try:
            from factors.sector_fear_greed import compute_sector_fear_greed
            codes = ohlcv["code"].unique()
            csi300 = set()
            sm = {c: classify_stock(c, csi300, set()) for c in codes}
            fg = compute_sector_fear_greed(ohlcv, sm, date)
            if fg:
                keys = ["fg_volatility", "fg_money_flow", "fg_momentum",
                        "fg_new_high_ratio", "fg_advance_decline"]
                for k in keys:
                    vals = [s[k] for s in fg.values() if k in s]
                    features.append(float(np.mean(vals)) if vals else 50.0)
            else:
                features.extend([50.0] * 5)
        except Exception:
            features.extend([50.0] * 5)

        # --- 因子IC ---
        for i in range(self.n_factors):
            features.append(float(factor_ic.get(i, 0.0)))

        return np.array(features, dtype=np.float32)
