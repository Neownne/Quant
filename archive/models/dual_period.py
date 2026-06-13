"""双周期模型：日频量价因子 ML 排序 + 月频基本面排雷过滤。

用法:
    model = DualPeriodModel(predictor, pv_factors, fund_factors)
    signals = model.run_daily(trade_date, factor_df, ohlcv_data, extra_data,
                               current_holdings, index_ohlcv)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from factors import FactorEngine
from portfolio.selector import select_topk_ndrop, filter_stocks
from portfolio.allocator import equal_weight, apply_position_limits


class DualPeriodModel:
    """双周期 ML 选股模型。

    每日：price-volume 因子 + 集成预测 → NDrop 增量调仓
    每月首个交易日：fundamental 因子排雷 → 过滤低质量股票
    指数大跌日：空仓规避系统性风险
    """

    def __init__(
        self,
        predictor,
        pv_factor_names: list[str],
        fund_factor_names: list[str],
        top_n: int = 20,
        ndrop_n: int = 2,
        quality_threshold: float = -3.0,
        crash_lookback: int = 15,
        crash_threshold: float = -0.12,
        max_single: float = 0.10,
        max_industry: float = 0.30,
    ):
        self.predictor = predictor
        self.pv_factor_names = pv_factor_names
        self.fund_factor_names = fund_factor_names
        self.top_n = top_n
        self.ndrop_n = ndrop_n
        self.quality_threshold = quality_threshold
        self.crash_lookback = crash_lookback
        self.crash_threshold = crash_threshold
        self.max_single = max_single
        self.max_industry = max_industry

        # 缓存月频质量过滤结果
        self._quality_cache: dict[str, set[str]] = {}
        self._last_quality_month: str | None = None

        # 因子引擎（仅基本面）
        self._fund_engine = FactorEngine(factor_names=fund_factor_names)

    # ── 公开接口 ──────────────────────────────────────────

    def quality_screen(
        self,
        ohlcv: pd.DataFrame,
        extra_data: dict[str, pd.DataFrame],
        trade_date: pd.Timestamp,
    ) -> set[str]:
        """月频质量筛选：计算基本面排雷评分，排除低质量股票。

        返回：passing_codes（通过筛选的股票代码集合）
        """
        month_key = trade_date.strftime("%Y-%m")
        if month_key in self._quality_cache:
            return self._quality_cache[month_key]

        if not self.fund_factor_names:
            return set(ohlcv["code"].unique())

        # 取当日数据计算基本面因子
        day_ohlcv = ohlcv[ohlcv["trade_date"] == trade_date]
        if day_ohlcv.empty:
            return set()

        try:
            fund_df = self._fund_engine.compute(day_ohlcv, extra_data=extra_data)
        except Exception as e:
            logger.warning(f"基本面因子计算失败: {e}")
            return set(day_ohlcv["code"].unique())

        if fund_df.empty:
            return set()

        # 用 fin_audit_score 排雷：评分 >= quality_threshold 的通过
        if "fin_audit_score" in fund_df.columns:
            fund_df["quality_pass"] = fund_df["fin_audit_score"] >= self.quality_threshold
        else:
            fund_df["quality_pass"] = True

        # 单项硬性排除：商誉>30%、质押>80%、负债>70%
        for col, max_val in [("fin_goodwill_ratio", -0.9), ("fin_pledge_risk", -0.9), ("fin_debt_ratio", -0.5)]:
            if col in fund_df.columns:
                fund_df["quality_pass"] &= fund_df[col] > max_val

        passing = set(fund_df[fund_df["quality_pass"]]["code"].unique())
        self._quality_cache[month_key] = passing
        logger.debug(f"月频质量筛选 {month_key}: {len(passing)} 只通过")
        return passing

    def check_index_crash(
        self,
        index_ohlcv: pd.DataFrame,
        trade_date: pd.Timestamp,
    ) -> bool:
        """指数大跌过滤器：中证1000 N日跌幅超阈值则空仓。

        返回 True 表示应空仓。
        """
        if index_ohlcv is None or index_ohlcv.empty:
            return False

        hist = index_ohlcv[index_ohlcv["trade_date"] <= trade_date].tail(self.crash_lookback)
        if len(hist) < max(5, self.crash_lookback // 2):
            return False

        start_close = float(hist["close"].iloc[0])
        end_close = float(hist["close"].iloc[-1])
        if start_close <= 0:
            return False

        return (end_close / start_close - 1) <= self.crash_threshold

    def run_daily(
        self,
        trade_date: pd.Timestamp,
        factor_df: pd.DataFrame,
        ohlcv: pd.DataFrame,
        extra_data: dict[str, pd.DataFrame] | None,
        current_holdings: set[str] | None,
        index_ohlcv: pd.DataFrame | None = None,
        industry_map: dict[str, str] | None = None,
        cash: float = 1_000_000,
        price_map: dict[str, float] | None = None,
        is_month_first: bool = False,
    ) -> dict:
        """单日完整信号生成。

        返回:
            {candidates, selected, weights, target_positions,
             to_buy, to_sell, is_empty, crash_warning, quality_month}
        """
        result = {
            "trade_date": trade_date,
            "candidates": [],
            "selected": [],
            "weights": {},
            "target_positions": {},
            "to_buy": set(),
            "to_sell": set(),
            "is_empty": False,
            "crash_warning": False,
            "quality_month": None,
        }

        # 1. 指数大跌 → 空仓
        if index_ohlcv is not None:
            if self.check_index_crash(index_ohlcv, trade_date):
                logger.warning(f"{trade_date.date()} 指数大跌，空仓")
                result["is_empty"] = True
                result["crash_warning"] = True
                result["to_sell"] = current_holdings.copy() if current_holdings else set()
                return result

        # 2. 月频质量筛选
        month_key = trade_date.strftime("%Y-%m")
        if is_month_first or month_key != self._last_quality_month:
            quality_codes = self.quality_screen(ohlcv, extra_data or {}, trade_date)
            self._last_quality_month = month_key
            self._quality_cache[month_key] = quality_codes
            result["quality_month"] = month_key
        else:
            quality_codes = self._quality_cache.get(month_key, set())

        # 3. 候选池过滤
        valid_codes = set(factor_df["code"].unique())
        if quality_codes:
            valid_codes &= quality_codes

        # ST/次新过滤
        if not factor_df.empty:
            filter_df = factor_df[factor_df["code"].isin(valid_codes)][["code"]].drop_duplicates()
            if "name" in factor_df.columns:
                names = factor_df[["code", "name"]].drop_duplicates(subset="code")
                filter_df = filter_df.merge(names, on="code", how="left")
            filter_df = filter_stocks(filter_df, ref_date=trade_date)
            valid_codes = set(filter_df["code"].unique())

        result["candidates"] = sorted(valid_codes)
        if not valid_codes:
            result["is_empty"] = True
            return result

        # 4. ML 预测 → 排名
        pred_slice = factor_df[factor_df["code"].isin(valid_codes)]
        try:
            scores = self.predictor.predict(pred_slice)
        except Exception as e:
            logger.warning(f"预测失败: {e}")
            return result

        # 转为 Series 供 NDrop
        score_series = pd.Series(
            scores.set_index("code")["score"].to_dict()
        ).sort_values(ascending=False)

        # 5. NDrop 增量调仓
        holdings = current_holdings or set()
        new_holdings, to_buy, to_sell = select_topk_ndrop(
            score_series, current_holdings=holdings,
            K=self.top_n, N=self.ndrop_n,
        )

        result["to_buy"] = to_buy
        result["to_sell"] = to_sell
        result["selected"] = sorted(new_holdings)

        # 6. 权重分配
        if new_holdings:
            alloc = equal_weight(list(new_holdings), cash)
            alloc = apply_position_limits(
                alloc, industry_map or {},
                self.max_single, self.max_industry,
            )
            weight_map = dict(zip(alloc["code"], alloc["weight"]))
            result["weights"] = {c: weight_map.get(c, 0) for c in new_holdings}

            # 目标仓位（股数）
            if price_map:
                for code in new_holdings:
                    w = weight_map.get(code, 0)
                    p = price_map.get(code, 0)
                    if p > 0 and w > 0:
                        target_value = cash * w
                        shares = int(target_value / p / 100) * 100
                        result["target_positions"][code] = {
                            "shares": shares, "weight": w, "price": p,
                        }

        return result

    def to_signal_json(
        self,
        daily_result: dict,
        signal_date: str | None = None,
    ) -> dict:
        """将单日结果转为 QMT 兼容的信号 JSON。

        格式:
            {
                "date": "2026-05-25",
                "signals": [
                    {"code": "000001.SZ", "weight": 0.05, "action": "BUY"},
                    ...
                ],
                "metadata": {"candidates": N, "crash_warning": false}
            }
        """
        date_str = signal_date or str(daily_result.get("trade_date", ""))
        signals = []

        for code in daily_result.get("to_buy", []):
            signals.append({
                "code": _to_qmt_code(code),
                "weight": daily_result.get("weights", {}).get(code, 0),
                "action": "BUY",
            })
        for code in daily_result.get("to_sell", []):
            signals.append({
                "code": _to_qmt_code(code),
                "weight": 0,
                "action": "SELL",
            })

        return {
            "date": date_str,
            "signals": signals,
            "metadata": {
                "n_candidates": len(daily_result.get("candidates", [])),
                "n_selected": len(daily_result.get("selected", [])),
                "crash_warning": daily_result.get("crash_warning", False),
                "quality_month": daily_result.get("quality_month"),
            },
        }


def _to_qmt_code(code: str) -> str:
    """将纯数字 code 转为 QMT 格式（带交易所后缀）。"""
    if "." in code:
        return code
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"
