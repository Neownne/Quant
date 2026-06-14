"""ML 因子优化管线 —— 三窗口（训练/验证/测试）+ 因子发现 + XGBoost 选股。

与涨停策略实验室互补：不再做规则筛选，而是用因子+ML 预测未来收益排序。
"""
from __future__ import annotations

import os, sys, json, time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data
from factors import FactorEngine, ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from factors.monitor import compute_ic_series, compute_ic_summary
from archive.models.dataset import build_factor_dataset

# ── 三窗口时间分割（用户指定）──
TRAIN_END = "2024-06-30"
VAL_END = "2025-06-30"
TEST_END = "2026-06-14"

# ── ML 参数 ──
DEFAULT_FACTOR_COUNT = 88         # ALL_FACTORS 总数
MAX_SELECTED_FACTORS = 20         # 正交化后最多保留
IC_THRESHOLD = 0.02               # IC 门禁
T_THRESHOLD = 1.5                 # t 统计量门禁（放宽以保留更多因子）
CORR_THRESHOLD = 0.6              # 正交化相关性阈值
FORWARD_DAYS = 5                  # 预测未来 N 日收益
TOP_N = 20                        # 持仓数


@dataclass
class MLRunResult:
    """一次 ML 因子优化运行的结果。"""
    run_name: str
    train_end: str
    val_end: str
    test_end: str
    # 因子统计
    n_factors_total: int = 0
    n_factors_passed_ic: int = 0
    n_factors_selected: int = 0
    selected_factors: list[str] = field(default_factory=list)
    # 模型指标
    train_sharpe: float = 0.0
    val_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_ret: float = 0.0
    val_ret: float = 0.0
    test_ret: float = 0.0
    train_mdd: float = 0.0
    val_mdd: float = 0.0
    test_mdd: float = 0.0
    # 衰减比（>0.3 表示显著过拟合）
    sharpe_decay_val: float = 0.0
    sharpe_decay_test: float = 0.0
    # 元数据
    elapsed: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["verdict"] = self.verdict
        return d

    @property
    def verdict(self) -> str:
        if self.error:
            return "error"
        if self.test_sharpe < 0 or self.test_mdd > 0.30:
            return "reject"
        if self.sharpe_decay_test > 0.30:
            return "reject"  # 过拟合
        if self.test_sharpe > 0.5 and self.sharpe_decay_test < 0.15:
            return "promising"
        return "baseline"


def load_data(engine, start="2019-01-01", end="2026-06-14"):
    """加载 OHLCV + 行业 + 市值数据。"""
    # 先获取所有非ST股票
    logger.info("加载股票池...")
    codes_df = pd.read_sql(
        text("SELECT code FROM stock_basic WHERE is_st = FALSE AND list_date <= :d"),
        engine, params={"d": end},
    )
    all_codes = codes_df["code"].tolist()
    logger.info(f"股票池: {len(all_codes)} 只")

    logger.info("加载日线数据...")
    daily = load_daily_data(engine, all_codes, start, end, cols=["open", "high", "low", "close", "volume", "amount", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)

    logger.info("加载市值数据...")
    extra = load_mcap_data(engine, all_codes, start, end, use_proxy=True)
    extra["code"] = extra["code"].astype(str).str.zfill(6)

    logger.info("加载行业分类...")
    industry = pd.read_sql(
        text("SELECT code, industry_sw1 FROM stock_industry"),
        engine,
    )
    industry["code"] = industry["code"].astype(str).str.zfill(6)

    # 构建 extra_data
    mcap_df = extra[["code", "trade_date", "market_cap"]].copy()
    mcap_df["log_mcap"] = np.log(mcap_df["market_cap"].clip(lower=0.1))
    mcap_df["trade_date"] = pd.to_datetime(mcap_df["trade_date"])

    # 将 industry 展开为每日（merge asof）
    # 简化：直接在 factor 合并后用 industry 表 join
    ind_daily = daily[["code", "trade_date"]].drop_duplicates().merge(
        industry, on="code", how="left"
    )
    ind_daily["trade_date"] = pd.to_datetime(ind_daily["trade_date"])

    extra_data = {
        "log_mcap": mcap_df[["code", "trade_date", "log_mcap"]],
        "industry_sw1": ind_daily[["code", "trade_date", "industry_sw1"]],
    }

    logger.info(f"日线: {len(daily)} 行, {daily['code'].nunique()} 只")
    return daily, extra_data


def run_factor_pipeline(daily, extra_data, run_name="ml_auto",
                        factor_names=None, forward_days=FORWARD_DAYS,
                        top_n=TOP_N, industry_neutralize=True):
    """完整的因子发现 → ML 训练 → 模拟回测管线。"""
    t0 = time.time()
    result = MLRunResult(run_name=run_name, train_end=TRAIN_END,
                         val_end=VAL_END, test_end=TEST_END)

    try:
        # ── 1. 因子计算 ──
        if factor_names is None:
            factor_names = list(ALL_FACTORS.keys())
        result.n_factors_total = len(factor_names)
        logger.info(f"计算 {len(factor_names)} 个因子...")

        factor_df = build_factor_dataset(
            daily, factor_names=factor_names, label_mode="regression",
            forward_days=forward_days, extra_data=extra_data,
            industry_neutralize=industry_neutralize,
        )
        factor_df["trade_date"] = pd.to_datetime(factor_df["trade_date"])

        # ── 2. 三窗口分割 ──
        train_mask = factor_df["trade_date"] <= TRAIN_END
        val_mask = (factor_df["trade_date"] > TRAIN_END) & (factor_df["trade_date"] <= VAL_END)
        test_mask = (factor_df["trade_date"] > VAL_END) & (factor_df["trade_date"] <= TEST_END)

        train_df = factor_df[train_mask].copy()
        val_df = factor_df[val_mask].copy()
        test_df = factor_df[test_mask].copy()

        if len(train_df) == 0:
            result.error = "训练集为空"
            return result
        logger.info(f"样本: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # ── 3. IC 筛选（仅在训练集上）──
        logger.info("IC 筛选...")
        ic_passed = filter_factors_by_ic(
            train_df, factor_names, ret_col=f"ret_{forward_days}d",
            ic_threshold=IC_THRESHOLD, t_threshold=T_THRESHOLD,
        )
        result.n_factors_passed_ic = len(ic_passed)
        if not ic_passed:
            result.error = f"IC 筛选后零因子通过 (阈值 IC>{IC_THRESHOLD}, t>{T_THRESHOLD})"
            return result
        logger.info(f"  IC 通过: {len(ic_passed)}/{len(factor_names)}")

        # ── 4. 正交化选择（仅在训练集上）──
        logger.info("正交化选择...")
        ic_summary = compute_ic_summary(compute_ic_series(train_df, ic_passed, ret_col=f"ret_{forward_days}d"))
        selected = select_orthogonal_factors(
            train_df, ic_passed, threshold=CORR_THRESHOLD, ic_summary=ic_summary,
        )
        if len(selected) > MAX_SELECTED_FACTORS:
            # 取 IC 最强的 top-MAX
            top_ic = ic_summary.loc[ic_summary.index.isin(selected)]
            top_ic = top_ic.sort_values("ic_mean", key=abs, ascending=False)
            selected = list(top_ic.index[:MAX_SELECTED_FACTORS])

        result.n_factors_selected = len(selected)
        result.selected_factors = selected
        logger.info(f"  正交化保留: {len(selected)} 个因子")
        for f in selected[:8]:
            ic_v = ic_summary.loc[f, "ic_mean"] if f in ic_summary.index else 0
            logger.info(f"    {f}: IC={ic_v:.4f}")

        # ── 5. 训练 XGBoost 排序模型 ──
        logger.info("训练 XGBoost...")
        from xgboost import XGBRegressor

        X_train = train_df[selected].fillna(0).values
        y_train = train_df[f"ret_{forward_days}d"].fillna(0).clip(-0.3, 0.3).values
        X_val = val_df[selected].fillna(0).values
        y_val = val_df[f"ret_{forward_days}d"].fillna(0).clip(-0.3, 0.3).values
        X_test = test_df[selected].fillna(0).values
        y_test = test_df[f"ret_{forward_days}d"].fillna(0).clip(-0.3, 0.3).values

        if len(X_train) < 1000:
            result.error = f"训练样本太少 ({len(X_train)})"
            return result

        model = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # ── 6. 模拟回测（按预测分数排序选股）──
        for name, df, X, y in [("train", train_df, X_train, y_train),
                                ("val", val_df, X_val, y_val),
                                ("test", test_df, X_test, y_test)]:
            if len(df) == 0:
                continue
            scores = model.predict(X)
            df = df.copy()
            df["score"] = scores

            # 按日期分组，每日选 top-N
            equity = []
            for td, group in df.groupby("trade_date"):
                top = group.nlargest(top_n, "score")
                if len(top) > 0:
                    daily_ret = top[f"ret_{forward_days}d"].mean()
                else:
                    daily_ret = 0
                equity.append({"date": str(td)[:10], "return": float(daily_ret)})

            eq_df = pd.DataFrame(equity)
            if len(eq_df) == 0:
                continue
            eq_df["value"] = (1 + eq_df["return"]).cumprod()
            returns = eq_df["return"].values
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            total_ret = float(eq_df["value"].iloc[-1] - 1) if len(eq_df) > 0 else 0
            # MDD
            peak = eq_df["value"].cummax()
            mdd = float(((peak - eq_df["value"]) / peak).max())

            if name == "train":
                result.train_sharpe, result.train_ret, result.train_mdd = sharpe, total_ret, mdd
            elif name == "val":
                result.val_sharpe, result.val_ret, result.val_mdd = sharpe, total_ret, mdd
            else:
                result.test_sharpe, result.test_ret, result.test_mdd = sharpe, total_ret, mdd

        # ── 7. 过拟合检测 ──
        if result.val_sharpe > 0.01:
            result.sharpe_decay_val = max(0, 1 - result.val_sharpe / max(result.train_sharpe, 0.01))
        if result.test_sharpe > 0.01:
            result.sharpe_decay_test = max(0, 1 - result.test_sharpe / max(result.val_sharpe, 0.01))

    except Exception as e:
        logger.error(f"因子管线异常: {e}")
        import traceback
        traceback.print_exc()
        result.error = str(e)[:200]

    result.elapsed = time.time() - t0
    return result


def main():
    """CLI 入口：跑一次 ML 因子优化。"""
    import argparse
    p = argparse.ArgumentParser(description="ML 因子优化管线")
    p.add_argument("--name", default=f"ml_auto_{date.today().strftime('%Y%m%d')}")
    p.add_argument("--forward-days", type=int, default=FORWARD_DAYS)
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--no-neutralize", action="store_true")
    p.add_argument("--factor-subset", type=str, default=None,
                   help="逗号分隔的因子子集（默认全部88个）")
    args = p.parse_args()

    engine = get_engine()
    daily, extra_data = load_data(engine)
    engine.dispose()

    factor_names = None
    if args.factor_subset:
        factor_names = [f.strip() for f in args.factor_subset.split(",")]

    result = run_factor_pipeline(
        daily, extra_data, run_name=args.name, factor_names=factor_names,
        forward_days=args.forward_days, top_n=args.top_n,
        industry_neutralize=not args.no_neutralize,
    )

    print(f"\n{'='*60}")
    print(f"  ML 因子优化: {args.name}")
    print(f"{'='*60}")
    print(f"  因子: {result.n_factors_total} → IC通过{result.n_factors_passed_ic} → 正交保留{result.n_factors_selected}")
    print(f"  三窗口 Sharpe: train={result.train_sharpe:.2f} val={result.val_sharpe:.2f} test={result.test_sharpe:.2f}")
    print(f"  衰减: val={result.sharpe_decay_val:.2%} test={result.sharpe_decay_test:.2%}")
    print(f"  判定: {result.verdict}")
    print(f"  耗时: {result.elapsed:.0f}s")
    if result.selected_factors:
        print(f"  入选因子 ({len(result.selected_factors)}):")
        for f in result.selected_factors[:15]:
            print(f"    {f}")
    if result.error:
        print(f"  ❌ 错误: {result.error}")

    return result


if __name__ == "__main__":
    main()
