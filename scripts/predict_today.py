#!/usr/bin/env python
"""今日持仓预测：加载最新数据 → 因子计算 → 模型训练 → 打分选股。

用法:
    python scripts/predict_today.py --top-n 30
    python scripts/predict_today.py --top-n 15 --factor-preset all
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from data.db import get_engine
from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train_ensemble, train_xgboost, train_lightgbm, EnsemblePredictor
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from portfolio.selector import select_topk_ndrop
from config.settings import TradingConfig


def get_factors_by_preset(preset: str) -> list[str]:
    """因子预设 → 因子名列表（仅返回 FactorEngine 中已注册的因子）。"""
    presets = {
        "momentum": ["mom_20", "mom_60", "ema_ratio_5_20",
                      "bb_position", "macd_dif", "macd_signal",
                      "macd_hist", "vwap_ratio", "vwap_momentum", "vpt",
                      "money_flow", "force_index", "cwt", "turnover_ret_corr"],
        "reversal": ["rev_5", "rev_10", "rev_20", "rsi_7", "rsi_14",
                      "vol_20", "atr_14", "down_vol_ratio",
                      "lower_shadow", "upper_shadow", "intra_day_rev",
                      "body_ratio", "high_low_ratio",
                      "overnight_ret", "overnight_ret_std", "open_auction_jump"],
        "volatility": ["vol_20", "atr_14", "high_low_ratio",
                       "ret_asymmetry", "vol_of_vol", "vol_swing",
                       "vol_conv", "vol_ratio_5_20", "kurtosis_20",
                       "skewness_20", "tail_risk"],
        "liquidity": ["amount_ratio", "turnover_ma_dev", "turnover_mom",
                      "turnover_ret_corr", "illiquidity", "dollar_volume",
                      "turnover_cv", "turnover_5",
                      "turnover_breakout", "turnover_skew"],
        "fundamental": [],  # 财务因子需要额外数据源，暂不在 predict_today 中使用
        "intraday": ["am_ret", "pm_ret", "intra_vol_skew", "close_auction_strength",
                      "volume_concentration", "vwap_gap", "am_pm_divergence"],
    }
    if preset == "all":
        result = []
        for factors in presets.values():
            result.extend(factors)
        result = list(dict.fromkeys(result))
    elif preset in presets:
        result = presets[preset]
    else:
        return []
    # 过滤掉 FactorEngine 不认识的因子名
    valid = [f for f in result if f in ALL_FACTORS]
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--factor-preset", default="all")
    parser.add_argument("--forward-days", type=int, default=1)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--industry-neutralize", action="store_true", help="对各因子做行业截面中性化")
    args = parser.parse_args()

    engine = get_engine()

    # ── 1. 加载数据 ──
    latest_date = pd.read_sql(
        "SELECT MAX(trade_date) FROM stock_daily", engine
    ).iloc[0, 0]
    latest_str = str(latest_date)[:10]
    train_start = str(latest_date - pd.DateOffset(years=args.train_years))[:10]
    logger.info(f"数据区间: {train_start} ~ {latest_str}")

    ohlcv = pd.read_sql(f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily
        WHERE trade_date BETWEEN '{train_start}' AND '{latest_str}'
        ORDER BY code, trade_date
    """, engine)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    logger.info(f"OHLCV: {len(ohlcv)} 行, {ohlcv['code'].nunique()} 只股票")

    # ── extra_data ──
    code_list = ",".join([f"'{c}'" for c in ohlcv["code"].unique()])
    extra_data = {}

    # 分钟频日内特征
    try:
        minute_sql = f"""
            SELECT code, trade_time, period, open, high, low, close, volume, amount
            FROM stock_minute
            WHERE code IN ({code_list})
              AND trade_time >= '{train_start}'::timestamp
              AND trade_time <  ('{latest_str}'::date + interval '1 day')::timestamp
            ORDER BY code, trade_time
        """
        minute_df = pd.read_sql(minute_sql, engine)
        if not minute_df.empty:
            from factors.intraday_minute import build_intraday_daily_features
            minute_extra = build_intraday_daily_features(minute_df)
            extra_data.update(minute_extra)
            logger.info(f"  分钟数据: {len(minute_df)} 行 → {len(minute_extra)} 个日内特征")
    except Exception as e:
        logger.warning(f"  分钟数据加载失败: {e}")

    # 行业数据
    try:
        industry_sql = f"""
            SELECT code, industry_sw1
            FROM stock_industry
            WHERE code IN ({code_list})
        """
        industry_df = pd.read_sql(industry_sql, engine)
        if not industry_df.empty:
            all_dates = sorted(ohlcv["trade_date"].unique())
            ind_expanded = industry_df.merge(
                pd.DataFrame({"trade_date": all_dates}), how="cross"
            )
            extra_data["industry_sw1"] = ind_expanded[["code", "trade_date", "industry_sw1"]]
            logger.info(f"  行业数据: {len(industry_df)} 只股票, {industry_df['industry_sw1'].nunique()} 个行业")
    except Exception as e:
        logger.warning(f"  行业数据加载失败: {e}")

    # ── 2. 因子 ──
    factor_names = get_factors_by_preset(args.factor_preset)
    factor_names = get_factors_by_preset(args.factor_preset)
    if not factor_names:
        factor_names = list(ALL_FACTORS.keys())
    logger.info(f"因子: {len(factor_names)} 个")

    dataset = build_factor_dataset(
        ohlcv, factor_names,
        label_mode="binary", forward_days=args.forward_days,
        extra_data=extra_data if extra_data else None,
        industry_neutralize=args.industry_neutralize,
    )
    dataset = dataset.dropna()
    logger.info(f"有效数据集: {len(dataset)} 行")

    # ── 3. IC + 正交筛选 ──
    passing = filter_factors_by_ic(dataset, factor_names)
    logger.info(f"IC 筛选: {len(factor_names)} → {len(passing)} 个因子")
    if len(passing) < 3:
        passing = factor_names[:min(16, len(factor_names))]

    selected = select_orthogonal_factors(dataset, passing, threshold=0.7)
    logger.info(f"正交筛选: {len(passing)} → {len(selected)} 个因子")

    # ── 4. 训练模型 ──
    X = dataset[selected].fillna(0)
    y = dataset["label"]

    # 按时间切分训练/验证集
    all_dates = sorted(dataset["trade_date"].unique())
    split_idx = len(all_dates) - min(252, len(all_dates) // 4)
    split_date = all_dates[split_idx] if split_idx > 0 and split_idx < len(all_dates) else all_dates[-252]
    train_mask = dataset["trade_date"] < split_date
    val_mask = dataset["trade_date"] >= split_date

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    logger.info(f"训练集: {len(X_train)} 行, 验证集: {len(X_val)} 行")

    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val)
    ensemble = EnsemblePredictor(xgb_model, lgb_model, selected, threshold=0.5)

    # ── 5. 今日预测 ──
    # 使用数据集中最新有效日期（去掉 NaN 后可能比 latest_date 少一天）
    pred_date = dataset["trade_date"].max()
    pred_str = str(pred_date)[:10]
    today_data = dataset[dataset["trade_date"] == pred_date].copy()
    if today_data.empty:
        logger.error(f"无 {pred_str} 数据")
        return

    preds = ensemble.predict(today_data)
    preds = preds.sort_values("score", ascending=False).reset_index(drop=True)

    # ── 6. 质量过滤 ──
    if not args.no_quality:
        quality_pass = _load_quality_filter(engine, latest_str)
        if quality_pass:
            n_before = len(preds)
            preds = preds[preds["code"].isin(quality_pass)]
            logger.info(f"排雷过滤: {n_before} → {len(preds)} 只")

    # ── 6.5 ST 过滤 ──
    n_before = len(preds)
    st_codes = pd.read_sql("SELECT code FROM stock_basic WHERE is_st = TRUE", engine)
    if not st_codes.empty:
        st_set = set(st_codes["code"].tolist())
        preds = preds[~preds["code"].isin(st_set)]
        if len(preds) < n_before:
            logger.info(f"ST 过滤: {n_before} → {len(preds)} 只")

    # ── 7. NDrop 选股 ──
    scores_series = pd.Series(preds["score"].values, index=preds["code"].values)
    scores_series = scores_series.sort_values(ascending=False)
    new_holdings, to_buy, _ = select_topk_ndrop(
        scores_series, current_holdings=set(),
        K=args.top_n, N=TradingConfig.NDROP_N,
    )

    # ── 8. 输出 ──
    top_codes = list(new_holdings)[:args.top_n]
    print(f"\n=== 今日持仓建议 (数据日: {pred_str}, 预测日: {latest_str}) ===")
    print(f"策略: ML-动态多因子v1.11")
    print(f"因子: {len(selected)} 个 (从 {len(factor_names)} 筛选)")
    print(f"模型: XGBoost + LightGBM 集成")
    print(f"候选池: {len(preds)} 只 (排雷后)")
    print(f"持仓数: {len(top_codes)} 只")
    print()

    # 获取股票名称
    if top_codes:
        code_list = ",".join([f"'{c}'" for c in top_codes])
        names = pd.read_sql(
            f"SELECT code, name FROM stock_basic WHERE code IN ({code_list})",
            engine
        )
        name_map = dict(zip(names["code"], names["name"]))

        print(f"{'代码':<10s} {'名称':<10s} {'得分':>8s}")
        print("-" * 30)
        for i, code in enumerate(top_codes):
            score = scores_series.get(code, 0)
            name = name_map.get(code, "?")
            print(f"{code:<10s} {name:<10s} {score:>8.4f}")

    # Top-5 factor importance
    if hasattr(xgb_model, "feature_importances_"):
        imp = dict(zip(selected, xgb_model.feature_importances_))
        top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop-5 因子重要性 (XGBoost):")
        for f, v in top5:
            print(f"  {f:<30s} {v:.4f}")

    engine.dispose()


def _load_quality_filter(engine, latest_str: str) -> set | None:
    """加载最新排雷过滤结果（8项检查，允许≤3项违规）。"""
    try:
        # Load financial data
        fin = pd.read_sql(f"""
            SELECT code, report_date, adjusted_profit, net_profit,
                   total_assets, total_liability, goodwill, holder_equity,
                   operating_cash_flow, roe, net_margin
            FROM stock_financial
            WHERE report_date >= '{str(pd.Timestamp(latest_str) - pd.DateOffset(months=9))[:10]}'
            ORDER BY code, report_date
        """, engine)
        if fin.empty:
            return None

        # Get latest report per stock
        fin["report_date"] = pd.to_datetime(fin["report_date"])
        fin = fin.sort_values("report_date").groupby("code").last().reset_index()

        # Load pledge data
        pledge_df = pd.read_sql(f"""
            SELECT code, trade_date, pledge_ratio
            FROM stock_pledge
            WHERE trade_date <= '{latest_str}'
            ORDER BY code, trade_date
        """, engine)
        if not pledge_df.empty:
            pledge_df["trade_date"] = pd.to_datetime(pledge_df["trade_date"])
            pledge_latest = pledge_df.sort_values("trade_date").groupby("code").last()
            pledge_map = pledge_latest["pledge_ratio"].to_dict()
        else:
            pledge_map = {}

        passing = set()
        for _, row in fin.iterrows():
            violations = 0
            code = row["code"]

            # 1. 扣非净利润为负
            adj = row.get("adjusted_profit")
            if pd.notna(adj) and float(adj) < 0:
                violations += 1

            # 2. 净利润为负
            np_val = row.get("net_profit")
            if pd.notna(np_val) and float(np_val) < 0:
                violations += 1

            # 3. 资产负债率 > 70%
            assets = row.get("total_assets")
            liability = row.get("total_liability")
            if pd.notna(assets) and pd.notna(liability) and float(assets) > 0:
                if float(liability) / float(assets) > 0.70:
                    violations += 1

            # 4. 商誉 / 股东权益 > 30%
            gw = row.get("goodwill")
            equity = row.get("holder_equity")
            if pd.notna(gw) and pd.notna(equity) and float(equity) > 0:
                if float(gw) / float(equity) > 0.30:
                    violations += 1

            # 5. 大股东高质押 > 80%
            pl_ratio = pledge_map.get(code)
            if pl_ratio is not None and pd.notna(pl_ratio) and float(pl_ratio) > 0.80:
                violations += 1

            # 6. 净利润>0 但经营现金流<0
            ocf = row.get("operating_cash_flow")
            if pd.notna(np_val) and pd.notna(ocf) and float(np_val) > 0 and float(ocf) < 0:
                violations += 1

            # 7. ROE 为负
            roe_val = row.get("roe")
            if pd.notna(roe_val) and float(roe_val) < 0:
                violations += 1

            # 8. 净利率为负
            nm_val = row.get("net_margin")
            if pd.notna(nm_val) and float(nm_val) < 0:
                violations += 1

            if violations <= 3:
                passing.add(code)

        return passing
    except Exception as e:
        logger.warning(f"排雷加载失败: {e}")
        return None


if __name__ == "__main__":
    main()
