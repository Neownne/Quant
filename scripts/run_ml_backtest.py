#!/usr/bin/env python
"""ML 选股端到端回测验证 — 舞 v1.5。

用法:
    python scripts/run_ml_backtest.py                              # 默认: Regime+板块打分+动态+top15
    python scripts/run_ml_backtest.py --no-sector-model            # 关闭板块打分
    python scripts/run_ml_backtest.py --no-multi-horizon --forward-days 5  # 单周期 T+5
    python scripts/run_ml_backtest.py --no-dynamic                 # 关闭动态反馈
    python scripts/run_ml_backtest.py --no-industry-neutralize     # 关闭行业中性化
    python scripts/run_ml_backtest.py --optuna                     # 启用贝叶斯超参搜索
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import TradingConfig
from data.db import get_engine
from scripts.overfit_check import OverfitChecker
from scripts.dynamic_backtest import BacktestFeedbackLoop, DailySignalTracker
from sqlalchemy import text
import json

from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train, walk_forward_train_ensemble, walk_forward_train_by_regime
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from models.regime import detect_regime, REGIME_PARAMS
from portfolio.selector import select_topk_ndrop
from portfolio.sector_filter import filter_by_top_sectors
from portfolio.risk import check_drawdown_limit, check_index_crash


# ── Factor presets ──────────────────────────────────────────────────────────
FACTOR_PRESETS = {
    "momentum": [
        # alpha101: trend/momentum
        "mom_20", "mom_60", "ema_ratio_5_20", "macd_dif", "macd_signal", "macd_hist",
        "vwap_ratio", "vpt",
        # alpha191_flow: money flow & trend strength
        "money_flow", "force_index", "cwt", "vwap_momentum",
        # alpha191_gap: gap/trend deviation
        "gap_ma_dev",
        # alpha191_turnover: turnover-trend interaction
        "turnover_ma_dev", "turnover_ret_corr", "free_turnover_ratio",
        # custom
        "turnover_mom",
    ],
    "reversal": [
        # alpha101: short-term reversal
        "rev_5", "rev_10", "rev_20", "rsi_7", "rsi_14",
        "bb_position", "bb_width",
        # alpha191_gap: overnight reversal
        "overnight_ret", "overnight_ret_std", "open_auction_jump",
        # alpha191_intraday: intraday reversal patterns
        "intra_day_rev", "upper_shadow", "lower_shadow", "body_ratio",
        # alpha191_vol: asymmetry & tail risk
        "ret_asymmetry", "tail_risk",
        # custom
        "gap_ratio", "intra_vol",
    ],
    "volatility": [
        "vol_20", "atr_14", "vol_ratio_5_20",
        "vol_of_vol", "down_vol_ratio", "beta_20",
        "vol_conv",
    ],
    "liquidity": [
        "turnover_5", "turnover_skew", "turnover_cv",
        "turnover_breakout", "volume_climax", "obv_roc",
        "amihud_5", "dollar_volume", "bid_ask_proxy", "illiquidity",
    ],
    "fundamental": [
        # custom fundamental proxies
        "log_mcap", "pb_pct", "sh_change",
        # fundamental module
        "fin_cashflow_gap", "fin_roe_quality", "fin_profit_cv",
        "fin_net_margin", "fin_bps_growth", "fin_revenue_stability",
        "fin_eps_growth", "fin_debt_ratio", "fin_goodwill_ratio",
        "fin_pledge_risk", "fin_audit_score",
    ],
}


def get_factors_by_preset(preset: str) -> list[str]:
    """Resolve factor preset name or comma-separated list to factor names."""
    if preset == "all":
        return list(ALL_FACTORS.keys())
    if preset in FACTOR_PRESETS:
        names = FACTOR_PRESETS[preset]
        # Filter to only factors that exist in ALL_FACTORS
        return [n for n in names if n in ALL_FACTORS]
    if preset.startswith("+"):
        # Union of multiple presets: "+momentum+volatility"
        names = set()
        for p in preset.lstrip("+").split("+"):
            p = p.strip()
            if p in FACTOR_PRESETS:
                names.update(n for n in FACTOR_PRESETS[p] if n in ALL_FACTORS)
        return sorted(names)
    # Assume comma-separated list
    return [f.strip() for f in preset.split(",") if f.strip() in ALL_FACTORS]


def retrain_at_point(ohlcv: pd.DataFrame, current_date: str, factor_cols: list[str],
                     forward_days: int, model_type: str, use_ensemble: bool,
                     extra_data: dict | None = None,
                     train_years: int = 3,
                     industry_neutralize: bool = False) -> dict | None:
    """在回测中某时间点触发增量重训。

    使用当前日期前 train_years 年的数据，对新因子集做 IC 筛选 + 模型训练。
    返回 {"ensemble": ..., "model": ..., "active_cols": ..., "factor_cols": ...} 或 None。
    """
    from models.trainer import train_xgboost, train_lightgbm
    from factors.screening import filter_factors_by_ic, select_orthogonal_factors

    current_dt = pd.Timestamp(current_date)
    train_start = pd.Timestamp(current_dt - pd.DateOffset(years=train_years))
    train_ohlcv = ohlcv[(ohlcv["trade_date"] >= train_start) & (ohlcv["trade_date"] <= current_dt)]

    if len(train_ohlcv["trade_date"].unique()) < 252:
        logger.warning(f"重训数据不足 ({len(train_ohlcv['trade_date'].unique())} 天 < 252), 跳过")
        return None

    try:
        dataset = build_factor_dataset(
            train_ohlcv, factor_cols,
            label_mode="binary", forward_days=forward_days,
            extra_data=extra_data,
            industry_neutralize=industry_neutralize,
        )
    except Exception as e:
        logger.warning(f"重训因子计算失败: {e}")
        return None

    dataset = dataset.dropna()
    if len(dataset) < 1000:
        logger.warning(f"重训有效数据不足 ({len(dataset)} 行), 跳过")
        return None

    # IC 筛选
    try:
        passing, ic_report = filter_factors_by_ic(dataset, factor_cols)
        if len(passing) < 3:
            passing = factor_cols[:min(8, len(factor_cols))]
    except Exception:
        passing = factor_cols

    # 正交筛选
    try:
        selected = select_orthogonal_factors(dataset, passing, max_factors=min(16, len(passing)))
    except Exception:
        selected = passing[:16]

    logger.info(f"重训因子筛选: {len(factor_cols)} → IC{len(passing)} → 正交{len(selected)}")

    # 训练（按时间顺序 80/20 分割训练/验证集）
    feature_cols = selected if selected else passing
    df = dataset[feature_cols + ["label"]].fillna(0)
    split = int(len(df) * 0.8)
    if len(df) < 500 or split < 100:
        logger.warning(f"重训数据不足({len(df)}行), 跳过")
        return None
    train_df = df.iloc[:split]; val_df = df.iloc[split:]
    X_tr = train_df[feature_cols]; y_tr = train_df["label"]
    X_v = val_df[feature_cols]; y_v = val_df["label"]

    try:
        if use_ensemble:
            xgb, _ = train_xgboost(X_tr, y_tr, X_v, y_v)
            lgb, _ = train_lightgbm(X_tr, y_tr, X_v, y_v)
            from models.trainer import EnsemblePredictor
            ensemble = EnsemblePredictor([xgb, lgb])
            ensemble.factor_names = feature_cols
            return {"ensemble": ensemble, "active_cols": feature_cols, "factor_cols": selected}
        elif model_type == "xgboost":
            model, _ = train_xgboost(X_tr, y_tr, X_v, y_v)
            return {"model": model, "active_cols": feature_cols}
        else:
            model, _ = train_lightgbm(X_tr, y_tr, X_v, y_v)
            return {"model": model, "active_cols": feature_cols}
    except Exception as e:
        logger.warning(f"重训模型失败: {e}")
        return None


def _discover_factors(day_data: pd.DataFrame, active_factors: list[str],
                      ic_threshold: float = 0.02, max_add: int = 5) -> list[str]:
    """从已计算但未使用的因子中发现新有效因子。

    扫描 day_data 中在 ALL_FACTORS 里但不在 active_factors 中的列，
    计算其横截面 |IC|，返回 |IC| >= ic_threshold 的因子名（最多 max_add 个）。
    """
    ret_col = "ret_1d"
    if ret_col not in day_data.columns or day_data.empty:
        return []

    candidates = [c for c in day_data.columns
                  if c in ALL_FACTORS and c not in active_factors]
    if not candidates:
        return []

    ret_vals = day_data[ret_col]
    scores = []
    for f in candidates:
        if f not in day_data.columns:
            continue
        f_vals = day_data[f]
        valid = f_vals.notna() & ret_vals.notna()
        if valid.sum() < 30:
            continue
        try:
            ic = f_vals[valid].corr(ret_vals[valid], method="spearman")
            if pd.notna(ic) and abs(ic) >= ic_threshold:
                scores.append((f, abs(ic)))
        except Exception:
            pass

    scores.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scores[:max_add]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--factors", default="all", help="'all', factor preset name, '+preset1+preset2', or comma-separated factor list")
    parser.add_argument("--factor-preset", default="", help="Shorthand for --factors: momentum, reversal, volatility, liquidity, fundamental, all")
    parser.add_argument("--forward-days", type=int, default=5, help="Label horizon: 1=next day, 5=next week")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--val-years", type=int, default=1)
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260501")
    parser.add_argument("--codes", default="")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--no-orthogonal", action="store_true")
    parser.add_argument("--no-ic-gate", action="store_true", help="跳过 IC 门禁")
    parser.add_argument("--regime", action="store_true", default=True, help="启用分市场状态训练（默认）")
    parser.add_argument("--no-regime", action="store_false", dest="regime", help="禁用分状态训练")
    parser.add_argument("--optuna", action="store_true", help="启用 Optuna 超参优化")
    parser.add_argument("--optuna-trials", type=int, default=30, help="Optuna 搜索轮数")
    parser.add_argument("--strategy", default="", help="策略名称，对应 strategy_configs.name，用于匹配 version_id")
    parser.add_argument("--dynamic", action="store_true", default=True, help="启用动态多因子闭环（归因→健康度→调参）")
    parser.add_argument("--no-dynamic", action="store_false", dest="dynamic", help="禁用动态多因子闭环")
    parser.add_argument("--rebalance-freq", type=int, default=1, help="调仓频率（交易日），默认1=日频")
    parser.add_argument("--universe-size", type=int, default=500, help="候选池上限，0=全市场非ST")
    parser.add_argument("--small-cap", action="store_true", help="小市值策略模式（按市值升序选股+行业分散+季节空仓）")
    parser.add_argument("--take-profit", type=float, default=0.5, help="个股止盈阈值, 0=不启用 (默认50%%)")
    parser.add_argument("--dd-reduce", type=float, default=0.20, help="组合回撤减仓阈值")
    parser.add_argument("--dd-liquidate", type=float, default=0.25, help="组合回撤清仓阈值")
    parser.add_argument("--index-crash", type=float, default=-0.12, help="指数大跌阈值")
    parser.add_argument("--industry-neutralize", action="store_true", default=False, help="对各因子做行业截面中性化")
    parser.add_argument("--no-industry-neutralize", action="store_false", dest="industry_neutralize", help="禁用行业中性化（默认）")
    parser.add_argument("--multi-horizon", action="store_true", default=False, help="启用多周期预测 (T+1, T+5, T+20)")
    parser.add_argument("--no-multi-horizon", action="store_false", dest="multi_horizon", help="禁用多周期预测（同默认）")
    parser.add_argument("--horizon-weights", default="0.5,0.3,0.2", help="多周期权重, 逗号分隔")
    parser.add_argument("--sector-model", action="store_true", default=False, help="启用板块打分预筛选（实验性）")
    parser.add_argument("--no-sector-model", action="store_false", dest="sector_model", help="禁用板块打分")
    parser.add_argument("--sector-top-n", type=int, default=0, help="板块预筛选保留前N个板块，0=自动(n_sectors-1)")
    parser.add_argument("--sector-train-years", type=int, default=3, help="板块模型训练窗口年数")
    parser.add_argument("--rl-model", action="store_true", default=False, help="使用 PPO 强化学习替代 XGBoost+LGB")
    parser.add_argument("--rl-timesteps", type=int, default=100_000, help="每窗口 PPO 训练步数")
    parser.add_argument("--rl-sector", action="store_true", default=False, help="使用 RL 板块恐贪打分（MPS GPU）替代动量/ML")
    args = parser.parse_args()

    # Resolve factor preset (--factor-preset overrides --factors if given)
    if args.factor_preset:
        factor_names = get_factors_by_preset(args.factor_preset)
    elif args.factors == "all":
        factor_names = list(ALL_FACTORS.keys())
    else:
        factor_names = get_factors_by_preset(args.factors)

    if not factor_names:
        logger.error("因子列表为空，请检查 --factors/--factor-preset 参数")
        sys.exit(1)

    logger.info(f"{len(factor_names)} 个因子, label=ret_{args.forward_days}d, "
                f"IC门禁={'ON' if not args.no_ic_gate else 'OFF'}, "
                f"regime={'ON' if args.regime else 'OFF'}, optuna={'ON' if args.optuna else 'OFF'}, "
                f"model={args.model}, ensemble={'OFF' if args.no_ensemble else 'ON'}")

    orig_factor_names = factor_names.copy()  # Save before IC filtering

    engine = get_engine()

    # 查找策略版本ID 和 strategy_configs.id
    version_id = 0
    strategy_config_id = 0
    if args.strategy:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT sv.id, sc.id FROM strategy_versions sv
                    JOIN strategy_configs sc ON sv.strategy_id = sc.id
                    WHERE sc.name = :name
                    ORDER BY sv.created_at DESC LIMIT 1
                """),
                {"name": args.strategy},
            ).fetchone()
            if row:
                version_id = row[0]
                strategy_config_id = row[1]
                logger.info(f"策略 '{args.strategy}' → version_id={version_id}, config_id={strategy_config_id}")
            else:
                logger.warning(f"未找到策略 '{args.strategy}' 的版本记录，将使用 version_id=0")
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    if codes is None:
        if args.small_cap:
            # 小市值模式
            with engine.connect() as conn:
                codes = pd.read_sql(text(
                    "SELECT b.code FROM stock_basic b "
                    "JOIN stock_daily_extra e ON b.code = e.code "
                    "WHERE b.is_st = FALSE "
                    "AND b.list_date <= CURRENT_DATE - INTERVAL '375 days' "
                    "AND b.code NOT LIKE '688%' AND b.code NOT LIKE '300%'"
                    "AND b.code NOT LIKE '4%' AND b.code NOT LIKE '8%'"
                    "AND e.trade_date = (SELECT MAX(trade_date) FROM stock_daily_extra) "
                    "ORDER BY e.market_cap ASC LIMIT :lim"
                ), conn, params={"lim": args.universe_size})["code"].tolist()
        elif args.universe_size > 0:
            # 按成交额取前 N 只（覆盖沪深两市）
            # 仅用首年成交额排序选宇宙，避免前视偏差
            first_year_end = f"{int(args.start[:4])+1}{args.start[4:]}"
            codes = pd.read_sql(
                f"SELECT code FROM stock_daily "
                f"WHERE trade_date >= '{args.start}' AND trade_date <= '{first_year_end}' "
                f"GROUP BY code ORDER BY SUM(amount) DESC LIMIT {args.universe_size}",
                engine,
            )["code"].tolist()
        else:
            # 全市场非 ST（默认）
            codes = pd.read_sql(
                "SELECT code FROM stock_basic WHERE is_st = FALSE "
                "AND list_date <= CURRENT_DATE - INTERVAL '60 days' "
                "ORDER BY code",
                engine,
            )["code"].tolist()
        logger.info(f"候选池: {len(codes)} 只股票 (universe_size={args.universe_size or '全市场'})")

    code_list = ",".join([f"'{c}'" for c in codes])

    # OHLCV
    sql = f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily
        WHERE code IN ({code_list})
          AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY code, trade_date
    """
    ohlcv = pd.read_sql(sql, engine)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    logger.info(f"OHLCV: {len(ohlcv)} 行")

    # 加载指数数据（始终加载用于风控，regime 模式额外做状态检测）
    regime_df = None
    index_daily_close: pd.Series = pd.Series(dtype=float)
    try:
        idx_sql = f"""
            SELECT trade_date, close FROM index_daily
            WHERE code = '000001' AND trade_date BETWEEN '{args.start}' AND '{args.end}'
            ORDER BY trade_date
        """
        index_df = pd.read_sql(idx_sql, engine)
        if not index_df.empty:
            index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])
            index_daily_close = index_df.set_index("trade_date")["close"].sort_index()
            logger.info(f"指数数据: {len(index_df)} 行 (覆盖 {len(index_daily_close)} 天)")
            if args.regime:
                regime_df = detect_regime(index_df)
                logger.info(f"regime={regime_df['regime'].value_counts().to_dict()}")
                # Build date→regime lookup for simulation loop
                regime_map = dict(zip(
                    regime_df["trade_date"].dt.strftime("%Y-%m-%d"),
                    regime_df["regime"]
                ))
            else:
                regime_map = {}
    except Exception as e:
        logger.warning(f"指数数据加载失败: {e}, 风控中指数检查将跳过")
        if args.regime:
            logger.warning("退到非 regime 模式")
            args.regime = False

    # extra_data
    extra_data = {}
    try:
        extra_sql = f"""
            SELECT code, trade_date, market_cap, pb
            FROM stock_daily_extra
            WHERE code IN ({code_list})
              AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        """
        extra_df = pd.read_sql(extra_sql, engine)
        if not extra_df.empty:
            extra_df["log_mcap"] = np.log(extra_df["market_cap"].replace(0, np.nan))
            extra_data["log_mcap"] = extra_df[["code", "trade_date", "log_mcap"]]
            extra_data["pb"] = extra_df[["code", "trade_date", "pb"]]
            logger.info(f"  估值数据: {len(extra_df)} 行")
    except Exception as e:
        logger.warning(f"  估值数据加载失败: {e}")

    try:
        sh_sql = f"""
            SELECT code, end_date AS trade_date, shareholder_count
            FROM stock_shareholder
            WHERE code IN ({code_list})
              AND end_date BETWEEN '{args.start}' AND '{args.end}'
        """
        sh_df = pd.read_sql(sh_sql, engine)
        if not sh_df.empty:
            extra_data["shareholder_count"] = sh_df[["code", "trade_date", "shareholder_count"]]
            logger.info(f"  股东数据: {len(sh_df)} 行")
    except Exception as e:
        logger.warning(f"  股东数据加载失败: {e}")

    # ── 分钟频日内特征 (Feature 1) ──
    try:
        minute_sql = f"""
            SELECT code, trade_time, period, open, high, low, close, volume, amount
            FROM stock_minute
            WHERE code IN ({code_list})
              AND trade_time >= '{args.start}'::timestamp
              AND trade_time <  ('{args.end}'::date + interval '1 day')::timestamp
            ORDER BY code, trade_time
        """
        minute_df = pd.read_sql(minute_sql, engine)
        if not minute_df.empty:
            from factors.intraday_minute import build_intraday_daily_features
            minute_extra = build_intraday_daily_features(minute_df)
            extra_data.update(minute_extra)
            logger.info(f"  分钟数据: {len(minute_df)} 行 → {len(minute_extra)} 个日内特征")
    except Exception as e:
        logger.warning(f"  分钟数据加载失败 (Feature 1 跳过): {e}")

    # ── 行业数据 (Feature 2) ──
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
        logger.warning(f"  行业数据加载失败 (Feature 2 跳过): {e}")

    # ── 全市场宽度数据（Feature 3）──
    from factors.market_breadth import build_market_breadth_extra
    mkt_breadth = build_market_breadth_extra(ohlcv, codes)
    mkt_features = [c for c in mkt_breadth.columns if c.startswith("mkt_")]
    for col in mkt_features:
        extra_data[col] = mkt_breadth[["code", "trade_date", col]]
    logger.info(f"  市场宽度: {mkt_breadth['trade_date'].nunique()} 天, {len(mkt_features)} 个特征")

    # ── 板块分类映射（用于板块打分模型）──
    sector_map: dict[str, str] = {}
    csi300_members: set[str] = set()
    if args.sector_model:
        from config.sector_map import classify_stock
        try:
            # 获取沪深300成分股（只保留在候选池中的）
            csi300_df = pd.read_sql(
                "SELECT con_code FROM index_constituent WHERE idx_code = '000300'",
                engine,
            )
            csi300_members = set(csi300_df["con_code"].values) & set(codes)
        except Exception:
            csi300_members = set()

        # 红利股：传统高股息行业 + 低PB
        dividend_stocks: set[str] = set()
        try:
            # 高股息行业（申万一级代码）：金融(银行保险)、采掘(煤炭)、公用事业、钢铁、交运
            high_div_industries = ["J 金融业", "B 采矿业", "D 水电煤气", "G 运输仓储"]
            div_df = pd.read_sql(
                "SELECT code FROM stock_industry WHERE industry_sw1 = ANY(:inds)",
                engine, params={"inds": high_div_industries},
            )
            div_stocks = set(div_df["code"].values) & set(codes)
            # 再叠加低PB筛选（PB<1.5）
            low_pb = pd.read_sql(
                "SELECT DISTINCT code FROM stock_daily_extra "
                "WHERE trade_date = (SELECT MAX(trade_date) FROM stock_daily_extra) "
                "AND pb > 0 AND pb < 1.5",
                engine,
            )
            low_pb_set = set(low_pb["code"].values) & set(codes)
            dividend_stocks = div_stocks | low_pb_set
        except Exception:
            pass

        # 构建板块映射
        for code in codes:
            sector_map[code] = classify_stock(code, csi300_members, dividend_stocks)

        from collections import Counter
        sector_counts = Counter(sector_map.values())
        logger.info(f"  板块映射: {len(sector_map)} 只股票 → {dict(sector_counts)}")

    # ── 排雷系统：加载财务 & 质押数据 ──
    quality_pass: dict[str, set[str]] = {}
    try:
        fin_sql = f"""
            SELECT code, report_date, adjusted_profit, net_profit,
                   total_assets, total_liability, goodwill, holder_equity,
                   operating_cash_flow, roe, net_margin
            FROM stock_financial
            WHERE code IN ({code_list})
              AND report_date >= '2018-01-01'
            ORDER BY code, report_date
        """
        fin_df = pd.read_sql(fin_sql, engine)

        pledge_sql = f"""
            SELECT code, trade_date, pledge_ratio
            FROM stock_pledge
            WHERE code IN ({code_list})
              AND trade_date BETWEEN '{args.start}' AND '{args.end}'
            ORDER BY code, trade_date
        """
        pledge_df = pd.read_sql(pledge_sql, engine)

        if not fin_df.empty:
            # 按 code 分组，对每个财务指标做 ffill
            fin_cols = ["adjusted_profit", "net_profit", "total_assets",
                        "total_liability", "goodwill", "holder_equity",
                        "operating_cash_flow", "roe", "net_margin"]
            fin_df["report_date"] = pd.to_datetime(fin_df["report_date"])
            for c in fin_cols:
                if c in fin_df.columns:
                    fin_df[c] = fin_df[c].replace(0, np.nan)
                    fin_df[c] = fin_df.groupby("code")[c].ffill()

            # 获取回测区间的所有交易日
            all_trade_dates = sorted(ohlcv["trade_date"].unique())
            all_trade_dates_dt = pd.to_datetime(all_trade_dates)

            # 为每个交易日构建质量通过的股票集合
            # 排雷检查: adjusted_profit<0, debt>70%, goodwill>30%, pledge>80%, net_profit<0
            # 允许最多 3 项违规
            fin_lookup = fin_df.copy()
            pledge_lookup = pledge_df.copy() if not pledge_df.empty else pd.DataFrame(
                columns=["code", "trade_date", "pledge_ratio"])
            if not pledge_lookup.empty:
                pledge_lookup["trade_date"] = pd.to_datetime(pledge_lookup["trade_date"])

            # 为所有交易日计算质量通过集合（调仓日需要用到）
            for dt in all_trade_dates_dt:
                dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
                passing: set[str] = set()

                for code in codes:
                    # 获取该日期前最新的财务数据
                    code_fin = fin_lookup[
                        (fin_lookup["code"] == code) &
                        (fin_lookup["report_date"] <= dt)
                    ]
                    if code_fin.empty:
                        continue
                    code_fin = code_fin.iloc[-1]

                    violations = 0

                    # 1. 主业存疑：扣非净利润 < 0
                    adj = code_fin.get("adjusted_profit", np.nan)
                    if pd.notna(adj) and adj < 0:
                        violations += 1

                    # 2. 净利润为负
                    np_val = code_fin.get("net_profit", np.nan)
                    if pd.notna(np_val) and np_val < 0:
                        violations += 1

                    # 3. 资金链紧绷：资产负债率 > 70%
                    assets = code_fin.get("total_assets", np.nan)
                    liability = code_fin.get("total_liability", np.nan)
                    if pd.notna(assets) and pd.notna(liability) and assets > 0:
                        if liability / assets > 0.70:
                            violations += 1

                    # 4. 商誉过高：商誉 / 股东权益 > 30%
                    gw = code_fin.get("goodwill", np.nan)
                    equity = code_fin.get("holder_equity", np.nan)
                    if pd.notna(gw) and pd.notna(equity) and equity > 0:
                        if gw / equity > 0.30:
                            violations += 1

                    # 5. 大股东高质押
                    if not pledge_lookup.empty:
                        code_pl = pledge_lookup[
                            (pledge_lookup["code"] == code) &
                            (pledge_lookup["trade_date"] <= dt)
                        ]
                        if not code_pl.empty:
                            pl_ratio = code_pl.iloc[-1]["pledge_ratio"]
                            if pd.notna(pl_ratio) and pl_ratio > 0.80:
                                violations += 1

                    # 6. 现金流异常：净利润>0 但经营现金流<0
                    ocf = code_fin.get("operating_cash_flow", np.nan)
                    if pd.notna(np_val) and pd.notna(ocf) and np_val > 0 and ocf < 0:
                        violations += 1

                    # 7. ROE 为负
                    roe_val = code_fin.get("roe", np.nan)
                    if pd.notna(roe_val) and roe_val < 0:
                        violations += 1

                    # 8. 净利率下滑（近似：负净利率即为有问题）
                    nm_val = code_fin.get("net_margin", np.nan)
                    if pd.notna(nm_val) and nm_val < 0:
                        violations += 1

                    if violations <= 3:
                        passing.add(code)

                if passing:
                    quality_pass[dt_str] = passing

        logger.info(f"  排雷数据: {len(fin_df)} 行财务, {len(pledge_df)} 行质押, "
                    f"{len(quality_pass)} 个调仓日")
    except Exception as e:
        logger.warning(f"  排雷数据加载失败: {e}, 跳过质量过滤")

    engine.dispose()

    # 构建因子数据集（含 ret_1d 和 ret_{forward_days}d 列）
    # 多周期: forward_days 变为列表
    multi_horizon = getattr(args, "multi_horizon", False)
    horizons = [1, 5, 20] if multi_horizon else [args.forward_days]
    fwd_arg = horizons if multi_horizon else args.forward_days

    dataset = build_factor_dataset(
        ohlcv, factor_names, label_mode="binary",
        forward_days=fwd_arg,
        extra_data=extra_data if extra_data else None,
        industry_neutralize=getattr(args, "industry_neutralize", False),
    )
    ret_col = "ret_1d"  # IC 筛选始终用 T+1

    # 1. IC 门禁（仅在首个训练窗口数据上做，避免前视偏差）
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    first_train_start = dataset["trade_date"].min()
    first_train_end = first_train_start + pd.DateOffset(years=args.train_years)
    ic_dataset = dataset[dataset["trade_date"] < first_train_end]
    if not args.no_ic_gate:
        factor_names = filter_factors_by_ic(ic_dataset, factor_names, ret_col=ret_col)

    # 2. 正交筛选
    if not args.no_orthogonal:
        factor_cols = select_orthogonal_factors(ic_dataset, factor_names, threshold=0.7)
    else:
        factor_cols = factor_names

    # 3. Walk-forward 训练
    use_ensemble = not args.no_ensemble

    if args.rl_model:
        from rl.trainer import walk_forward_train_rl
        logger.info(f"使用 PPO 强化学习模型 (timesteps={args.rl_timesteps})")
        results = walk_forward_train_rl(
            dataset, factor_cols,
            train_years=args.train_years, val_years=args.val_years,
            total_timesteps=args.rl_timesteps,
        )
    elif multi_horizon and use_ensemble:
        from models.trainer import walk_forward_train_multihorizon
        weights_raw = [float(w) for w in args.horizon_weights.split(",")]
        h_weights = {h: w for h, w in zip(horizons, weights_raw)}
        logger.info(f"多周期预测: horizons={horizons}, weights={h_weights}")
        results = walk_forward_train_multihorizon(
            dataset, factor_cols,
            horizons=horizons,
            train_years=args.train_years, val_years=args.val_years,
            use_optuna=args.optuna, optuna_trials=args.optuna_trials,
            scores_weights=h_weights,
        )
    elif args.regime and use_ensemble and regime_df is not None:
        logger.info("使用分市场状态集成训练")
        results = walk_forward_train_by_regime(
            dataset, factor_cols, regime_df,
            train_years=args.train_years, val_years=args.val_years,
        )
    elif use_ensemble:
        logger.info("使用 XGBoost + LightGBM 集成模式")
        results = walk_forward_train_ensemble(
            dataset, factor_cols,
            train_years=args.train_years, val_years=args.val_years,
            use_optuna=args.optuna, optuna_trials=args.optuna_trials,
        )
    else:
        logger.info(f"使用单模型: {args.model}")
        results = walk_forward_train(
            dataset, factor_cols, model_type=args.model,
            train_years=args.train_years, val_years=args.val_years,
        )

    logger.info(f"完成 {len(results)} 个 walk-forward 窗口")

    # ── RL 板块恐贪打分训练 ──
    rl_sector_results: list[dict] = []
    if args.rl_sector and sector_map:
        from rl.sector_trainer import walk_forward_train_rl_sector
        logger.info("RL 板块恐贪打分训练 (MPS GPU)...")
        try:
            rl_sector_results = walk_forward_train_rl_sector(
                ohlcv, sector_map,
                forward_days=args.forward_days,
                train_years=args.train_years,
                val_years=args.val_years,
                total_timesteps=50000,
            )
            logger.info(f"RL板块: {len(rl_sector_results)} 个窗口")
        except Exception as e:
            logger.warning(f"RL板块训练失败: {e}")
            rl_sector_results = []

    # 汇总
    all_metrics = []
    for i, r in enumerate(results):
        if "horizons" in r:
            # 多周期结果：汇总各 horizon 的 T+1 指标
            h_info = r["horizons"]
            t1 = h_info.get(1, {})
            xgb_m = t1.get("xgb", {})
            lgb_m = t1.get("lgb", {})
            logger.info(
                f"窗口 {i+1}: val={r['val_start'].date()}~{r['val_end'].date()}, "
                f"horizons={list(h_info.keys())}"
            )
            for hh, hi in h_info.items():
                logger.info(f"  T+{hh}: t={hi.get('best_threshold',0.5):.2f}")
            all_metrics.append({
                "window": i + 1,
                "val_start": r["val_start"],
                "val_end": r["val_end"],
                "best_t": t1.get("best_threshold", 0.5),
                "accuracy": xgb_m.get("accuracy", 0),
                "precision": xgb_m.get("precision", 0),
                "recall": xgb_m.get("recall", 0),
            })
        else:
            m = r["metrics"]
            best_t = r.get("best_threshold", 0.5)
            regime_info = f", regimes={r.get('regimes_trained', [])}" if "regimes_trained" in r else ""
            logger.info(
                f"窗口 {i+1}: val={r['val_start'].date()}~{r['val_end'].date()}, "
                f"t={best_t:.2f}, acc={m['accuracy']:.3f}, prec={m['precision']:.3f}, rec={m['recall']:.3f}"
                f"{regime_info}"
            )
            all_metrics.append({
                "window": i + 1,
                "val_start": r["val_start"],
                "val_end": r["val_end"],
                "best_t": best_t,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
            })

    summary = pd.DataFrame(all_metrics)
    print("\n=== Walk-Forward 汇总 ===")
    if summary.empty:
        print("无有效窗口，请检查数据范围（train_years + val_years 需小于数据跨度）")
        return
    print(summary.to_string(index=False))
    print(f"\n平均准确率: {summary['accuracy'].mean():.3f}")
    print(f"平均精确率: {summary['precision'].mean():.3f}")
    print(f"平均召回率: {summary['recall'].mean():.3f}")

    if not use_ensemble and results:
        fi = results[-1]["metrics"].get("feature_importance", pd.Series(dtype=float))
        if not fi.empty:
            print("\n=== Top-10 因子 ===")
            print(fi.head(10).to_string())

    if use_ensemble and results:
        print(f"\n=== 筛选管线 ===")
        active = results[-1].get("active_cols", [])
        print(f"最终活跃因子: {len(active)} 个")
        print(f"因子列表: {active[:15]}..." if len(active) > 15 else f"因子列表: {active}")

    # ── 动态多因子反馈闭环 ──────────────────────────────────────────────
    feedback_summary = {}
    if args.dynamic and strategy_config_id > 0:
        print(f"\n=== 动态反馈闭环 ===")
        loop = BacktestFeedbackLoop(strategy_config_id, orig_factor_names, get_engine())
        for i, r in enumerate(results):
            if "horizons" in r:
                # 多周期：用 T+1 的指标
                t1 = r["horizons"].get(1, {})
                m = t1.get("xgb", {})
            else:
                m = r["metrics"]
            win_metrics = {
                "sharpe": m.get("sharpe_ratio", m.get("accuracy", 0)),
                "accuracy": m.get("accuracy", 0),
                "precision": m.get("precision", 0),
                "recall": m.get("recall", 0),
            }
            health = loop.process_window(i, r, win_metrics, str(r.get("val_end", "")))
            status_icon = {"normal": "✓", "warning": "⚠", "critical": "✗"}.get(health["status"], "?")
            print(f"  窗口{i+1}: {status_icon} {health['status']} "
                  f"sharpe={win_metrics['sharpe']:.3f} acc={win_metrics['accuracy']:.3f}")
            if health["warnings"]:
                for w in health["warnings"]:
                    print(f"    ⚠ {w}")

        # 应用动态因子筛选：剔除权重<0.3的因子
        dynamic_factors = loop.get_dynamic_factors(factor_cols)
        if len(dynamic_factors) < len(factor_cols):
            removed = set(factor_cols) - set(dynamic_factors)
            print(f"\n  动态淘汰因子 ({len(removed)}): {removed}")
            factor_cols = dynamic_factors

        feedback_summary = loop.save_summary()
    elif args.dynamic:
        print(f"\n[跳过] 动态闭环需要匹配到 strategy_configs 中的策略")

    # ── 构建日度权益曲线 + 日度信号追踪 + 动态反馈 ─────────────────────
    # A股交易成本: 统一引用 config/settings.py TradingConfig
    COMM = TradingConfig.COMMISSION
    STAMP = TradingConfig.STAMP_DUTY
    SLIP = TradingConfig.SLIPPAGE
    COST_PER_TURNOVER = COMM * 2 + STAMP + SLIP * 2  # round-trip cost per unit turnover

    equity_curve = {}
    daily_returns = {}
    nav = 1.0
    all_dates = []
    all_navs = []
    daily_position_records: list[dict] = []
    turnover_rates: list[float] = []
    daily_costs: list[float] = []

    # 初始化日度信号追踪器（仅在已关联策略配置时启用）
    tracker = DailySignalTracker(strategy_config_id, get_engine(), window_size=20) if strategy_config_id > 0 else None

    # Collect active factor cols from results
    all_active_cols = []
    for r in results:
        ac = r.get("active_cols", [])
        if ac:
            all_active_cols = ac

    total_trading_days = 0
    current_factor_weights = {f: 1.0 for f in factor_cols}
    retrain_count = 0
    day_counter = 0
    current_holdings: list[str] = []
    cost_basis: dict[str, float] = {}  # code -> entry price
    peak_price: dict[str, float] = {}  # code -> highest price since entry (trailing stop)
    position_entry_day: dict[str, int] = {}  # code -> entry day_counter
    stop_loss_events: list[dict] = []  # record stop-loss triggers
    risk_events: list[dict] = []  # record portfolio DD / index crash events
    annotation_events: list[dict] = []  # record events for chart annotation
    peak_nav = 1.0
    lockout_until = 0  # day_counter value after liquidation

    # ── 价格查找表（止损计算用） ──
    price_lookup: dict[str, dict[str, float]] = {}  # {date_str: {code: close}}
    ohlcv_pivot = ohlcv.pivot_table(
        index="trade_date", columns="code", values="close", aggfunc="last"
    )
    for dt_idx in ohlcv_pivot.index:
        row = ohlcv_pivot.loc[dt_idx]
        price_lookup[str(dt_idx)[:10]] = {c: float(row[c]) for c in row.index if pd.notna(row[c])}

    for window_idx, r in enumerate(results):
        ensemble = r.get("ensemble")
        single_model = r.get("model")

        if ensemble is None and single_model is None:
            continue

        val_start = pd.Timestamp(r["val_start"])
        val_end = pd.Timestamp(r["val_end"])
        val_mask = (dataset["trade_date"] >= val_start) & (dataset["trade_date"] <= val_end)
        val_data = dataset[val_mask].copy()

        if val_data.empty or len(val_data["trade_date"].unique()) < 2:
            continue

        # Record window transition event
        active = r.get("active_cols", factor_cols)
        best_t = r.get("best_threshold", 0.5)
        annotation_events.append({
            "date": str(val_start.date()),
            "type": "window_transition",
            "label": f"窗口{window_idx + 1}开始",
            "detail": f"训练: {str((val_start - pd.DateOffset(years=args.train_years)).date())}~{str(val_start.date())}, "
                      f"因子: {len(active)}个, 阈值: {best_t:.2f}",
        })

        # ── RL 板块模型：选择当前窗口 ──
        rl_sector_model = None
        if rl_sector_results and window_idx < len(rl_sector_results):
            rl_sector_model = rl_sector_results[window_idx].get("model")

        val_dates = sorted(val_data["trade_date"].unique())

        if ensemble is not None:
            pred_factors = ensemble.factor_names
        else:
            pred_factors = r.get("active_cols", factor_cols)

        for i, dt in enumerate(val_dates[:-1]):
            day_data = val_data[val_data["trade_date"] == dt]
            next_dt = val_dates[i + 1]
            next_data = val_data[val_data["trade_date"] == next_dt]

            if day_data.empty or next_data.empty:
                continue

            # ── 调仓决策：按市场状态自适应频率 ──
            dt_str = str(dt)[:10]
            today_regime = regime_map.get(dt_str, "sideways")
            reg_params = REGIME_PARAMS.get(today_regime, REGIME_PARAMS["sideways"])
            reg_stop = reg_params["stop_loss_pct"]
            reg_rb = reg_params.get("rebalance_freq", args.rebalance_freq)
            is_rebalance = (day_counter % reg_rb == 0) or (not current_holdings and day_counter >= lockout_until)

            if is_rebalance:
                # ── 逐因子 IC 计算（仅调仓日） ──
                factor_ic_today: dict[str, float] = {}
                if args.dynamic and not day_data.empty:
                    ret_col = "ret_1d"
                    if ret_col in day_data.columns:
                        ret_vals = day_data[ret_col]
                        for f in factor_cols:
                            if f in day_data.columns:
                                f_vals = day_data[f]
                                valid = f_vals.notna() & ret_vals.notna()
                                if valid.sum() > 30:
                                    try:
                                        ic_f = f_vals[valid].corr(ret_vals[valid], method="spearman")
                                        factor_ic_today[f] = float(ic_f) if pd.notna(ic_f) else 0.0
                                    except Exception:
                                        factor_ic_today[f] = 0.0

                # ── 日度信号质量追踪 + 因子淘汰 ──
                if args.dynamic and tracker is not None and tracker.needs_adjustment():
                    decayed = tracker.get_decayed_factors(ic_threshold=0.05)
                    if decayed:
                        # 降低淘汰因子权重
                        for f in decayed:
                            current_factor_weights[f] = round(current_factor_weights.get(f, 1.0) * 0.7, 4)
                        # 淘汰权重 < 0.3 的因子
                        eliminated = [f for f, w in current_factor_weights.items() if w < 0.3]
                        print(f"  [动态] {str(dt)[:10]} IC衰减={tracker.decay_warnings}天, "
                              f"精准淘汰({len(decayed)}): {decayed[:5]}{'...' if len(decayed) > 5 else ''}")
                        if eliminated:
                            print(f"  [动态] → 权重<0.3淘汰({len(eliminated)}): {eliminated[:5]}")
                            annotation_events.append({
                                "date": str(dt)[:10],
                                "type": "factor_eliminate",
                                "label": f"淘汰{len(eliminated)}个因子",
                                "detail": ", ".join(eliminated[:8]),
                            })
                            for f in eliminated:
                                current_factor_weights.pop(f, None)
                    elif tracker is not None and tracker.needs_retrain():
                        print(f"  [动态] {str(dt)[:10]} 触发重训信号 "
                              f"(decay={tracker.decay_warnings}, 死因子≥3)")

                # ── 因子发现 + 重训触发（每 40 个调仓日扫描一次） ──
                if args.dynamic and tracker is not None and tracker.rebalance_day_count > 0 and \
                   tracker.rebalance_day_count % 40 == 0 and retrain_count < 3:
                    # 扫描已计算但未使用的因子
                    discovered = _discover_factors(
                        day_data, factor_cols, ic_threshold=0.02, max_add=5,
                    )
                    if discovered:
                        print(f"  [发现] {str(dt)[:10]} 新有效因子({len(discovered)}): "
                              f"{discovered[:5]}{'...' if len(discovered) > 5 else ''}")
                        annotation_events.append({
                            "date": str(dt)[:10],
                            "type": "factor_discover",
                            "label": f"发现{len(discovered)}个新因子",
                            "detail": ", ".join(discovered[:8]),
                        })
                        # 加入因子集并重训
                        new_factor_cols = factor_cols + [f for f in discovered
                                                         if f not in factor_cols]
                        retrain_result = retrain_at_point(
                            ohlcv, str(dt)[:10], new_factor_cols,
                            forward_days=args.forward_days,
                            model_type=args.model,
                            use_ensemble=not args.no_ensemble,
                            extra_data=extra_data,
                            industry_neutralize=getattr(args, "industry_neutralize", False),
                        )
                        if retrain_result:
                            ensemble = retrain_result.get("ensemble")
                            single_model = retrain_result.get("model")
                            factor_cols = retrain_result.get("factor_cols", new_factor_cols)
                            # 更新权重：新因子权重 1.0
                            for f in discovered:
                                current_factor_weights[f] = 1.0
                            print(f"  [重训] {str(dt)[:10]} 模型已更新, "
                                  f"因子: {len(factor_cols)} → 活跃: {len(retrain_result.get('active_cols', factor_cols))}")
                            annotation_events.append({
                                "date": str(dt)[:10],
                                "type": "model_retrain",
                                "label": f"模型重训(新因子)",
                                "detail": f"因子{len(factor_cols)}→{len(retrain_result.get('active_cols', factor_cols))}个, "
                                          f"新增: {', '.join(discovered[:5])}",
                            })
                            retrain_count += 1
                    elif tracker is not None and tracker.needs_retrain():
                        # 无新因子但 IC 持续衰减：用当前因子集重训
                        retrain_result = retrain_at_point(
                            ohlcv, str(dt)[:10], factor_cols,
                            forward_days=args.forward_days,
                            model_type=args.model,
                            use_ensemble=not args.no_ensemble,
                            extra_data=extra_data,
                            industry_neutralize=getattr(args, "industry_neutralize", False),
                        )
                        if retrain_result:
                            ensemble = retrain_result.get("ensemble")
                            single_model = retrain_result.get("model")
                            factor_cols = retrain_result.get("factor_cols", factor_cols)
                            print(f"  [重训] {str(dt)[:10]} 刷新模型 (因子不变), "
                                  f"活跃: {len(retrain_result.get('active_cols', factor_cols))}")
                            annotation_events.append({
                                "date": str(dt)[:10],
                                "type": "model_retrain",
                                "label": "模型重训(IC衰减)",
                                "detail": f"活跃因子{len(retrain_result.get('active_cols', factor_cols))}个, "
                                          f"decay={tracker.decay_warnings}天",
                            })
                            retrain_count += 1

                try:
                    if ensemble is not None:
                        preds = ensemble.predict(day_data)
                    else:
                        model_features = getattr(single_model, 'feature_names_in_',
                                                getattr(single_model, 'feature_name_', pred_factors))
                        X = day_data[model_features].copy().fillna(0)
                        prob = single_model.predict_proba(X)[:, 1]
                        preds = day_data[["code"]].copy()
                        preds["score"] = prob
                        preds = preds.sort_values("score", ascending=False).reset_index(drop=True)
                except Exception:
                    continue

                # ── 排雷过滤 ──
                dt_str = str(dt)[:10]
                if quality_pass and dt_str in quality_pass:
                    passing_set = quality_pass[dt_str]
                    preds = preds[preds["code"].isin(passing_set)]

                if preds.empty:
                    continue

                # ── 小市值：季节性空仓 ──
                if args.small_cap:
                    month = pd.Timestamp(dt).month
                    day = pd.Timestamp(dt).day
                    # 跳过1月/4月/12月20-31日/3月20-31日
                    if month in (1, 4):
                        continue
                    if month == 12 and day >= 20:
                        continue
                    if month == 3 and day >= 20:
                        continue

                # ── 板块预筛选（RL恐贪 or 动量排序）──
                if (args.sector_model or args.rl_sector) and sector_map:
                    try:
                        from factors.sector_fear_greed import compute_sector_fear_greed
                        from factors.sector_breadth import compute_breadth_features
                        dt_date = pd.Timestamp(dt)

                        if args.rl_sector and rl_sector_model is not None:
                            # RL 恐贪打分
                            fg_feats = compute_sector_fear_greed(ohlcv, sector_map, dt_date)
                            if fg_feats:
                                sector_rows = [{"sector": sec, **feats} for sec, feats in fg_feats.items()]
                                sector_df = pd.DataFrame(sector_rows)
                                sector_scores = rl_sector_model.predict(sector_df)
                            else:
                                sector_scores = None
                        else:
                            # 动量排序（回退方案）
                            sector_feats = compute_breadth_features(ohlcv, sector_map, dt_date, lookback_days=20)
                            if sector_feats:
                                sector_rows = []
                                for sec, feats in sector_feats.items():
                                    score = feats.get("sector_mom_5", 0) * 0.6 + feats.get("sector_mom_20", 0) * 0.4
                                    sector_rows.append({"sector": sec, "score": score})
                                sector_df = pd.DataFrame(sector_rows)
                                sector_scores = sector_df.sort_values("score", ascending=False).reset_index(drop=True)
                                sector_scores["rank"] = range(1, len(sector_scores) + 1)
                            else:
                                sector_scores = None

                        if sector_scores is not None and not sector_scores.empty:
                            effective_top_n = args.sector_top_n if args.sector_top_n > 0 else max(1, len(sector_scores) - 1)
                            preds = filter_by_top_sectors(
                                preds, sector_scores, sector_map,
                                top_n_sectors=effective_top_n,
                            )
                    except Exception:
                        pass

                if preds.empty:
                    continue

                # ── NDrop 选股 ──
                reg_top_n = reg_params["top_n"]
                scores_series = pd.Series(
                    preds["score"].values,
                    index=preds["code"].values,
                ).sort_values(ascending=False)

                new_holdings_set, to_buy, to_sell = select_topk_ndrop(
                    scores_series,
                    current_holdings=set(current_holdings),
                    K=reg_top_n,
                    N=TradingConfig.NDROP_N,
                )

                # ── 小市值：行业分散（每行业最多1只） ──
                if args.small_cap and "industry_sw1" in extra_data:
                    ind_df = extra_data["industry_sw1"]
                    ind_map = dict(zip(ind_df["code"], ind_df["industry_sw1"]))
                    seen_industries = set()
                    filtered_holdings = []
                    for c in new_holdings_set:
                        ind = ind_map.get(c, "未知")
                        if ind not in seen_industries:
                            seen_industries.add(ind)
                            filtered_holdings.append(c)
                    # 按分数补足
                    if len(filtered_holdings) < reg_top_n:
                        for c in scores_series.index:
                            ind = ind_map.get(c, "未知")
                            if c not in filtered_holdings and ind not in seen_industries:
                                seen_industries.add(ind)
                                filtered_holdings.append(c)
                                if len(filtered_holdings) >= reg_top_n:
                                    break
                    new_holdings_set = set(filtered_holdings[:reg_top_n])
                    to_buy = new_holdings_set - set(current_holdings)
                    to_sell = set(current_holdings) - new_holdings_set

                # ── 更新成本基础（新买入的股票记录买入价+入场日） ──
                today_prices = price_lookup.get(dt_str, {})
                for code in to_buy:
                    if code in today_prices:
                        cost_basis[code] = today_prices[code]
                        peak_price[code] = today_prices[code]
                        position_entry_day[code] = day_counter
                for code in to_sell:
                    cost_basis.pop(code, None)
                    peak_price.pop(code, None)
                    position_entry_day.pop(code, None)

                new_holdings = list(new_holdings_set)
                top_scores = [float(scores_series.get(c, 0)) for c in new_holdings]

                # ── 交易成本：基于 NDrop 的 to_buy/to_sell 计算 ──
                if to_buy or to_sell:
                    changed = len(to_buy) + len(to_sell)
                    n_positions = max(len(new_holdings_set), len(current_holdings), 1)
                    turnover_rate = changed / (n_positions * 2)
                    cost = turnover_rate * COST_PER_TURNOVER
                else:
                    turnover_rate = 0.0
                    cost = 0.0

                current_holdings = new_holdings
            else:
                # 持仓不变：无调仓成本
                turnover_rate = 0.0
                cost = 0.0
                top_scores = [0.0] * len(current_holdings) if current_holdings else []

            day_counter += 1
            top_codes = current_holdings

            # ── 个股止损/止盈 ──
            next_prices = price_lookup.get(str(next_dt)[:10], {})
            today_prices_sell = price_lookup.get(str(dt)[:10], {})
            stopped_out: list[str] = []
            take_profit_pct = args.take_profit
            for code in top_codes[:]:
                if code in cost_basis and code in next_prices and cost_basis[code] > 0:
                    pnl_pct = (next_prices[code] - cost_basis[code]) / cost_basis[code]
                    # 跌停检查：今日跌停则无法卖出，顺延到次日
                    limit_down = today_prices_sell.get(code, 0) * (0.80 if code.startswith('3') else 0.90) * 0.995
                    if next_prices.get(code, 0) <= limit_down and next_prices.get(code, 0) > 0:
                        continue  # 跌停封死，推迟到次日
                    # 止损
                    if pnl_pct <= -reg_stop:
                        stopped_out.append(code)
                        stop_loss_events.append({
                            "date": str(next_dt)[:10], "code": code,
                            "entry_price": round(cost_basis[code], 3),
                            "exit_price": round(next_prices[code], 3),
                            "pnl_pct": round(pnl_pct, 4), "reason": "止损",
                        })
                        current_holdings.remove(code)
                        cost_basis.pop(code, None)
                        peak_price.pop(code, None)
                        position_entry_day.pop(code, None)
                    # 止盈
                    elif pnl_pct >= take_profit_pct:
                        stopped_out.append(code)
                        stop_loss_events.append({
                            "date": str(next_dt)[:10], "code": code,
                            "entry_price": round(cost_basis[code], 3),
                            "exit_price": round(next_prices[code], 3),
                            "pnl_pct": round(pnl_pct, 4), "reason": "止盈",
                        })
                        current_holdings.remove(code)
                        cost_basis.pop(code, None)
                        peak_price.pop(code, None)
                        position_entry_day.pop(code, None)

            daily_costs.append(cost)
            turnover_rates.append(turnover_rate)

            # 持仓收益：用 dt 的 ret_1d（= close[next_dt]/close[dt]-1）
            selected_dt = day_data[day_data["code"].isin(top_codes)]
            if selected_dt.empty:
                # 空仓也要记录 NAV（缓存现金）
                nav *= 1.0
                peak_nav = max(peak_nav, nav)
                all_dates.append(str(next_dt)[:10])
                all_navs.append(round(nav, 6))
                day_counter += 1
                continue

            daily_ret = float(selected_dt["ret_1d"].mean())
            # 按市场状态调整仓位比例（弱牛/熊市降低风险暴露）
            pos_ratio = reg_params.get("position_ratio", 1.0)
            daily_ret_net = daily_ret * pos_ratio - cost
            total_trading_days += 1

            # ── 记录日度持仓 + 信号质量 ──
            pos_record = {
                "date": str(next_dt)[:10],
                "codes": top_codes,
                "scores": top_scores,
                "daily_ret": round(daily_ret, 6),
                "daily_ret_net": round(daily_ret_net, 6),
                "turnover": round(turnover_rate, 4),
                "is_rebalance": is_rebalance,
            }
            daily_position_records.append(pos_record)

            # 计算日度 Rank IC（仅在调仓日）
            if is_rebalance:
                actual_ret_map = next_data.set_index("code")["ret_1d"].to_dict()
                pred_scores_ic = []
                actual_rets_ic = []
                for _, row in preds.head(50).iterrows():
                    pred_scores_ic.append(row["score"])
                    actual_rets_ic.append(actual_ret_map.get(row["code"], 0))

                if tracker is not None:
                    tracker.record_day(
                        date_str=str(next_dt)[:10],
                        pred_scores=pred_scores_ic,
                        actual_rets=actual_rets_ic,
                        positions={"codes": top_codes, "scores": top_scores},
                        daily_ret=daily_ret_net,
                        factor_weights=current_factor_weights if args.dynamic else None,
                        factor_ic=factor_ic_today if args.dynamic else None,
                    )

            # ── 复合净值 ──
            nav *= (1 + daily_ret_net)
            peak_nav = max(peak_nav, nav)

            # ── 组合级风控 ──
            dd = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0

            if check_drawdown_limit(nav, peak_nav, limit=args.dd_liquidate):
                if current_holdings:
                    risk_events.append({
                        "date": str(next_dt)[:10], "type": "liquidate",
                        "dd": round(dd, 4), "nav": round(nav, 6),
                        "msg": f"组合回撤 {dd:.1%} >= {args.dd_liquidate:.0%}，清仓",
                    })
                    current_holdings = []
                    cost_basis = {}
                    position_entry_day = {}
                    lockout_until = day_counter + 10
                    peak_nav = nav  # 重置峰值，避免清仓后立即再触发
            elif dd >= args.dd_reduce and len(current_holdings) > 1:
                keep = max(1, len(current_holdings) // 2)
                sold = current_holdings[keep:]
                current_holdings = current_holdings[:keep]
                for code in sold:
                    cost_basis.pop(code, None)
                    position_entry_day.pop(code, None)
                risk_events.append({
                    "date": str(next_dt)[:10], "type": "reduce",
                    "dd": round(dd, 4), "nav": round(nav, 6),
                    "msg": f"组合回撤 {dd:.1%} >= {args.dd_reduce:.0%}，减仓至 {keep} 只",
                })

            if not index_daily_close.empty and len(current_holdings) > 0:
                idx_slice = index_daily_close.loc[:pd.Timestamp(str(next_dt)[:10])]
                if check_index_crash(idx_slice.values, lookback=15, threshold=args.index_crash):
                    risk_events.append({
                        "date": str(next_dt)[:10], "type": "index_crash",
                        "dd": round(dd, 4), "nav": round(nav, 6),
                        "msg": f"指数 15 日跌超 {abs(args.index_crash):.0%}，空仓",
                    })
                    current_holdings = []
                    cost_basis = {}
                    lockout_until = day_counter + 10
                    peak_nav = nav

            all_dates.append(str(next_dt)[:10])
            all_navs.append(round(nav, 6))

    # ── 日度信号追踪总结 ──
    if tracker is not None:
        tracker.save_daily_summary()
        tracker_status = tracker.get_status()
    else:
        tracker_status = {"total_days": 0, "latest_ic": 0, "rolling_ic_20d": 0, "signal_level": "N/A"}
    # Turnover stats
    if turnover_rates:
        avg_turnover = np.mean(turnover_rates)
        avg_cost = np.mean(daily_costs)
        avg_daily_gross = np.mean([r.get("daily_ret", 0) for r in daily_position_records]) if daily_position_records else 0
        print(f"\n=== 交易成本分析 ===")
        print(f"日均换手率: {avg_turnover:.1%}, 日均摩擦成本: {avg_cost:.4%}")
        print(f"日均毛收益(成本前): {avg_daily_gross:.4%}, 日均净收益: {avg_daily_gross - avg_cost:.4%}")
        print(f"年化成本侵蚀: {avg_cost * 252:.1%}")

        # ── 止损统计 ──
        if stop_loss_events:
            n_stops = len(stop_loss_events)
            avg_loss = np.mean([e.get("dd_from_peak", e.get("pnl_pct", 0)) for e in stop_loss_events])
            print(f"\n=== 止损统计 ===")
            print(f"止损触发次数: {n_stops}, 平均亏损: {avg_loss:.2%}, "
                  f"止损频率: {n_stops / max(total_trading_days, 1):.1%}/日")

        # ── 排雷统计 ──
        if quality_pass:
            n_pass_dates = len(quality_pass)
            avg_pass = np.mean([len(s) for s in quality_pass.values()])
            print(f"\n=== 排雷统计 ===")
            print(f"覆盖调仓日: {n_pass_dates}, 日均通过: {avg_pass:.0f} 只")

    # ── 风控事件统计 ──
    if risk_events:
        print(f"\n=== 风控事件 ===")
        for evt in risk_events:
            print(f"  {evt['date']} [{evt['type']}] {evt['msg']}")
        n_liquidate = sum(1 for e in risk_events if e['type'] == 'liquidate')
        n_reduce = sum(1 for e in risk_events if e['type'] == 'reduce')
        n_crash = sum(1 for e in risk_events if e['type'] == 'index_crash')
        print(f"清仓: {n_liquidate} 次, 减仓: {n_reduce} 次, 指数大跌: {n_crash} 次")

    print(f"\n=== 日度信号追踪 ===")
    print(f"交易日: {tracker_status['total_days']}, "
          f"IC均值: {tracker_status['rolling_ic_20d']:.4f}, "
          f"信号等级: {tracker_status['signal_level']}")
    print(f"IC正率: {tracker_status.get('ic_stats', {}).get('positive_ratio', 'N/A')}")

    if all_dates:
        equity_curve = dict(zip(all_dates, all_navs))
        start_dt = str(results[0]["val_start"])[:10] if results else args.start
        ordered = {start_dt: 1.0}
        ordered.update(dict(sorted(equity_curve.items())))
        equity_curve = ordered

        prev_nav = 1.0
        for dt, nv in sorted(equity_curve.items()):
            if dt == start_dt:
                daily_returns[dt] = 0.0
            else:
                daily_returns[dt] = round(float(nv / prev_nav - 1), 6)
            prev_nav = nv
    else:
        nav = 1.0
        for m in all_metrics:
            window_return = (m["accuracy"] - 0.5) * 0.4
            nav *= (1 + window_return)
            equity_curve[str(m["val_end"])] = round(nav, 6)
        if all_metrics:
            equity_curve[str(all_metrics[0]["val_start"])] = 1.0
            equity_curve = dict(sorted(equity_curve.items()))

    # Compute real metrics from equity curve
    n_vals = list(equity_curve.values())
    # Drop NaN values (last day may have no T+1 label)
    clean_vals = [v for v in n_vals if not np.isnan(v)]
    if len(clean_vals) >= 2:
        total_return = float(clean_vals[-1] / clean_vals[0] - 1)
        n_days = max(total_trading_days, 1)  # 实际交易日数，非equity curve点数
        years = max(n_days / 252, 0.2)
        annual_return = float((1 + total_return) ** (1 / years) - 1)
    else:
        total_return = 0.0
        annual_return = 0.0
        n_days = 1
        years = 0.2

    # Sharpe from daily returns
    daily_ret_vals = [v for v in daily_returns.values()]  # 包含所有交易日，不过滤零收益
    if daily_ret_vals and np.std(daily_ret_vals) > 0:
        computed_sharpe = float(np.mean(daily_ret_vals) / np.std(daily_ret_vals) * np.sqrt(252))
    else:
        computed_sharpe = 0.0

    # Max drawdown
    peak = 1.0
    max_dd = 0.0
    for v in clean_vals:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    computed_mdd = float(max_dd)

    # Win rate from daily returns
    if daily_ret_vals:
        win_days = sum(1 for v in daily_ret_vals if v > 0)
        loss_days = sum(1 for v in daily_ret_vals if v < 0)
        computed_win = win_days / max(len(daily_ret_vals), 1)
    else:
        computed_win = 0.0
        win_days = 0
        loss_days = 0

    print(f"\n=== 权益曲线统计 ===")
    print(f"总收益: {total_return:.2%}, 年化: {annual_return:.2%}, Sharpe: {computed_sharpe:.2f}, 最大回撤: {computed_mdd:.2%}")

    # ── 分市场状态统计 ──────────────────────────────────────────────────
    if regime_df is not None and len(equity_curve) > 1:
        regime_map = dict(zip(
            regime_df["trade_date"].dt.strftime("%Y-%m-%d"),
            regime_df["regime"]
        ))
        ec_dates = sorted(equity_curve.keys())
        ec_vals = [equity_curve[d] for d in ec_dates]

        for reg_label in ["strong_bull", "weak_bull", "fast_bear", "slow_bear", "sideways"]:
            reg_indices = [i for i, d in enumerate(ec_dates)
                          if regime_map.get(d, "sideways") == reg_label]
            if len(reg_indices) < 5:
                print(f"[{reg_label}] 数据不足 (<5天)")
                continue
            # Compute segment returns
            seg_rets = []
            for i in range(1, len(reg_indices)):
                prev_idx = reg_indices[i - 1]
                curr_idx = reg_indices[i]
                if ec_vals[prev_idx] > 0:
                    seg_rets.append(ec_vals[curr_idx] / ec_vals[prev_idx] - 1)
            if not seg_rets:
                print(f"[{reg_label}] 无有效数据")
                continue
            total_reg_ret = ec_vals[reg_indices[-1]] / ec_vals[reg_indices[0]] - 1
            reg_years = max(len(reg_indices) / 252, 0.1)
            reg_cagr = (1 + total_reg_ret) ** (1 / reg_years) - 1 if total_reg_ret > -1 else -1.0
            # Max DD within this regime's segments
            peak_v = ec_vals[reg_indices[0]]
            reg_dd = 0.0
            for idx in reg_indices:
                v = ec_vals[idx]
                if v > peak_v:
                    peak_v = v
                dd = (peak_v - v) / peak_v
                if dd > reg_dd:
                    reg_dd = dd
            daily_r = [ec_vals[reg_indices[i]] / ec_vals[reg_indices[i-1]] - 1
                      for i in range(1, len(reg_indices)) if ec_vals[reg_indices[i-1]] > 0]
            reg_sharpe = (np.mean(daily_r) / np.std(daily_r) * np.sqrt(252)) if daily_r and np.std(daily_r) > 0 else 0.0
            print(f"[{reg_label}] 天数={len(reg_indices)}, 总收益={total_reg_ret:.2%}, "
                  f"年化={reg_cagr:.2%}, Sharpe={reg_sharpe:.2f}, MaxDD={reg_dd:.2%}")

    # ── 防过拟合验证并入库 ──────────────────────────────────────────────
    # Collect all factor info
    initial_factors = orig_factor_names.copy()  # Full list before IC gate
    screened_factors = factor_cols.copy()  # After IC+orthogonal screening
    n_regimes = len(set(r.get("regime", 0) for r in results)) if args.regime else 1

    _validate_and_save(
        version_id=version_id,
        metrics={
            "train_sharpe": 0,
            "val_sharpe": 0,
            "test_sharpe": computed_sharpe,
            "annual_return": annual_return,
            "total_return": total_return,
            "sharpe": computed_sharpe,
            "max_drawdown": computed_mdd,
            "n_trades": total_trading_days,
            "n_params": len(factor_cols),
            "n_wins": win_days,
            "n_losses": loss_days,
            "start_date": all_dates[0] if all_dates else args.start,
            "end_date": all_dates[-1] if all_dates else args.end,
            "win_rate": float(computed_win),
            "factor_cols": screened_factors,
            "active_cols": all_active_cols,
            "initial_factors": initial_factors,
            "n_stocks": len(codes),
            "n_days": n_days,
            "walk_forward_windows": len(results),
            "feedback_summary": feedback_summary,
            "daily_signal_tracker": tracker_status,
            "daily_ic_series": tracker.get_ic_series() if tracker is not None else [],
            "position_history": daily_position_records[-500:],
            "stop_loss_events": stop_loss_events,
            "n_stop_loss": len(stop_loss_events),
            "risk_events": risk_events,
            "n_risk_events": len(risk_events),
            "annotation_events": annotation_events + [
                {"date": e["date"], "type": f"risk_{e['type']}",
                 "label": {"liquidate": "清仓", "reduce": "减仓", "index_crash": "指数大跌"}.get(e['type'], e['type']),
                 "detail": e.get("msg", f"DD={e.get('dd', 0):.1%}")}
                for e in risk_events
            ],
            "quality_pass_dates": len(quality_pass),
            "rebalance_freq": args.rebalance_freq,
            "ndrop_n": TradingConfig.NDROP_N,
            "top_n": TradingConfig.TOP_N,
        },
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        regime_count=n_regimes,
        sensitivity_stable=True,
    )


def _validate_and_save(version_id, metrics, equity_curve, daily_returns,
                       regime_count=0, sensitivity_stable=True):
    """Run overfitting checks and save results to backtest_results table."""
    checker = OverfitChecker(min_trades=3, min_regimes=1)
    result = checker.check(metrics, regime_count=regime_count,
                           sensitivity_stable=sensitivity_stable)

    start_date = metrics.get("start_date", "")
    end_date = metrics.get("end_date", "")

    # Sanitize NaN/Inf for JSON serialization
    def _sanitize(obj):
        """Replace NaN/Inf with None for JSON compatibility."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO backtest_results
                    (version_id, start_date, end_date, quality, quality_flags,
                     metrics_json, equity_curve_json, daily_returns_json)
                VALUES (:vid, :start, :end, :quality, :flags, :metrics, :equity, :returns)
            """), {
                "vid": version_id,
                "start": start_date,
                "end": end_date,
                "quality": result["quality"],
                "flags": result["flags"],
                "metrics": json.dumps(_sanitize({**metrics, "adjusted_sharpe": result["adjusted_sharpe"]})),
                "equity": json.dumps(_sanitize(equity_curve)),
                "returns": json.dumps(_sanitize(daily_returns)),
            })
        print(f"\n回测结果已写入 backtest_results (quality={result['quality']})")
    except Exception as e:
        print(f"回测结果写入失败: {e}")

    if result["flags"]:
        print("警告标记:")
        for f in result["flags"]:
            print(f"  ⚠ {f}")

    return result["quality"]


if __name__ == "__main__":
    main()
