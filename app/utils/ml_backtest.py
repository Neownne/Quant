"""ML 策略 Walk-Forward 回测引擎。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from data.db import get_engine
from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train_ensemble
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from portfolio.selector import select_top_n, select_topk_ndrop


def _get_daily_ohlcv(codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """从 stock_daily 加载日线 OHLCV。"""
    engine = get_engine()
    code_list = ",".join([f"'{c}'" for c in codes])
    try:
        df = pd.read_sql(
            f"SELECT code, trade_date, open, high, low, close, volume, amount, turnover "
            f"FROM stock_daily WHERE code IN ({code_list}) "
            f"AND trade_date BETWEEN '{start_date}' AND '{end_date}' "
            f"ORDER BY code, trade_date",
            engine,
        )
    finally:
        engine.dispose()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _get_minute_ohlcv(codes: list[str], start_date: str, end_date: str, period: str = "60") -> pd.DataFrame:
    """从 stock_minute 加载分钟 K 线。返回列包含 trade_date（date 类型）。"""
    engine = get_engine()
    code_list = ",".join([f"'{c}'" for c in codes])
    try:
        df = pd.read_sql(
            f"SELECT code, trade_time, trade_time::date AS trade_date, "
            f"open, high, low, close, volume, amount "
            f"FROM stock_minute WHERE code IN ({code_list}) "
            f"AND trade_time >= '{start_date}' AND trade_time < '{end_date}235959' "
            f"AND period = '{period}' "
            f"ORDER BY code, trade_time",
            engine,
        )
    finally:
        engine.dispose()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["turnover"] = None
    return df


def load_price_data(codes: list[str], start_date: str, end_date: str,
                    freq: str = "daily") -> pd.DataFrame:
    """统一价格数据加载入口。

    参数
    ----
    freq : "daily" | "60min"
    """
    if freq == "daily":
        return _get_daily_ohlcv(codes, start_date, end_date)
    elif freq == "60min":
        return _get_minute_ohlcv(codes, start_date, end_date, period="60")
    else:
        raise ValueError(f"不支持的频率: {freq}")


def run_ml_backtest(
    config: dict,
    codes: list[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 1_000_000,
    progress_callback=None,
) -> dict:
    """运行 ML 策略 Walk-Forward 回测。

    config: ML 策略配置字典 (from ml_strategy_config)
    codes: 回测股票代码列表
    start_date/end_date: YYYYMMDD
    initial_cash: 初始资金
    progress_callback: Optional[Callable[[str, float], None]]

    返回: {"metrics": {...}, "equity_curve": DataFrame, "trades": DataFrame, "results_json": {...}}
    """
    # 1. 确定因子
    factor_names_cfg = config.get("factor_names") or []
    if not factor_names_cfg:
        factor_names = [f for f in ALL_FACTORS if not f.startswith("fin_")]
    else:
        factor_names = [f for f in factor_names_cfg if f in ALL_FACTORS]

    if not factor_names:
        return {"error": "无可用因子"}

    # 2. 加载 OHLCV
    if progress_callback:
        progress_callback("加载OHLCV数据...", 0.05)

    freq = config.get("freq", "daily")
    ohlcv = load_price_data(codes, start_date, end_date, freq=freq)

    if ohlcv.empty:
        return {"error": "无OHLCV数据"}

    # 3. 构建因子数据集
    if progress_callback:
        progress_callback("计算因子...", 0.10)

    bar_per_day = 4 if freq == "60min" else 1
    try:
        dataset = build_factor_dataset(ohlcv, factor_names,
                                        label_mode=config.get("label_mode", "binary"),
                                        bar_per_day=bar_per_day)
    except Exception as e:
        return {"error": f"因子计算失败: {e}"}

    # 4. 因子筛选
    ic_threshold = float(config.get("ic_threshold", 0.02))
    t_threshold = float(config.get("t_threshold", 2.0))
    ortho_threshold = float(config.get("orthogonal_threshold", 0.7))

    if progress_callback:
        progress_callback("IC筛选...", 0.20)

    pv_names = [f for f in factor_names if not f.startswith("fin_")]
    try:
        filtered = filter_factors_by_ic(dataset, pv_names, ret_col="ret_1d",
                                         ic_threshold=ic_threshold, t_threshold=t_threshold)
    except Exception as e:
        return {"error": f"IC筛选失败: {e}"}

    if progress_callback:
        progress_callback("正交筛选...", 0.25)

    selected = select_orthogonal_factors(dataset, filtered, threshold=ortho_threshold)

    if not selected:
        return {"error": "因子筛选后无可用因子"}

    # 5. 训练模型
    if progress_callback:
        progress_callback("训练模型...", 0.30)

    train_years = int(config.get("train_years", 3))
    val_years = int(config.get("val_years", 1))

    try:
        results = walk_forward_train_ensemble(
            dataset, selected, train_years=train_years, val_years=val_years,
        )
    except Exception as e:
        return {"error": f"模型训练失败: {e}"}

    if not results:
        return {"error": "模型训练无结果"}

    # 6. 模拟交易
    if progress_callback:
        progress_callback("模拟交易...", 0.50)

    top_n = int(config.get("top_n", 15))
    rebalance_mode = config.get("rebalance_mode", "ndrop")
    ndrop_n = int(config.get("ndrop_n", 2))
    stop_loss_pct = float(config.get("stop_loss_pct", 0.08))
    max_dd_limit = float(config.get("max_dd_limit", 0.25))

    all_dates = sorted(dataset["trade_date"].unique())
    if len(all_dates) < 20:
        return {"error": "交易日不足20天"}

    cash = float(initial_cash)
    positions: dict[str, dict] = {}
    equity_curve: list[dict] = []
    all_trades: list[dict] = []
    peak_value = float(initial_cash)
    current_holdings: set[str] = set()
    n_windows = len(results)

    for wi, wr in enumerate(results):
        ensemble = wr["ensemble"]
        val_start = wr.get("val_start")
        val_end = wr.get("val_end")

        window_dates = [d for d in all_dates if val_start <= d <= val_end]
        if not window_dates:
            continue

        for di, trade_date in enumerate(window_dates):
            day_factors = dataset[dataset["trade_date"] == trade_date].dropna(subset=selected)
            if day_factors.empty:
                continue

            try:
                pred_df = ensemble.predict(day_factors[["code"] + selected])
                scores = pred_df[["code", "score", "rank"]].copy()
            except Exception:
                continue

            # 选股
            if rebalance_mode == "ndrop" and current_holdings:
                score_series = pd.Series(
                    scores.set_index("code")["score"].to_dict()
                ).sort_values(ascending=False)
                try:
                    new_holdings, to_buy, to_sell = select_topk_ndrop(
                        score_series, current_holdings=current_holdings,
                        K=top_n, N=ndrop_n,
                    )
                    selected_codes = list(new_holdings)
                    sell_codes_list = list(to_sell)
                    buy_codes_list = list(to_buy)
                except Exception:
                    continue
            else:
                selected_df = select_top_n(scores, n=top_n)
                selected_codes = selected_df["code"].tolist() if not selected_df.empty else []
                sell_codes_list = [c for c in current_holdings if c not in set(selected_codes)]
                buy_codes_list = [c for c in selected_codes if c not in current_holdings]

            day_ohlcv = ohlcv[ohlcv["trade_date"] == trade_date]
            price_map = dict(zip(day_ohlcv["code"], day_ohlcv["close"]))

            # 卖出
            for code in sell_codes_list:
                if code in positions and code in price_map:
                    sell_price = float(price_map[code])
                    pos = positions[code]
                    pnl = (sell_price - pos["cost_basis"]) * pos["shares"]
                    cash += sell_price * pos["shares"]
                    all_trades.append({
                        "code": code, "direction": "SELL",
                        "date": trade_date, "price": sell_price,
                        "shares": pos["shares"], "pnl": pnl,
                    })
                    del positions[code]

            # 买入
            n_buy = len(buy_codes_list)
            if n_buy > 0:
                cash_per_stock = cash * 0.95 / n_buy
                for code in buy_codes_list:
                    if code in price_map and code not in positions:
                        price = float(price_map[code])
                        shares = int(cash_per_stock / price / 100) * 100
                        if shares >= 100:
                            cost = shares * price
                            if cost <= cash:
                                cash -= cost
                                positions[code] = {"shares": shares, "cost_basis": price}
                                all_trades.append({
                                    "code": code, "direction": "BUY",
                                    "date": trade_date, "price": price,
                                    "shares": shares, "pnl": 0.0,
                                })

            # 止损
            for code in list(positions.keys()):
                if code in price_map:
                    cp = float(price_map[code])
                    cb = float(positions[code]["cost_basis"])
                    if cb > 0 and (cp - cb) / cb <= -stop_loss_pct:
                        pos = positions[code]
                        pnl = (cp - cb) * pos["shares"]
                        cash += cp * pos["shares"]
                        all_trades.append({
                            "code": code, "direction": "SELL",
                            "date": trade_date, "price": cp,
                            "shares": pos["shares"], "pnl": pnl, "reason": "stop_loss",
                        })
                        del positions[code]

            # 记录权益
            position_value = sum(
                positions[c]["shares"] * float(price_map.get(c, positions[c]["cost_basis"]))
                for c in positions
            )
            total_value = cash + position_value
            peak_value = max(peak_value, total_value)
            dd = (peak_value - total_value) / peak_value if peak_value > 0 else 0.0

            equity_curve.append({
                "date": trade_date, "equity": total_value,
                "cash": cash, "position_value": position_value,
            })

            # 最大回撤清仓
            if dd >= max_dd_limit:
                for code in list(positions.keys()):
                    if code in price_map:
                        pos = positions[code]
                        cp2 = float(price_map[code])
                        pnl2 = (cp2 - pos["cost_basis"]) * pos["shares"]
                        cash += cp2 * pos["shares"]
                        all_trades.append({
                            "code": code, "direction": "SELL",
                            "date": trade_date, "price": cp2,
                            "shares": pos["shares"], "pnl": pnl2, "reason": "max_dd",
                        })
                        del positions[code]
                peak_value = cash

            current_holdings = set(positions.keys())

        if progress_callback:
            pct = 0.50 + 0.40 * (wi + 1) / n_windows
            progress_callback(f"模拟交易... 窗口 {wi+1}/{n_windows}", pct)

    # 7. 计算指标
    if not equity_curve:
        return {"error": "无交易记录"}

    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(all_trades)

    final_value = float(eq_df["equity"].iloc[-1])
    total_return = (final_value - initial_cash) / initial_cash
    n_days = len(eq_df)
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    eq_series = eq_df["equity"]
    running_max = eq_series.cummax()
    drawdowns = (eq_series - running_max) / running_max.replace(0, 1.0)
    max_drawdown = float(drawdowns.min())

    if n_days > 1:
        returns = eq_df["equity"].pct_change().dropna()
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    sell_trades = trades_df[trades_df["direction"] == "SELL"] if not trades_df.empty else pd.DataFrame()
    if not sell_trades.empty:
        win_rate = float((sell_trades["pnl"] > 0).mean())
    else:
        win_rate = 0.0

    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown * 100,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "final_value": final_value,
        "n_trades": len(trades_df),
        "_name": f"ML: {config.get('name', 'unknown')}",
    }

    if progress_callback:
        progress_callback("完成", 1.0)

    return {
        "metrics": metrics,
        "equity_curve": eq_df,
        "trades": trades_df,
        "results_json": {
            "config_name": config.get("name"),
            "active_factors": selected,
            "n_windows": len(results),
            "n_codes": len(codes),
        },
    }
