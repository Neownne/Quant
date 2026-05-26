"""批量运行所有策略回测并保存到 backtest_results。"""
import json
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text

from data.db import get_engine, init_db
from app.utils.backtest_runner import (
    run_backtest, load_index_data, load_benchmark_indices, compute_benchmark_returns,
)
from app.utils.data_loader import load_ohlcv
from strategies import get_all_strategies, list_all_strategies
from app.utils.ml_config_manager import list_ml_configs
from app.utils.ml_backtest import run_ml_backtest

init_db()


def _safe_json(obj):
    """JSON-safe serialize, stripping non-serializable values."""
    def _convert(o):
        if isinstance(o, (str, int, float, bool, type(None))):
            return o
        if isinstance(o, (pd.Timestamp, datetime)):
            return str(o)
        if isinstance(o, dict):
            return {k: _convert(v) for k, v in o.items()
                    if not k.startswith("created") and not k.startswith("updated")}
        if isinstance(o, (list, tuple)):
            return [_convert(v) for v in o]
        try:
            return str(o)
        except Exception:
            return None
    return json.dumps(_convert(obj))


def save_backtest_result(engine, account_id, strategy_type, strategy_name, strategy_params,
                         asset_mode, pool_name, start_date, end_date,
                         n_stocks, avg_return, avg_sharpe, avg_drawdown, avg_win_rate,
                         results_json):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO backtest_results
                (account_id, strategy_type, strategy_name, strategy_params,
                 asset_mode, pool_name, start_date, end_date,
                 n_stocks, avg_return, avg_sharpe, avg_drawdown, avg_win_rate, results_json)
            VALUES (:aid, :stype, :sname, CAST(:sparams AS jsonb),
                 :amode, :pname, :sdate, :edate,
                 :nstocks, :aret, :asharpe, :add, :awin, CAST(:rjson AS jsonb))
        """), {
            "aid": account_id, "stype": strategy_type, "sname": strategy_name,
            "sparams": _safe_json(strategy_params),
            "amode": asset_mode, "pname": pool_name,
            "sdate": start_date, "edate": end_date, "nstocks": n_stocks,
            "aret": avg_return, "asharpe": avg_sharpe, "add": avg_drawdown,
            "awin": avg_win_rate, "rjson": _safe_json(results_json),
        })


engine = get_engine()

# -- 获取候选股票池 --
codes_df = pd.read_sql(
    "SELECT code FROM stock_basic WHERE is_st = FALSE "
    "AND list_date <= CURRENT_DATE - INTERVAL '365 days' "
    "ORDER BY code LIMIT 50",
    engine,
)
all_codes = codes_df["code"].tolist()
print(f"候选股票池: {len(all_codes)} 只")

start_date = date.today() - timedelta(days=365 * 5)
end_date = date.today()
start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")
print(f"回测区间: {start_str} ~ {end_str}")

# 加载基准收益（所有回测共用）
benchmarks = load_benchmark_indices(start_str, end_str)
benchmark_returns = compute_benchmark_returns(benchmarks, start_str, end_str)

# ============================================================
# 静态策略回测（股票池模式）
# ============================================================
print("\n=== 静态策略回测 ===")
static_strategies = get_all_strategies()

for sname, scls in static_strategies.items():
    print(f"\n回测: {sname} ({scls.__name__})")
    all_results = []
    index_df = load_index_data(start_str, end_str)
    params = {}

    for i, code in enumerate(all_codes):
        df = load_ohlcv(code, start_str, end_str)
        if df.empty or len(df) < 50:
            continue
        try:
            result = run_backtest(
                strategy_class=scls, df=df, strategy_params=params,
                initial_cash=1_000_000, commission=0.00009, stamp_duty=0.0005,
                slippage=0.01, index_df=index_df,
            )
        except Exception:
            continue
        m = result["metrics"]
        all_results.append({
            "code": code, "total_return": m.get("total_return", 0),
            "annual_return": m.get("annual_return", 0),
            "max_drawdown": m.get("max_drawdown", 0),
            "sharpe_ratio": m.get("sharpe_ratio", 0),
            "win_rate": m.get("win_rate", 0),
            "final_value": m.get("final_value", 0),
            "n_trades": len(result.get("trades", pd.DataFrame())),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(all_codes)} done")

    if not all_results:
        print(f"  SKIP: 无有效回测结果")
        continue

    res_df = pd.DataFrame(all_results)
    print(f"  完成 {len(res_df)} 只, 平均收益: {res_df['total_return'].mean()*100:.2f}%")

    save_backtest_result(
        engine=engine, account_id=0, strategy_type="static",
        strategy_name=scls.__name__, strategy_params=params,
        asset_mode="pool", pool_name="",
        start_date=start_date, end_date=end_date,
        n_stocks=len(res_df),
        avg_return=float(res_df["total_return"].mean()),
        avg_sharpe=float(res_df["sharpe_ratio"].mean()),
        avg_drawdown=float(res_df["max_drawdown"].mean()),
        avg_win_rate=float(res_df["win_rate"].mean()),
        results_json={
            "pool_results": res_df.to_dict(orient="records"),
            "benchmark_returns": benchmark_returns,
        },
    )

# ============================================================
# ML 策略回测
# ============================================================
print("\n=== ML 策略回测 ===")
ml_configs = list_ml_configs()

for _, row in ml_configs.iterrows():
    cfg_name = row["name"]
    # Load full config
    from app.utils.ml_config_manager import get_ml_config_by_name
    cfg = get_ml_config_by_name(cfg_name)
    if not cfg:
        print(f"  SKIP: 无法加载配置 {cfg_name}")
        continue

    print(f"\n回测: {cfg_name}")
    print(f"  模型: {cfg.get('model_type')}, Top-{cfg.get('top_n')} {cfg.get('rebalance_mode')}")

    try:
        ml_result = run_ml_backtest(
            config=cfg, codes=all_codes,
            start_date=start_str, end_date=end_str,
            initial_cash=1_000_000,
            progress_callback=lambda stage, pct: print(f"  {stage} ({pct*100:.0f}%)"),
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        continue

    if "error" in ml_result:
        print(f"  ERROR: {ml_result['error']}")
        continue

    m = ml_result["metrics"]
    eq = ml_result["equity_curve"]
    print(f"  收益: {m.get('total_return', 0)*100:.2f}%, 夏普: {m.get('sharpe_ratio', 0):.2f}")

    eq_data = {}
    if not eq.empty:
        eq_data = {
            "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                      for d in eq["date"].tolist()],
            "values": [float(v) for v in eq["equity"].tolist()],
        }

    save_backtest_result(
        engine=engine, account_id=0, strategy_type="ml",
        strategy_name=cfg_name, strategy_params=cfg,
        asset_mode="pool", pool_name="",
        start_date=start_date, end_date=end_date,
        n_stocks=len(all_codes),
        avg_return=m.get("total_return", 0),
        avg_sharpe=m.get("sharpe_ratio", 0),
        avg_drawdown=m.get("max_drawdown", 0),
        avg_win_rate=m.get("win_rate", 0),
        results_json={
            **ml_result.get("results_json", {}),
            "equity_curve": eq_data,
            "benchmark_returns": benchmark_returns,
        },
    )

engine.dispose()
print("\n=== 全部回测完成 ===")
