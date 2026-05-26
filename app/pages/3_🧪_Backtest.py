import hashlib
import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

from app.utils.backtest_runner import (
    load_index_data, run_backtest, load_benchmark_indices,
    compute_benchmark_returns, BENCHMARK_INDICES,
)
from app.utils.data_loader import get_stock_list, load_ohlcv
from app.utils.stock_pools import list_pools, load_and_execute_pool, get_pool_data
from app.utils.account_manager import promote_strategy_to_account
from data.db import init_db, get_engine
from strategies import get_all_strategies, list_all_strategies, is_ml_strategy, is_static_strategy
from app.utils.ml_backtest import run_ml_backtest

st.set_page_config(page_title="策略回测", page_icon="🧪", layout="wide")
st.title("🧪 策略回测")

init_db()

# ── 回测结果保存 ─────────────────────────────────────────

def _safe_json(obj):
    """JSON-safe serialize, stripping non-serializable values."""
    def _convert(o):
        if isinstance(o, (str, int, float, bool, type(None))):
            return o
        if isinstance(o, (datetime, pd.Timestamp)):
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


def save_backtest_result(account_id, strategy_type, strategy_name, strategy_params,
                         asset_mode, pool_name, start_date, end_date,
                         n_stocks, avg_return, avg_sharpe, avg_drawdown, avg_win_rate,
                         results_json):
    engine = get_engine()
    try:
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
                "aid": account_id,
                "stype": strategy_type,
                "sname": strategy_name,
                "sparams": _safe_json(strategy_params),
                "amode": asset_mode,
                "pname": pool_name or "",
                "sdate": start_date,
                "edate": end_date,
                "nstocks": n_stocks,
                "aret": avg_return,
                "asharpe": avg_sharpe,
                "add": avg_drawdown,
                "awin": avg_win_rate,
                "rjson": _safe_json(results_json),
            })
    finally:
        engine.dispose()


@st.cache_data(ttl=30)
def load_backtest_history() -> pd.DataFrame:
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(
                "SELECT id, strategy_type, strategy_name, strategy_params, "
                "asset_mode, pool_name, start_date, end_date, n_stocks, "
                "avg_return, avg_sharpe, avg_drawdown, avg_win_rate, results_json, created_at "
                "FROM backtest_results ORDER BY created_at DESC LIMIT 50"
            ), conn)
    finally:
        engine.dispose()
    return df

# ---- 侧边栏：回测参数 ----
with st.sidebar:
    st.header("回测设置")

    # ── 统一策略选择 ──
    st.subheader("策略选择")
    unified_strategies = list_all_strategies()
    all_names = list(unified_strategies.keys())
    static_names = [n for n in all_names if is_static_strategy(unified_strategies[n])]
    ml_names = [n for n in all_names if is_ml_strategy(unified_strategies[n])]

    strategy_type_filter = st.radio("策略类型", ["静态策略", "ML策略"], horizontal=True)

    if strategy_type_filter == "静态策略":
        strategy_name = st.selectbox("策略", static_names if static_names else ["无可用策略"])
        strategy_info = unified_strategies.get(strategy_name, {})
        strategy_class = strategy_info.get("class") if strategy_info.get("type") == "static" else None
        ml_config = {}
    else:
        strategy_name = st.selectbox("ML配置", ml_names if ml_names else ["无可用ML配置"])
        strategy_info = unified_strategies.get(strategy_name, {})
        strategy_class = None
        ml_config = strategy_info.get("config", {}) if strategy_info.get("type") == "ml" else {}

    # ── 策略参数 ──
    if strategy_type_filter == "静态策略" and strategy_class is not None:
        st.subheader("策略参数")
        strategy_params = {}
        cls_name = strategy_class.__name__

        param_defs = {
            "SMACross": {"fast": (5, 1, 120), "slow": (20, 1, 250)},
            "MACDStrategy": {"fast": (12, 2, 50), "slow": (26, 5, 100), "signal": (9, 2, 30)},
            "RSIStrategy": {"period": (14, 2, 50), "oversold": (30, 10, 40), "overbought": (70, 60, 90)},
        }

        if cls_name in param_defs:
            for pname, (default, pmin, pmax) in param_defs[cls_name].items():
                strategy_params[pname] = st.number_input(
                    pname, min_value=pmin, max_value=pmax, value=default, step=1
                )
        else:
            params = getattr(strategy_class, "params", ())
            for pdef in params:
                pname, default = pdef[0], pdef[1]
                if isinstance(default, (int, float)):
                    strategy_params[pname] = st.number_input(
                        pname, value=default, step=1 if isinstance(default, int) else 1
                    )
    elif strategy_type_filter == "ML策略":
        st.subheader("ML 策略参数")
        if ml_config:
            st.caption(f"**{ml_config.get('name', '')}** — {ml_config.get('description', '')}")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("模型", ml_config.get("model_type", "ensemble"))
                tr = ml_config.get("train_years", 3)
                vr = ml_config.get("val_years", 1)
                st.metric("训练窗", f"{tr}+{vr}年")
                st.metric("选股", f"Top-{ml_config.get('top_n', 15)} {ml_config.get('rebalance_mode', 'ndrop')}")
            with col_b:
                st.metric("IC阈值", f"{ml_config.get('ic_threshold', 0.02):.3f}")
                st.metric("正交阈值", f"{ml_config.get('orthogonal_threshold', 0.7):.2f}")
                st.metric("止损", f"{float(ml_config.get('stop_loss_pct', 0.08))*100:.0f}%")
            freq = st.selectbox(
                "数据频率",
                options=["daily", "60min"],
                index=0,
                key="ml_backtest_freq",
                help="60分钟线使用 Sina 前复权数据。分钟频回测耗时更长，建议先用少量股票测试。",
            )
            if freq == "60min":
                st.caption("⏱ 分钟频回测耗时较长（因子计算量 ×4），建议先用 50-100 只股票测试。")
            strategy_params = ml_config
        else:
            st.warning("请先在「策略编辑器」中创建 ML 策略配置")
            strategy_params = {}
    else:
        strategy_params = {}

    st.divider()
    st.subheader("资金 & 费用 (A股实盘)")
    initial_cash = st.number_input("初始资金", min_value=10000, value=1_000_000, step=10000)
    commission = st.number_input("佣金费率（万0.9，买卖双向）", min_value=0.0, value=0.00009, step=0.00001, format="%.5f")
    stamp_duty = st.number_input("印花税率（万5，仅卖出）", min_value=0.0, value=0.0005, step=0.0001, format="%.4f")
    slippage = st.number_input("滑点（元/股）", min_value=0.0, value=0.01, step=0.01, format="%.2f")
    use_market_filter = st.checkbox(
        "大盘年线过滤器",
        value=True,
        help="启用后，上证指数在 200 日均线下方时自动将仓位降至 40%",
    )
    st.caption("T+1 交易制度由日线级别回测天然保证（当日买入次日才能卖出）")

    st.divider()
    st.subheader("标的选择")

    # 选择模式：单只 or 股票池
    asset_mode = st.radio(
        "选择方式",
        ["单只股票", "股票池"],
        horizontal=True,
    )

    if asset_mode == "单只股票":
        from app.utils.data_loader import get_all_assets
        all_assets = get_all_assets()
        tradeable = all_assets[all_assets["type"].isin(["stock"])]
        code_options = [f"{r['code']} {r['name']}" for _, r in tradeable.iterrows()] if not tradeable.empty else ["000001"]
        selected = st.selectbox("股票", code_options)
        code = selected.split()[0] if selected else "000001"
        pool_codes = []
    else:
        pools = list_pools()
        if not pools:
            st.warning("暂无自定义股票池，请先在「股票池」页面创建")
            pool_codes = []
            selected_pool = ""
        else:
            selected_pool = st.selectbox("选择股票池", pools)
            if st.button("🔍 预览股票池"):
                engine = get_engine()
                try:
                    basic, extra, shareholder = get_pool_data(engine)
                    pool_codes = load_and_execute_pool(selected_pool, basic, extra, shareholder)
                    st.info(f"共 {len(pool_codes)} 只股票: {', '.join(pool_codes[:10])}{'...' if len(pool_codes) > 10 else ''}")
                finally:
                    engine.dispose()
            else:
                pool_codes = []
            code = ""  # pool mode doesn't use single code

    # 快捷区间选择
    quick_ranges = {
        "1年": 365, "3年": 365 * 3, "5年": 365 * 5,
        "10年": 365 * 10, "全部": 365 * 20,
    }
    quick_cols = st.columns(len(quick_ranges))
    selected_range = st.session_state.get("backtest_range", "5年")
    for i, (label, days) in enumerate(quick_ranges.items()):
        with quick_cols[i]:
            if st.button(label, key=f"range_{label}", use_container_width=True,
                        type="primary" if label == selected_range else "secondary"):
                st.session_state["backtest_range"] = label
                st.session_state["backtest_start"] = date.today() - timedelta(days=days)
                st.session_state["backtest_end"] = date.today()
                st.rerun()

    default_start = st.session_state.get("backtest_start", date.today() - timedelta(days=365 * 5))
    default_end = st.session_state.get("backtest_end", date.today())
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("起始", value=default_start)
    with col2:
        end_date = st.date_input("截止", value=default_end)

    run_btn = st.button("🚀 运行回测", use_container_width=True, type="primary")

# ---- 主区域：历史记录 & 策略对比（始终可见） ----
st.subheader("📋 历史回测记录")

history = load_backtest_history()
if history.empty:
    st.info("暂无保存的回测记录，运行一次回测后自动保存")
else:
    for _, row in history.iterrows():
        sp = row["strategy_params"]
        if isinstance(sp, str):
            try:
                sp = json.loads(sp)
            except Exception:
                sp = {}

        with st.expander(
            f"{row['created_at'].strftime('%m-%d %H:%M') if hasattr(row['created_at'], 'strftime') else str(row['created_at'])[:16]} | "
            f"{row['strategy_name']} | "
            f"{row['asset_mode']} | "
            f"收益: {row['avg_return']*100:.2f}% | "
            f"夏普: {row['avg_sharpe']:.2f} | "
            f"回撤: {row['avg_drawdown']:.2f}%"
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"策略类型: {row['strategy_type']} | 模式: {row['asset_mode']}")
                st.caption(f"区间: {row['start_date']} ~ {row['end_date']} | 股票数: {row['n_stocks']}")
                if sp:
                    st.caption(f"参数: {sp}")
            with col2:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🚀 升级到模拟盘", key=f"promote_{row['id']}"):
                        new_id = promote_strategy_to_account(
                            strategy_type=row["strategy_type"],
                            strategy_name=row["strategy_name"],
                            strategy_params=sp,
                        )
                        st.success(f"已创建模拟账户 #{new_id}，请切换到「模拟盘」页面运行")
                with c2:
                    if st.button("📊 查看详情", key=f"detail_{row['id']}"):
                        rj = row.get("results_json")
                        if isinstance(rj, str):
                            rj = json.loads(rj)
                        st.json(rj if rj else {})

# -- 策略对比 --
st.divider()
st.subheader("📊 策略对比")
history_df = load_backtest_history()
if not history_df.empty:
    history_opts = []
    for _, r in history_df.iterrows():
        label = f"#{r['id']} {r['strategy_name']} ({r['strategy_type']}) - {str(r['start_date'])[:10]}"
        history_opts.append(label)

    selected_for_compare = st.multiselect(
        "选择要对比的回测记录（最多5条）", history_opts,
        max_selections=5, key="compare_select"
    )

    if selected_for_compare:
        selected_ids = [int(opt.split()[0][1:]) for opt in selected_for_compare]
        compare_rows = history_df[history_df["id"].isin(selected_ids)]

        # -- 对比表格 --
        compare_data = []
        for _, row in compare_rows.iterrows():
            rj = row.get("results_json")
            if isinstance(rj, str):
                try:
                    rj = json.loads(rj)
                except Exception:
                    rj = {}
            bm_returns = rj.get("benchmark_returns", {}) if isinstance(rj, dict) else {}
            bm_000001 = bm_returns.get("000001", 0) if isinstance(bm_returns, dict) else 0
            strat_ret = float(row["avg_return"])
            excess = strat_ret - float(bm_000001) if bm_000001 else 0

            # 计算累计盈亏 & 年化收益
            eq_data = rj.get("equity_curve", {}) if isinstance(rj, dict) else {}
            eq_vals = eq_data.get("values", [])
            if eq_vals and len(eq_vals) > 1:
                init_val = eq_vals[0]
                final_val = eq_vals[-1]
                cum_pnl = final_val - init_val
                cum_pnl_str = f"{cum_pnl:+,.0f}"
            else:
                cum_pnl_str = f"{strat_ret * 100:+.2f}%"

            # 年化收益率
            try:
                sdate = pd.Timestamp(row["start_date"])
                edate = pd.Timestamp(row["end_date"])
                n_years = max((edate - sdate).days, 1) / 365.25
                ann_ret = (1 + strat_ret) ** (1 / n_years) - 1 if strat_ret > -1 else strat_ret
            except Exception:
                ann_ret = strat_ret

            compare_data.append({
                "策略名": row["strategy_name"],
                "类型": row["strategy_type"],
                "累计盈亏": cum_pnl_str,
                "总收益%": f"{strat_ret * 100:.2f}",
                "年化%": f"{ann_ret * 100:.2f}",
                "上证同期%": f"{float(bm_000001) * 100:.2f}" if bm_000001 else "-",
                "超额%": f"{excess * 100:+.2f}",
                "夏普": f"{float(row['avg_sharpe']):.2f}",
                "回撤%": f"{float(row['avg_drawdown']):.2f}",
                "胜率%": f"{float(row['avg_win_rate']) * 100:.1f}",
                "股票数": int(row["n_stocks"]),
                "区间": f"{str(row['start_date'])[:10]}~{str(row['end_date'])[:10]}",
            })
        st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

        # -- 叠加权益曲线 --
        st.subheader("📈 归一化权益曲线对比")
        import plotly.graph_objects as go

        # 基准选择
        bm_opts = [f"{c} {BENCHMARK_INDICES[c]}" for c in BENCHMARK_INDICES]
        selected_bms = st.multiselect(
            "叠加基准指数", bm_opts,
            default=[f"000001 {BENCHMARK_INDICES['000001']}", f"000300 {BENCHMARK_INDICES['000300']}"],
            key="compare_bm",
        )
        selected_bm_codes = [o.split()[0] for o in selected_bms]

        # 加载基准数据
        bm_start = str(compare_rows["start_date"].min()).replace("-", "")[:8]
        bm_end = str(compare_rows["end_date"].max()).replace("-", "")[:8]
        compare_bms = load_benchmark_indices(bm_start, bm_end)

        fig_compare = go.Figure()
        strategy_colors = ["#2196f3", "#ff9800", "#4caf50", "#f44336", "#9c27b0"]
        bm_line_styles = ["dash", "dot", "dashdot", "longdash", "longdashdot", "solid"]

        # 先画基准曲线
        for bi, bm_code in enumerate(selected_bm_codes):
            bm_df = compare_bms.get(bm_code)
            if bm_df is not None and not bm_df.empty and len(bm_df) > 1:
                bm_norm = bm_df["close"] / bm_df["close"].iloc[0]
                fig_compare.add_trace(go.Scatter(
                    x=bm_df["trade_date"], y=bm_norm, mode="lines",
                    name=f"基准: {BENCHMARK_INDICES.get(bm_code, bm_code)}",
                    line=dict(color="gray", width=1.2, dash=bm_line_styles[bi % len(bm_line_styles)]),
                    opacity=0.65,
                ))

        for ci, (_, row) in enumerate(compare_rows.iterrows()):
            rj = row.get("results_json")
            if isinstance(rj, str):
                try:
                    rj = json.loads(rj)
                except Exception:
                    rj = {}
            eq_data = rj.get("equity_curve", {}) if isinstance(rj, dict) else {}
            dates = eq_data.get("dates", [])
            values = eq_data.get("values", [])
            if dates and values and len(dates) > 1:
                norm_values = [v / values[0] for v in values] if values[0] > 0 else values
                color = strategy_colors[ci % len(strategy_colors)]
                fig_compare.add_trace(go.Scatter(
                    x=dates, y=norm_values, mode="lines",
                    name=f"{row['strategy_name']} ({row['strategy_type']})",
                    line=dict(color=color, width=1.8),
                ))

        fig_compare.add_hline(y=1.0, line_dash="dash", line_color="gray",
                             annotation_text="0% 收益线")
        fig_compare.update_layout(
            height=400, template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="归一化净值 (1.0 = 0%收益)",
        )
        if len(fig_compare.data) > 0:
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.caption("所选记录暂无可展示的权益曲线数据")

st.divider()

# ---- 主区域：回测结果 ----
if not run_btn:
    st.info("👆 上方可直接查看历史记录与策略对比，或左侧设置参数后运行新回测")
    st.stop()

start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

# 确定回测标的列表
if asset_mode == "股票池" and selected_pool in list_pools():
    engine = get_engine()
    try:
        basic, extra, shareholder = get_pool_data(engine)
        target_codes = load_and_execute_pool(selected_pool, basic, extra, shareholder)
    finally:
        engine.dispose()
    if not target_codes:
        st.error("股票池筛选结果为空")
        st.stop()
    st.info(f"将对 {len(target_codes)} 只股票逐一回测")
else:
    target_codes = [code]

# 加载大盘指数（只加载一次）
index_df = None
if use_market_filter:
    with st.spinner("加载上证指数数据 ..."):
        index_df = load_index_data(start_str, end_str)

# 加载基准指数数据
with st.spinner("加载基准指数数据 ..."):
    benchmarks = load_benchmark_indices(start_str, end_str)
    benchmark_returns = compute_benchmark_returns(benchmarks, start_str, end_str)

# ---- 逐只回测 ----
all_results: list[dict] = []
progress = st.progress(0, text=f"0/{len(target_codes)}")

if strategy_type_filter == "静态策略":
    for i, c in enumerate(target_codes):
        progress.progress((i + 1) / len(target_codes), text=f"{i + 1}/{len(target_codes)}: {c}")

        df = load_ohlcv(c, start_str, end_str)
        if df.empty or len(df) < 50:
            continue

        try:
            result = run_backtest(
                strategy_class=strategy_class,
                df=df,
                strategy_params=strategy_params,
                initial_cash=initial_cash,
                commission=commission,
                stamp_duty=stamp_duty,
                slippage=slippage,
                index_df=index_df,
            )
        except Exception:
            continue

        m = result["metrics"]
        all_results.append({
            "code": c,
            "name": m.get("_name", ""),
            "total_return": m.get("total_return", 0),
            "annual_return": m.get("annual_return", 0),
            "max_drawdown": m.get("max_drawdown", 0),
            "sharpe_ratio": m.get("sharpe_ratio", 0),
            "win_rate": m.get("win_rate", 0),
            "final_value": m.get("final_value", 0),
            "n_trades": len(result.get("trades", pd.DataFrame())),
        })

    progress.empty()
else:
    # ML 回测：一次性在所有标的上运行
    progress.empty()
    with st.spinner("运行 ML 回测（因子计算+训练+模拟交易，约需3-5分钟）..."):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        def ml_progress(stage, pct):
            progress_text.text(f"⏳ {stage}")
            progress_bar.progress(min(pct, 1.0))

        ml_config_with_freq = dict(ml_config)
        ml_config_with_freq["freq"] = freq

        ml_result = run_ml_backtest(
            config=ml_config_with_freq,
            codes=target_codes,
            start_date=start_str,
            end_date=end_str,
            initial_cash=initial_cash,
            progress_callback=ml_progress,
        )
        progress_text.empty()
        progress_bar.empty()

    if "error" in ml_result:
        st.error(f"ML 回测失败: {ml_result['error']}")
        st.stop()

    all_results.append({
        "code": "ML组合",
        "name": ml_config.get("name", "ML策略"),
        "total_return": ml_result["metrics"]["total_return"],
        "annual_return": ml_result["metrics"]["annual_return"],
        "max_drawdown": ml_result["metrics"]["max_drawdown"],
        "sharpe_ratio": ml_result["metrics"]["sharpe_ratio"],
        "win_rate": ml_result["metrics"]["win_rate"],
        "final_value": ml_result["metrics"]["final_value"],
        "n_trades": ml_result["metrics"]["n_trades"],
    })
    ml_single_result = ml_result

if not all_results:
    st.error("所有股票均无足够数据或回测失败")
    st.stop()

results_df = pd.DataFrame(all_results)

# ---- 单只模式：展示原有详情 ----
if len(target_codes) == 1 and strategy_type_filter == "静态策略":
    # Re-run with full detail for single stock
    c = target_codes[0]
    df = load_ohlcv(c, start_str, end_str)
    st.caption(f"数据范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}，共 {len(df)} 条")

    result = run_backtest(
        strategy_class=strategy_class,
        df=df,
        strategy_params=strategy_params,
        initial_cash=initial_cash,
        commission=commission,
        stamp_duty=stamp_duty,
        slippage=slippage,
        index_df=index_df,
    )
    metrics = result["metrics"]
    equity_curve = result["equity_curve"]
    trades_df = result["trades"]

    st.subheader("核心指标")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("总收益率", f"{metrics.get('total_return', 0) * 100:.2f}%")
    with col2:
        st.metric("年化收益", f"{metrics.get('annual_return', 0) * 100:.2f}%")
    with col3:
        st.metric("最大回撤", f"{metrics.get('max_drawdown', 0):.2f}%")
    with col4:
        st.metric("夏普比率", f"{metrics.get('sharpe_ratio', 0):.2f}")
    with col5:
        st.metric("胜率", f"{metrics.get('win_rate', 0) * 100:.1f}%")
    st.metric("最终权益", f"{metrics.get('final_value', 0):,.0f} 元")

    # 基准收益
    with st.expander("📊 基准指数同期收益", expanded=False):
        bcols = st.columns(min(len(benchmark_returns), 6))
        for i, (code, ret) in enumerate(benchmark_returns.items()):
            if code in BENCHMARK_INDICES:
                with bcols[i % len(bcols)]:
                    st.metric(
                        BENCHMARK_INDICES[code],
                        f"{ret * 100:.2f}%",
                        delta=f"vs 策略 {(metrics.get('total_return', 0) - ret) * 100:+.2f}%",
                    )

    st.subheader("权益曲线")
    if not equity_curve.empty:
        import plotly.graph_objects as go

        # 基准选择
        bm_opts_static = [f"{c} {BENCHMARK_INDICES[c]}" for c in BENCHMARK_INDICES]
        selected_bms_static = st.multiselect(
            "叠加基准指数", bm_opts_static,
            default=[f"000001 {BENCHMARK_INDICES['000001']}", f"000300 {BENCHMARK_INDICES['000300']}"],
            key="single_static_bm",
        )
        selected_bm_codes_static = [o.split()[0] for o in selected_bms_static]

        fig = go.Figure()
        bm_line_styles = ["dash", "dot", "dashdot", "longdash", "longdashdot", "solid"]

        # 基准归一化曲线
        for bi, bm_code in enumerate(selected_bm_codes_static):
            bm_df = benchmarks.get(bm_code)
            if bm_df is not None and not bm_df.empty:
                bm_period = bm_df[(bm_df["trade_date"] >= equity_curve["date"].iloc[0]) &
                                  (bm_df["trade_date"] <= equity_curve["date"].iloc[-1])]
                if not bm_period.empty and len(bm_period) > 1:
                    bm_norm = bm_period["close"] / bm_period["close"].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=bm_period["trade_date"], y=bm_norm, mode="lines",
                        name=f"基准: {BENCHMARK_INDICES.get(bm_code, bm_code)}",
                        line=dict(color="gray", width=1.2,
                                  dash=bm_line_styles[bi % len(bm_line_styles)]),
                        opacity=0.65,
                    ))

        # 策略归一化
        eq_norm = equity_curve["equity"] / equity_curve["equity"].iloc[0]
        fig.add_trace(go.Scatter(
            x=equity_curve["date"], y=eq_norm,
            mode="lines", name="策略权益",
            fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
            line=dict(color="#2196f3", width=2),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="0% 收益线")
        fig.update_layout(
            height=400, template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
            yaxis_title="归一化净值 (1.0 = 0%收益)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("交易明细")
    if not trades_df.empty:
        trades_display = trades_df.copy()
        # Format P&L columns
        for col in ["盈亏(元)", "盈亏%"]:
            if col in trades_display.columns:
                trades_display[col] = trades_display[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
                )
        # Format price columns
        for col in ["买入价", "卖出价"]:
            if col in trades_display.columns:
                trades_display[col] = trades_display[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                )
        st.dataframe(trades_display, use_container_width=True, hide_index=True)
        pnl_col = "盈亏(元)" if "盈亏(元)" in trades_df.columns else "pnl"
        total_pnl = trades_df[pnl_col].sum() if pnl_col in trades_df.columns else 0
        st.caption(f"总盈亏: {total_pnl:,.0f} 元  |  交易次数: {len(trades_df)}")

    # ── 单只回测：保存结果 ──
    cls_name = strategy_class.__name__
    save_backtest_result(
        account_id=0,
        strategy_type="static",
        strategy_name=cls_name,
        strategy_params=strategy_params,
        asset_mode="single",
        pool_name="",
        start_date=start_date,
        end_date=end_date,
        n_stocks=1,
        avg_return=metrics.get("total_return", 0),
        avg_sharpe=metrics.get("sharpe_ratio", 0),
        avg_drawdown=metrics.get("max_drawdown", 0),
        avg_win_rate=metrics.get("win_rate", 0),
        results_json={
            "code": c, "name": metrics.get("_name", ""),
            "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
            "equity_curve": {
                "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                          for d in equity_curve["date"].tolist()],
                "values": [float(v) for v in equity_curve["equity"].tolist()],
            } if not equity_curve.empty else {},
            "benchmark_returns": benchmark_returns,
        },
    )

elif len(target_codes) == 1:
    # ML single result display
    m = ml_single_result["metrics"]
    eq = ml_single_result["equity_curve"]

    st.subheader("核心指标")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("总收益率", f"{m.get('total_return', 0) * 100:.2f}%")
    with col2:
        st.metric("年化收益", f"{m.get('annual_return', 0) * 100:.2f}%")
    with col3:
        st.metric("最大回撤", f"{m.get('max_drawdown', 0):.2f}%")
    with col4:
        st.metric("夏普比率", f"{m.get('sharpe_ratio', 0):.2f}")
    with col5:
        st.metric("胜率", f"{m.get('win_rate', 0) * 100:.1f}%")
    st.metric("最终权益", f"{m.get('final_value', 0):,.0f} 元")

    # 基准收益
    with st.expander("📊 基准指数同期收益", expanded=False):
        bcols = st.columns(min(len(benchmark_returns), 6))
        for i, (code, ret) in enumerate(benchmark_returns.items()):
            if code in BENCHMARK_INDICES:
                with bcols[i % len(bcols)]:
                    st.metric(
                        BENCHMARK_INDICES[code],
                        f"{ret * 100:.2f}%",
                        delta=f"vs 策略 {(m.get('total_return', 0) - ret) * 100:+.2f}%",
                    )

    if not eq.empty:
        st.subheader("权益曲线")
        import plotly.graph_objects as go

        bm_opts_ml = [f"{c} {BENCHMARK_INDICES[c]}" for c in BENCHMARK_INDICES]
        selected_bms_ml = st.multiselect(
            "叠加基准指数", bm_opts_ml,
            default=[f"000001 {BENCHMARK_INDICES['000001']}", f"000300 {BENCHMARK_INDICES['000300']}"],
            key="single_ml_bm",
        )
        selected_bm_codes_ml = [o.split()[0] for o in selected_bms_ml]

        fig = go.Figure()
        bm_line_styles = ["dash", "dot", "dashdot", "longdash", "longdashdot", "solid"]

        for bi, bm_code in enumerate(selected_bm_codes_ml):
            bm_df = benchmarks.get(bm_code)
            if bm_df is not None and not bm_df.empty:
                bm_period = bm_df[(bm_df["trade_date"] >= eq["date"].iloc[0]) &
                                  (bm_df["trade_date"] <= eq["date"].iloc[-1])]
                if not bm_period.empty and len(bm_period) > 1:
                    bm_norm = bm_period["close"] / bm_period["close"].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=bm_period["trade_date"], y=bm_norm, mode="lines",
                        name=f"基准: {BENCHMARK_INDICES.get(bm_code, bm_code)}",
                        line=dict(color="gray", width=1.2,
                                  dash=bm_line_styles[bi % len(bm_line_styles)]),
                        opacity=0.65,
                    ))

        # 策略归一化
        eq_norm_ml = eq["equity"] / eq["equity"].iloc[0]
        fig.add_trace(go.Scatter(
            x=eq["date"], y=eq_norm_ml, mode="lines", name="策略权益",
            fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
            line=dict(color="#2196f3", width=2),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="0% 收益线")
        fig.update_layout(height=400, template="plotly_white",
                          margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                          yaxis_title="归一化净值 (1.0 = 0%收益)")
        st.plotly_chart(fig, use_container_width=True)

    trades = ml_single_result["trades"]
    if not trades.empty:
        st.subheader("交易明细")
        st.dataframe(trades, use_container_width=True, hide_index=True)
        total_pnl = float(trades[trades["direction"] == "SELL"]["pnl"].sum()) if "pnl" in trades.columns else 0
        st.caption(f"总盈亏: {total_pnl:,.0f} 元  |  交易次数: {len(trades)}")

    # Save ML result
    save_backtest_result(
        account_id=0, strategy_type="ml",
        strategy_name=ml_config.get("name", "unknown"),
        strategy_params=ml_config,
        asset_mode="single",
        pool_name="",
        start_date=start_date, end_date=end_date,
        n_stocks=len(target_codes),
        avg_return=m.get("total_return", 0),
        avg_sharpe=m.get("sharpe_ratio", 0),
        avg_drawdown=m.get("max_drawdown", 0),
        avg_win_rate=m.get("win_rate", 0),
        results_json={
            **ml_single_result.get("results_json", {}),
            "equity_curve": {
                "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                          for d in eq["date"].tolist()],
                "values": [float(v) for v in eq["equity"].tolist()],
            } if not eq.empty else {},
            "benchmark_returns": benchmark_returns,
        },
    )

# ---- 股票池模式：汇总统计 ----
else:
    st.subheader(f"汇总统计（{len(results_df)} 只股票）")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均总收益率", f"{results_df['total_return'].mean() * 100:.2f}%")
    with col2:
        st.metric("中位数总收益率", f"{results_df['total_return'].median() * 100:.2f}%")
    with col3:
        st.metric("平均夏普", f"{results_df['sharpe_ratio'].mean():.2f}")
    with col4:
        st.metric("平均回撤", f"{results_df['max_drawdown'].mean():.2f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        win_pct = (results_df["total_return"] > 0).mean() * 100
        st.metric("胜率（正收益比例）", f"{win_pct:.1f}%")
    with col2:
        st.metric("平均胜率", f"{results_df['win_rate'].mean() * 100:.1f}%")
    with col3:
        st.metric("最佳收益", f"{results_df['total_return'].max() * 100:.2f}%")
    with col4:
        st.metric("最差收益", f"{results_df['total_return'].min() * 100:.2f}%")

    # 基准收益
    with st.expander("📊 基准指数同期收益", expanded=False):
        bcols = st.columns(min(len(benchmark_returns), 6))
        for i, (code, ret) in enumerate(benchmark_returns.items()):
            if code in BENCHMARK_INDICES:
                with bcols[i % len(bcols)]:
                    avg_ret = float(results_df["total_return"].mean())
                    st.metric(
                        BENCHMARK_INDICES[code],
                        f"{ret * 100:.2f}%",
                        delta=f"vs 均值 {(avg_ret - ret) * 100:+.2f}%",
                    )

    st.divider()

    # 排序表
    st.subheader("按总收益率排名")
    sort_col = st.selectbox("排序依据", ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"])
    top_n = st.slider("展示前 N 只", 5, min(len(results_df), 100), 20)

    ranked = results_df.sort_values(sort_col, ascending="total_return" not in sort_col).head(top_n)
    display = ranked.copy()
    for pct_col in ["total_return", "annual_return", "max_drawdown", "win_rate"]:
        display[pct_col] = (display[pct_col] * 100).round(2)
    display["sharpe_ratio"] = display["sharpe_ratio"].round(2)
    display["final_value"] = display["final_value"].round(0).astype(int)

    display = display.rename(columns={
        "code": "代码", "name": "名称",
        "total_return": "总收益%", "annual_return": "年化%",
        "max_drawdown": "回撤%", "sharpe_ratio": "夏普",
        "win_rate": "胜率%", "n_trades": "交易次数",
        "final_value": "最终权益",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    # 查看单只股票交易明细
    if strategy_type_filter == "静态策略" and strategy_class is not None:
        st.subheader("交易明细（选择股票查看）")
        detail_code = st.selectbox("选择股票", [""] + [r["code"] for r in all_results])
    else:
        detail_code = ""
    if detail_code:
        df_detail = load_ohlcv(detail_code, start_str, end_str)
        if not df_detail.empty:
            with st.spinner(f"回测 {detail_code} 中 ..."):
                result_detail = run_backtest(
                    strategy_class=strategy_class,
                    df=df_detail,
                    strategy_params=strategy_params,
                    initial_cash=initial_cash,
                    commission=commission,
                    stamp_duty=stamp_duty,
                    slippage=slippage,
                    index_df=index_df,
                )
                trades_d = result_detail["trades"]
                if not trades_d.empty:
                    td = trades_d.copy()
                    for c in ["盈亏(元)", "盈亏%"]:
                        if c in td.columns:
                            td[c] = td[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
                    for c in ["买入价", "卖出价"]:
                        if c in td.columns:
                            td[c] = td[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                    st.caption(f"{detail_code} — {len(td)} 笔交易")
                    st.dataframe(td, use_container_width=True, hide_index=True)
                else:
                    st.info(f"{detail_code} 无交易记录")
        else:
            st.warning(f"{detail_code} 无数据")

    # 分布直方图
    st.subheader("收益率分布")
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=results_df["total_return"] * 100,
        nbinsx=30,
        marker=dict(color="#2196f3", line=dict(color="white", width=0.5)),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="盈亏线")
    fig.add_vline(x=results_df["total_return"].mean() * 100, line_dash="dash",
                  line_color="green", annotation_text="均值")
    fig.update_layout(
        height=350, template="plotly_white",
        xaxis_title="总收益率 (%)", yaxis_title="股票数量",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 导出
    st.download_button(
        "📥 导出 CSV",
        results_df.to_csv(index=False).encode("utf-8"),
        file_name=f"backtest_{selected_pool}_{strategy_name}_{start_str}_{end_str}.csv",
        mime="text/csv",
    )

    # ── 股票池回测：保存结果 ──
    if strategy_type_filter == "静态策略" and strategy_class is not None:
        cls_name = strategy_class.__name__
        save_backtest_result(
            account_id=0,
            strategy_type="static",
            strategy_name=cls_name,
            strategy_params=strategy_params,
            asset_mode="pool",
            pool_name=selected_pool,
            start_date=start_date,
            end_date=end_date,
            n_stocks=len(results_df),
            avg_return=float(results_df["total_return"].mean()),
            avg_sharpe=float(results_df["sharpe_ratio"].mean()),
            avg_drawdown=float(results_df["max_drawdown"].mean()),
            avg_win_rate=float(results_df["win_rate"].mean()),
            results_json={
                "pool_results": results_df.to_dict(orient="records"),
                "benchmark_returns": benchmark_returns,
            },
        )
    elif strategy_type_filter == "ML策略":
        save_backtest_result(
            account_id=0,
            strategy_type="ml",
            strategy_name=ml_config.get("name", "unknown"),
            strategy_params=ml_config,
            asset_mode="pool",
            pool_name=selected_pool,
            start_date=start_date,
            end_date=end_date,
            n_stocks=len(results_df),
            avg_return=float(results_df["total_return"].mean()),
            avg_sharpe=float(results_df["sharpe_ratio"].mean()),
            avg_drawdown=float(results_df["max_drawdown"].mean()),
            avg_win_rate=float(results_df["win_rate"].mean()),
            results_json={
                **ml_single_result.get("results_json", {}),
                "equity_curve": {
                    "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                              for d in eq["date"].tolist()],
                    "values": [float(v) for v in eq["equity"].tolist()],
                } if not eq.empty else {},
                "benchmark_returns": benchmark_returns,
                "pool_results": results_df.to_dict(orient="records"),
            },
        )
