import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from app.utils.backtest_runner import load_index_data, run_backtest
from app.utils.data_loader import get_stock_list, load_ohlcv
from app.utils.stock_pools import list_pools, load_and_execute_pool, get_pool_data
from data.db import init_db, get_engine
from strategies import get_all_strategies

st.set_page_config(page_title="策略回测", page_icon="🧪", layout="wide")
st.title("🧪 策略回测")

init_db()

# ---- 侧边栏：回测参数 ----
with st.sidebar:
    st.header("回测设置")

    all_strategies = get_all_strategies()
    strategy_name = st.selectbox("策略", list(all_strategies.keys()))
    strategy_class = all_strategies[strategy_name]

    # 动态策略参数
    st.subheader("策略参数")
    strategy_params = {}

    # 内置策略的参数定义
    param_defs = {
        "SMACross": {"fast": (5, 1, 120), "slow": (20, 1, 250)},
        "MACDStrategy": {"fast": (12, 2, 50), "slow": (26, 5, 100), "signal": (9, 2, 30)},
        "RSIStrategy": {"period": (14, 2, 50), "oversold": (30, 10, 40), "overbought": (70, 60, 90)},
    }
    cls_name = strategy_class.__name__

    # 内置策略用预定义参数表，自定义策略从 params 推导
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

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("起始", value=date.today() - timedelta(days=365 * 3))
    with col2:
        end_date = st.date_input("截止", value=date.today())

    run_btn = st.button("🚀 运行回测", use_container_width=True, type="primary")

# ---- 主区域：回测结果 ----
if not run_btn:
    st.info("在左侧设置参数后点击「运行回测」")
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

# ---- 逐只回测 ----
all_results: list[dict] = []
progress = st.progress(0, text=f"0/{len(target_codes)}")

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

if not all_results:
    st.error("所有股票均无足够数据或回测失败")
    st.stop()

results_df = pd.DataFrame(all_results)

# ---- 单只模式：展示原有详情 ----
if len(target_codes) == 1:
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

    st.subheader("权益曲线")
    if not equity_curve.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve["date"], y=equity_curve["equity"],
            mode="lines", name="权益",
            fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
            line=dict(color="#2196f3", width=2),
        ))
        fig.add_hline(y=initial_cash, line_dash="dash", line_color="gray",
                      annotation_text=f"初始 {initial_cash:,.0f}")
        fig.update_layout(
            height=400, template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("交易明细")
    if not trades_df.empty:
        trades_display = trades_df.copy()
        for col in ["pnl", "pnl_pct"]:
            if col in trades_display.columns:
                trades_display[col] = trades_display[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                )
        st.dataframe(trades_display, use_container_width=True, hide_index=True)
        total_pnl = trades_df["pnl"].sum() if "pnl" in trades_df.columns else 0
        st.caption(f"总盈亏: {total_pnl:,.0f} 元  |  交易次数: {len(trades_df)}")

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
