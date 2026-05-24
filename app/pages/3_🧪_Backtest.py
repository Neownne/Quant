import hashlib
import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app.utils.backtest_runner import run_backtest
from app.utils.data_loader import get_stock_list, load_ohlcv
from data.db import init_db
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
    st.subheader("资金 & 费用")
    initial_cash = st.number_input("初始资金", min_value=10000, value=200000, step=10000)
    commission = st.number_input("佣金费率（默认万0.85）", min_value=0.0, value=0.000085, step=0.00001, format="%.5f")
    slippage = st.number_input("滑点（元）", min_value=0.0, value=0.01, step=0.01, format="%.2f")

    st.divider()
    st.subheader("标的选择")
    from app.utils.data_loader import get_all_assets
    all_assets = get_all_assets()
    # 只保留有 OHLCV 的资产（股票 + ETF），排除基金
    tradeable = all_assets[all_assets["type"].isin(["stock", "etf"])]
    code_options = [f"{r['code']} {r['name']}" for _, r in tradeable.iterrows()] if not tradeable.empty else ["000001"]
    selected = st.selectbox("股票/ETF", code_options)
    code = selected.split()[0] if selected else "000001"

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

# 加载数据
start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")
with st.spinner(f"加载 {code} 数据中 ..."):
    df = load_ohlcv(code, start_str, end_str)

if df.empty:
    st.error(f"{code} 在 {start_str} ~ {end_str} 区间无数据")
    st.stop()

st.caption(f"数据范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}，共 {len(df)} 条")

# 运行回测
with st.spinner("回测运行中 ..."):
    result = run_backtest(
        strategy_class=strategy_class,
        df=df,
        strategy_params=strategy_params,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
    )

metrics = result["metrics"]
equity_curve = result["equity_curve"]
trades_df = result["trades"]

# ---- 指标卡片 ----
st.subheader("核心指标")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_ret = metrics.get("total_return", 0) * 100
    st.metric("总收益率", f"{total_ret:.2f}%")
with col2:
    annual_ret = metrics.get("annual_return", 0) * 100
    st.metric("年化收益", f"{annual_ret:.2f}%")
with col3:
    max_dd = metrics.get("max_drawdown", 0)
    st.metric("最大回撤", f"{max_dd:.2f}%")
with col4:
    sharpe = metrics.get("sharpe_ratio", 0)
    st.metric("夏普比率", f"{sharpe:.2f}")
with col5:
    wr = metrics.get("win_rate", 0) * 100
    st.metric("胜率", f"{wr:.1f}%", delta=None)

st.metric("最终权益", f"{metrics.get('final_value', 0):,.0f} 元")

# ---- 权益曲线 ----
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

# ---- 交易明细 ----
st.subheader("交易明细")
if trades_df.empty:
    st.info("无交易记录")
else:
    trades_display = trades_df.copy()
    for col in ["pnl", "pnl_pct"]:
        if col in trades_display.columns:
            trades_display[col] = trades_display[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "-"
            )
    st.dataframe(trades_display, use_container_width=True, hide_index=True)

    total_pnl = trades_df["pnl"].sum() if "pnl" in trades_df.columns else 0
    st.caption(f"总盈亏: {total_pnl:,.0f} 元  |  交易次数: {len(trades_df)}")
