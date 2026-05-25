"""实盘数据录制面板 —— 分钟K线 / 逐笔成交，支持股票池 & 自选标的。"""

import time

import pandas as pd
import streamlit as st

from app.utils.stock_pools import list_pools, load_and_execute_pool, get_pool_data
from data.db import init_db, get_engine
from data.recorder import RecorderSession, is_trading_time, is_trading_day, trading_status

st.set_page_config(page_title="数据录制", page_icon="🔴", layout="wide")
st.title("🔴 实盘数据录制")

init_db()

# ---- 初始化 session state ----
if "rec_session" not in st.session_state:
    st.session_state.rec_session = None
if "rec_logs" not in st.session_state:
    st.session_state.rec_logs = []

# ---- 侧边栏：配置 ----
with st.sidebar:
    st.header("录制设置")

    mode = st.selectbox("录制模式", ["minute", "tick"], format_func=lambda x: {
        "minute": "分钟K线（交易时段轮询）", "tick": "逐笔成交（收盘后抓取）",
    }[x])

    if mode == "minute":
        period = st.selectbox("K线周期", ["1", "5", "15", "30", "60"], format_func=lambda x: f"{x}分钟")
    else:
        period = "1"

    st.divider()
    st.subheader("标的选择")

    source = st.radio("选择方式", ["股票池", "手动选择"], horizontal=True)

    if source == "股票池":
        pools = list_pools()
        if not pools:
            st.warning("暂无股票池，请先在「股票池」页面创建")
            selected_codes = []
            selected_pool = ""
        else:
            selected_pool = st.selectbox("选择股票池", pools)
            if st.button("🔍 预览"):
                engine = get_engine()
                try:
                    basic, extra, shareholder = get_pool_data(engine)
                    codes = load_and_execute_pool(selected_pool, basic, extra, shareholder)
                    st.info(f"{len(codes)} 只: {', '.join(codes[:20])}{'...' if len(codes) > 20 else ''}")
                finally:
                    engine.dispose()
    else:
        selected_pool = ""
        from app.utils.data_loader import get_all_assets
        all_assets = get_all_assets()
        tradeable = all_assets[all_assets["type"].isin(["stock"])]
        options = [f"{r['code']} {r['name']}" for _, r in tradeable.iterrows()]
        selected = st.multiselect("手动选择标的", options, max_selections=50)
        st.caption(f"已选 {len(selected)} 只（最多 50）")

    st.divider()

    # 交易状态指示
    status = trading_status()
    status_color = {
        "交易中": "🟢", "午休": "🟡", "已收盘": "🔴",
        "盘前": "🟠",
    }
    st.caption(f"{status_color.get(status, '⚪')} 状态: {status}")

    session = st.session_state.rec_session
    is_recording = session is not None and session.is_running

    col1, col2 = st.columns(2)
    with col1:
        has_targets = (
            (source == "股票池" and pools and selected_pool) or
            (source == "手动选择" and selected)
        )
        if st.button("▶️ 开始录制", use_container_width=True, type="primary", disabled=is_recording or not has_targets):
            # 确定标的列表
            if source == "股票池":
                engine = get_engine()
                try:
                    basic, extra, shareholder = get_pool_data(engine)
                    codes = load_and_execute_pool(selected_pool, basic, extra, shareholder)
                finally:
                    engine.dispose()
            else:
                codes = [s.split()[0] for s in selected] if selected else []

            if not codes:
                st.error("没有选中任何标的")
            else:
                st.session_state.rec_session = RecorderSession(
                    watchlist=codes, mode=mode, period=period,
                )
                st.session_state.rec_session.start()
                st.session_state.rec_logs = []
                st.rerun()

    with col2:
        if st.button("⏹️ 停止录制", use_container_width=True, disabled=not is_recording):
            if session:
                session.stop()
                # 收集剩余日志
                st.session_state.rec_logs.extend(session.get_logs())
                st.session_state.rec_session = None
            st.rerun()

# ---- 主区域 ----
if not is_recording:
    st.info("配置好标的后点击「开始录制」")

    # 显示已有数据概况
    engine = get_engine()
    try:
        col1, col2 = st.columns(2)
        with col1:
            n_min = pd.read_sql("SELECT COUNT(*) as c FROM stock_minute", engine).iloc[0, 0]
            n_tick = pd.read_sql("SELECT COUNT(*) as c FROM stock_tick", engine).iloc[0, 0]
            st.metric("stock_minute", f"{n_min:,} 条")
            st.metric("stock_tick", f"{n_tick:,} 条")
        with col2:
            if n_min > 0:
                recent = pd.read_sql(
                    "SELECT trade_time FROM stock_minute ORDER BY trade_time DESC LIMIT 1", engine
                )
                st.caption(f"分钟数据最新: {recent.iloc[0, 0] if not recent.empty else 'N/A'}")
            if n_tick > 0:
                recent = pd.read_sql(
                    "SELECT trade_time FROM stock_tick ORDER BY trade_time DESC LIMIT 1", engine
                )
                st.caption(f"逐笔数据最新: {recent.iloc[0, 0] if not recent.empty else 'N/A'}")
    finally:
        engine.dispose()
else:
    # ---- 录制中 ----
    # 拉取新日志
    new_logs = session.get_logs()
    if new_logs:
        st.session_state.rec_logs.extend(new_logs)

    st.success(f"🔴 录制中 — {len(session.watchlist)} 只标的, 模式: {mode}")

    # 标的列表
    with st.expander(f"标的列表（{len(session.watchlist)} 只）"):
        cols = st.columns(8)
        for i, c in enumerate(session.watchlist):
            cols[i % 8].code(c)

    # 统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已写日志", len(st.session_state.rec_logs))
    with col2:
        errors = sum(1 for e in st.session_state.rec_logs if e["level"] == "error")
        st.metric("错误", errors)
    with col3:
        successes = sum(1 for e in st.session_state.rec_logs if e["level"] == "success")
        st.metric("成功轮次", successes)

    # 实时日志
    st.subheader("运行日志")
    visible = st.session_state.rec_logs[-200:]
    html = '<div style="font-family:monospace;font-size:13px;line-height:1.6;max-height:500px;overflow-y:auto;background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:6px">'
    for entry in reversed(visible):
        color = {"error": "#f44747", "warning": "#cca700", "success": "#6a9955", "info": "#9cdcfe"}.get(entry["level"], "#d4d4d4")
        html += f'<div><span style="color:#808080">[{entry["time"]}]</span> <span style="color:{color}">{entry["msg"]}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    if st.button("🔄 刷新日志"):
        st.rerun()

    # 自动刷新：2 秒后重载页面
    time.sleep(2)
    st.rerun()
