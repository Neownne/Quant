"""数据同步监控页面：数据质量概览 + 手动同步触发 + 覆盖度追踪。"""
from datetime import date

import pandas as pd
import streamlit as st

from data.db import get_engine, init_db
from data.sync import check_data_quality

st.set_page_config(page_title="数据监控", page_icon="📡", layout="wide")
st.title("📡 数据同步监控")

init_db()


@st.cache_data(ttl=60)
def get_sync_status() -> dict:
    engine = get_engine()
    try:
        result = check_data_quality(engine)
    finally:
        engine.dispose()
    return result


# ── 自动刷新 ──
if st.button("🔄 刷新数据质量报告"):
    get_sync_status.clear()
    st.rerun()

# ── 数据质量总览 ──
st.subheader("数据质量概览")
status_data = get_sync_status()

if not status_data:
    st.warning("无法获取数据质量信息")
    st.stop()

# Summary
ok = sum(1 for v in status_data.values() if v["status"] == "ok")
warn = sum(1 for v in status_data.values() if v["status"] in ("stale", "low_coverage"))
err = sum(1 for v in status_data.values() if v["status"] not in ("ok", "stale", "low_coverage"))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("✅ 正常", ok)
with col2:
    st.metric("⚠️ 需关注", warn)
with col3:
    st.metric("❌ 异常", err)

# Detail table
rows = []
status_icons = {"ok": "✅", "stale": "⚠️", "low_coverage": "⚠️", "missing": "❌"}
for table, info in status_data.items():
    icon = status_icons.get(info["status"], "❌")
    rows.append({
        "表名": table,
        "状态": f"{icon} {info['status']}",
        "最新日期": info.get("latest_date") or "-",
        "过期天数": info.get("stale_days", 0),
        "记录数": f"{info['n_records']:,}",
        "代码数": f"{info['n_codes']:,}",
    })

df_status = pd.DataFrame(rows)
st.dataframe(df_status, use_container_width=True, hide_index=True)

# Stale tables warning
stale_tables = [t for t, v in status_data.items() if v["status"] in ("stale", "missing")]
if stale_tables:
    st.warning(f"⚠️ 以下表数据需要同步: {', '.join(stale_tables)}")
    st.info("在终端运行同步命令或使用下方手动触发按钮")

# ── 手动同步触发 ──
st.divider()
st.subheader("手动同步触发")

sync_modes = {
    "index": "指数日线",
    "stock-daily": "股票日线",
    "daily-extra": "估值指标",
    "shareholder": "股东户数",
    "financial": "财务数据",
    "industry": "行业分类",
    "pledge": "股权质押",
}

cols = st.columns(len(sync_modes))
for i, (mode, label) in enumerate(sync_modes.items()):
    with cols[i]:
        if st.button(f"📥 {label}", key=f"sync_{mode}", help=f"立即同步 {label} 数据"):
            with st.spinner(f"正在同步 {label}..."):
                import subprocess
                import sys
                import os
                _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                result = subprocess.run(
                    [sys.executable, "-m", "data.sync", "--mode", mode],
                    capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=600,
                )
                if result.returncode == 0:
                    st.success(f"{label} 同步完成")
                    get_sync_status.clear()
                else:
                    st.error(f"{label} 同步失败")
                    if result.stderr:
                        st.code(result.stderr[-500:])

# ── 最近交易日覆盖度 ──
st.divider()
st.subheader("最近交易日覆盖度")

engine = get_engine()
try:
    with engine.connect() as conn:
        total_stocks = pd.read_sql("SELECT COUNT(*) as n FROM stock_basic", conn.conn if hasattr(conn, 'conn') else conn)["n"].iloc[0]
        latest = pd.read_sql(
            "SELECT MAX(trade_date) as dt FROM stock_daily",
            conn.conn if hasattr(conn, 'conn') else conn,
        )["dt"].iloc[0]
        if latest:
            latest_str = str(latest)[:10]
            covered = pd.read_sql(
                f"SELECT COUNT(DISTINCT code) as n FROM stock_daily WHERE trade_date = '{latest_str}'",
                conn.conn if hasattr(conn, 'conn') else conn,
            )["n"].iloc[0]
            pct = covered / max(total_stocks, 1) * 100

            st.caption(f"基准日期: {latest_str} | 总股票数: {total_stocks:,}")
            st.progress(min(pct / 100, 1.0), text=f"覆盖度: {covered:,} / {total_stocks:,} ({pct:.1f}%)")

            if pct < 50:
                st.error(f"股票日线覆盖度严重不足 ({pct:.1f}%)，请运行数据同步")
            elif pct < 80:
                st.warning(f"股票日线覆盖度偏低 ({pct:.1f}%)，建议运行增量同步")
            else:
                st.success(f"股票日线覆盖度良好 ({pct:.1f}%)")
finally:
    engine.dispose()
