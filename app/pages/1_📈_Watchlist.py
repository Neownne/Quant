import json
import os

import pandas as pd
import streamlit as st

from app.utils.data_loader import (
    get_all_assets,
    get_latest_daily_batch,
    get_realtime_quotes,
)
from data.db import init_db

WATCHLIST_FILE = os.path.expanduser("~/.quant_watchlist.json")
DEFAULT_WATCHLIST = ["000001", "510050", "159915", "000002", "600519", "300750"]


def load_watchlist() -> list[str]:
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    return DEFAULT_WATCHLIST.copy()


def save_watchlist(codes: list[str]) -> None:
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(codes, f)


st.set_page_config(page_title="自选股实时报价", page_icon="📈", layout="wide")
st.title("📈 自选股实时报价（股票 · ETF · 基金）")

init_db()

# ---- 侧边栏 ----
with st.sidebar:
    st.header("管理自选")

    add_input = st.text_input(
        "添加代码（逗号分隔）",
        placeholder="000001,510050,159915",
        help="支持股票、ETF、基金代码",
    )
    if st.button("添加", use_container_width=True):
        watchlist = load_watchlist()
        new_codes = [c.strip() for c in add_input.replace("，", ",").split(",") if c.strip()]
        for c in new_codes:
            if c not in watchlist:
                watchlist.append(c)
        save_watchlist(watchlist)
        st.rerun()

    watchlist = load_watchlist()

    if watchlist:
        to_remove = st.selectbox("选择要删除的代码", [""] + watchlist)
        if to_remove and st.button("删除选中", use_container_width=True):
            watchlist.remove(to_remove)
            save_watchlist(watchlist)
            st.rerun()

    st.divider()
    if st.button("🔄 手动刷新", use_container_width=True):
        get_realtime_quotes.clear()
        get_latest_daily_batch.clear()
        st.rerun()

# ---- 主区域 ----
quotes = get_realtime_quotes(watchlist)
is_realtime = not quotes.empty

if not is_realtime and watchlist:
    quotes = get_latest_daily_batch(watchlist)

# 构建名称映射（股票+ETF+基金）
all_assets = get_all_assets()
name_map = {}
type_map = {}
if not all_assets.empty:
    for _, r in all_assets.iterrows():
        name_map[r["code"]] = r["name"]
        type_map[r["code"]] = r.get("type", "")

rows = []
for code in watchlist:
    q = quotes[quotes["code"] == code] if not quotes.empty else pd.DataFrame()
    at = type_map.get(code, "")
    type_label = {"stock": "股", "etf": "ETF", "fund": "基金"}.get(at, "")

    if not q.empty:
        row = q.iloc[0]
        pre_close = row.get("open")
        price = row.get("close", row.get("price"))
        change_pct = row.get("change_pct")
        if change_pct is None and not is_realtime and pre_close and price:
            change_pct = round((price - pre_close) / pre_close * 100, 2)

        vol = row.get("volume")
        amt = row.get("amount")
        rows.append({
            "类型": type_label,
            "代码": row["code"],
            "名称": name_map.get(row["code"], row.get("name", "未知")),
            "最新价": f"{price:.2f}" if price else "-",
            "涨跌幅(%)": f"{change_pct:+.2f}%" if change_pct is not None else "-",
            "成交量": f"{vol:,.0f}" if vol is not None and not pd.isna(vol) else "-",
            "成交额": f"{amt:,.0f}" if amt is not None and not pd.isna(amt) else "-",
        })
    else:
        rows.append({
            "类型": type_label,
            "代码": code,
            "名称": name_map.get(code, "未知"),
            "最新价": "-",
            "涨跌幅(%)": "-",
            "成交量": "-",
            "成交额": "-",
        })

df = pd.DataFrame(rows)

st.dataframe(
    df,
    column_config={
        "涨跌幅(%)": st.column_config.NumberColumn("涨跌幅(%)", format="%.2f%%"),
    },
    use_container_width=True,
    hide_index=True,
    height=min(38 * len(rows) + 38, 600),
)

if is_realtime:
    st.caption("📡 实时行情（股票+ETF）| 点击侧边栏🔄按钮手动刷新")
else:
    st.caption("📋 最近交易日数据（非交易时段，含股票/ETF/基金净值）| 点击侧边栏🔄按钮手动刷新")
