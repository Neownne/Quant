import json
import os
from datetime import date, timedelta

import streamlit as st

from app.utils.data_loader import (
    build_kline_chart,
    detect_asset_type,
    get_all_assets,
    get_latest_trade_date,
    load_ohlcv,
)
from data.db import init_db

FAVORITES_FILE = os.path.expanduser("~/.quant_chart_favorites.json")
DEFAULT_FAVORITES = {
    "大盘蓝筹": ["000001", "000002", "000858", "600036", "600519", "601318"],
    "科技成长": ["002415", "300750", "688981", "002230"],
    "ETF": ["510050", "510300", "159915", "588000"],
    "基金": ["000001", "000002"],
}


def load_favorites() -> dict[str, list[str]]:
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE) as f:
            return json.load(f)
    return DEFAULT_FAVORITES.copy()


def save_favorites(fav: dict) -> None:
    with open(FAVORITES_FILE, "w") as f:
        json.dump(fav, f, ensure_ascii=False)


st.set_page_config(page_title="K线图", page_icon="📊", layout="wide")
st.title("📊 K线图 & 技术指标（股票 · ETF · 基金）")

init_db()

assets = get_all_assets()
if assets.empty:
    st.error("数据表为空，请先运行 python -m data.sync")
    st.stop()

name_map = dict(zip(assets["code"], assets["name"]))
type_map = dict(zip(assets["code"], assets.get("type", [])))

# ---- 侧边栏 ----
with st.sidebar:
    st.header("选择资产")

    # 代码直输
    code_input = st.text_input(
        "输入代码", placeholder="000001 / 510050 / 159915",
        help="支持股票、ETF、基金代码，回车确认",
    )
    code_from_input = code_input.strip() if code_input else ""

    # 下拉搜索（全部资产）
    asset_options = []
    for _, r in assets.iterrows():
        at_label = {"stock": "股", "etf": "ETF", "fund": "基"}.get(r.get("type", ""), "")
        asset_options.append(f"{r['code']} [{at_label}] {r['name']}")
    selected = st.selectbox("或从列表搜索", asset_options, index=0)
    code_from_select = selected.split()[0] if selected else "000001"

    code = code_from_input if code_from_input else code_from_select
    at = detect_asset_type(code)
    at_label_full = {"stock": "股票", "etf": "ETF", "fund": "基金"}.get(at, "")

    # ---- 自选分组 ----
    st.divider()
    st.caption("自选分组")

    favorites = load_favorites()

    with st.expander("管理分组"):
        new_group = st.text_input("新建分组名称", placeholder="我的自选")
        if new_group and st.button("创建分组"):
            if new_group not in favorites:
                favorites[new_group] = []
                save_favorites(favorites)
                st.rerun()

        if favorites:
            del_group = st.selectbox("删除分组", [""] + list(favorites.keys()))
            if del_group and st.button("删除分组"):
                del favorites[del_group]
                save_favorites(favorites)
                st.rerun()

        add_to_group = st.selectbox("添加到分组", [""] + list(favorites.keys()))
        if add_to_group and st.button(f"将 {code} 加入「{add_to_group}」"):
            if code not in favorites[add_to_group]:
                favorites[add_to_group].append(code)
                save_favorites(favorites)
                st.rerun()

    # 分组快捷按钮
    for group_name, group_codes in favorites.items():
        if not group_codes:
            continue
        with st.expander(f"📁 {group_name} ({len(group_codes)}只)"):
            cols = st.columns(4)
            for i, c in enumerate(group_codes):
                n = name_map.get(c, "?")
                display = f"{c}\n{n[:6]}"
                if cols[i % 4].button(
                    display, key=f"fav_{group_name}_{c}",
                    use_container_width=True,
                ):
                    code = c
                    st.rerun()

    st.divider()

    # ---- 日期 & 指标 ----
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("起始日期", value=date.today() - timedelta(days=365))
    with col2:
        latest = get_latest_trade_date(code)
        default_end = date.fromisoformat(latest) if latest else date.today()
        end_date = st.date_input("截止日期", value=default_end)

    st.divider()
    st.caption("技术指标")
    ma_options = st.multiselect("MA 均线", [5, 10, 20, 60], default=[5, 20])
    ema_options = st.multiselect("EMA", [12, 26], default=[])
    show_macd = st.checkbox("MACD", value=True)
    show_rsi = st.checkbox("RSI (14)", value=False)
    show_bollinger = st.checkbox("Bollinger (20,2)", value=False)

    if st.button("🔄 刷新图表", use_container_width=True):
        load_ohlcv.clear()
        st.rerun()

# ---- 主区域 ----
stock_name = name_map.get(code, code)
st.caption(f"当前: {code} {stock_name}（{at_label_full}）")

start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")
df = load_ohlcv(code, start_str, end_str)

if df.empty:
    st.warning(f"{code} {stock_name}（{at_label_full}）在 {start_str} ~ {end_str} 区间无数据")
    st.stop()

indicators = {
    "ma_periods": ma_options if ma_options else None,
    "ema_periods": ema_options if ema_options else None,
    "macd": show_macd,
    "rsi": show_rsi,
    "bollinger": show_bollinger,
}

is_fund = (at == "fund")
fig = build_kline_chart(df, indicators, is_fund=is_fund)
st.plotly_chart(fig, use_container_width=True)

with st.expander("数据概览"):
    display_cols = ["trade_date"]
    if is_fund:
        display_cols += ["close", "high"]
        display_labels = {"close": "单位净值", "high": "累计净值"}
    else:
        display_cols += ["open", "high", "low", "close", "volume"]
        display_labels = {}
    avail_cols = [c for c in display_cols if c in df.columns]
    show_df = df[avail_cols].sort_values("trade_date", ascending=False)
    if display_labels:
        show_df = show_df.rename(columns=display_labels)
    st.dataframe(show_df, use_container_width=True, hide_index=True)
