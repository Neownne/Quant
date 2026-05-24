import pandas as pd
import streamlit as st
from sqlalchemy import text

from data.db import get_engine, init_db

st.set_page_config(page_title="模拟盘", page_icon="📋", layout="wide")
st.title("📋 模拟盘交易")

init_db()


@st.cache_data(ttl=5)
def get_accounts() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(text("SELECT * FROM paper_account ORDER BY id"), conn)
    engine.dispose()
    return df


@st.cache_data(ttl=5)
def get_positions(account_id: int) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT * FROM paper_positions WHERE account_id = :aid"),
            conn, params={"aid": account_id},
        )
    engine.dispose()
    return df


@st.cache_data(ttl=5)
def get_orders(account_id: int) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT * FROM paper_orders WHERE account_id = :aid ORDER BY order_time DESC LIMIT 100"),
            conn, params={"aid": account_id},
        )
    engine.dispose()
    return df


# ---- 侧边栏：账户管理 ----
with st.sidebar:
    st.header("模拟账户")

    with st.form("new_account"):
        acc_name = st.text_input("账户名称", placeholder="我的模拟盘")
        acc_capital = st.number_input("初始资金", min_value=10000, value=100000, step=10000)
        if st.form_submit_button("创建账户"):
            engine = get_engine()
            with engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO paper_account (name, initial_capital, cash) VALUES (:n, :c, :c)"),
                    {"n": acc_name, "c": acc_capital},
                )
            engine.dispose()
            get_accounts.clear()
            st.success(f"账户「{acc_name}」创建成功")
            st.rerun()

# ---- 主区域 ----
accounts = get_accounts()

if accounts.empty:
    st.info("暂无模拟账户，请在左侧创建")
    st.stop()

# 账户选择
acc_ids = accounts["id"].tolist()
acc_labels = [f"#{r['id']} {r['name']} (¥{r['initial_capital']:,.0f})" for _, r in accounts.iterrows()]
if not acc_labels:
    st.stop()

default_idx = 0
if "paper_account_idx" in st.session_state:
    if st.session_state.paper_account_idx < len(acc_labels):
        default_idx = st.session_state.paper_account_idx
selected_label = st.selectbox("选择账户", acc_labels, index=default_idx)
if selected_label is None or selected_label not in acc_labels:
    selected_label = acc_labels[0]
selected_idx = acc_labels.index(selected_label)
st.session_state.paper_account_idx = selected_idx
account = accounts.iloc[selected_idx]
account_id = int(account["id"])

# ---- 账户概览 ----
st.subheader("账户概览")
pos = get_positions(account_id)
cash = float(account["cash"])

# 计算持仓市值
positions_value = 0.0
if not pos.empty:
    import akshare as ak
    try:
        spot = ak.stock_zh_a_spot_em()
        spot["代码"] = spot["代码"].astype(str)
        for _, p in pos.iterrows():
            code = str(p["code"])
            match = spot[spot["代码"] == code]
            if not match.empty:
                positions_value += float(match.iloc[0]["最新价"]) * int(p["volume"])
    except Exception:
        pass

total_assets = cash + positions_value
total_pnl = total_assets - float(account["initial_capital"])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("总资产", f"{total_assets:,.0f} 元")
with col2:
    st.metric("现金", f"{cash:,.0f} 元")
with col3:
    st.metric("持仓市值", f"{positions_value:,.0f} 元")
with col4:
    st.metric("累计盈亏", f"{total_pnl:,.0f} 元",
              delta=f"{total_pnl / float(account['initial_capital']) * 100:.2f}%")

# ---- 持仓明细 ----
st.subheader("当前持仓")
if pos.empty:
    st.info("暂无持仓")
else:
    pos_display = pos.copy()
    pos_display["volume"] = pos_display["volume"].astype(int)
    pos_display["avg_cost"] = pos_display["avg_cost"].apply(lambda x: f"{x:.2f}")
    st.dataframe(
        pos_display[["code", "volume", "avg_cost"]],
        use_container_width=True,
        hide_index=True,
    )

# ---- 委托历史 ----
st.subheader("最近委托")
orders = get_orders(account_id)
if orders.empty:
    st.info("暂无委托记录")
else:
    orders_display = orders.copy()
    orders_display["order_time"] = pd.to_datetime(orders_display["order_time"])
    orders_display["price"] = orders_display["price"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    orders_display["amount"] = orders_display["amount"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
    st.dataframe(
        orders_display[["code", "direction", "price", "volume", "amount", "status", "order_time"]],
        use_container_width=True,
        hide_index=True,
    )
