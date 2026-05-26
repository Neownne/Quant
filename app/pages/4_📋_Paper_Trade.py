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
    # 尝试获取实时价格
    price_map = {}
    try:
        import akshare as ak
        spot = ak.stock_zh_a_spot_em()
        spot["代码"] = spot["代码"].astype(str)
        for _, p in pos.iterrows():
            code = str(p["code"])
            match = spot[spot["代码"] == code]
            if not match.empty:
                price_map[code] = float(match.iloc[0]["最新价"])
    except Exception:
        pass

    pos_display = pos.copy()
    pos_display["volume"] = pos_display["volume"].astype(int)
    pos_display["均价"] = pos_display["avg_cost"].apply(lambda x: f"{x:.2f}")
    pos_display["现价"] = pos_display["code"].apply(
        lambda c: f"{price_map.get(str(c), 0):.2f}" if price_map.get(str(c)) else "-"
    )
    pos_display["持仓成本"] = pos_display.apply(
        lambda r: r["volume"] * float(r["avg_cost"]), axis=1
    )
    pos_display["持仓市值"] = pos_display.apply(
        lambda r: r["volume"] * price_map.get(str(r["code"]), 0)
        if price_map.get(str(r["code"])) else 0, axis=1
    )
    pos_display["浮动盈亏"] = pos_display["持仓市值"] - pos_display["持仓成本"]
    pos_display["盈亏%"] = pos_display.apply(
        lambda r: f"{(r['浮动盈亏'] / r['持仓成本'] * 100):.2f}%"
        if r["持仓成本"] > 0 else "-", axis=1
    )
    pos_display["浮动盈亏"] = pos_display["浮动盈亏"].apply(lambda x: f"{x:,.0f}")
    pos_display["持仓成本"] = pos_display["持仓成本"].apply(lambda x: f"{x:,.0f}")
    pos_display["持仓市值"] = pos_display["持仓市值"].apply(lambda x: f"{x:,.0f}" if x > 0 else "-")

    st.dataframe(
        pos_display[["code", "volume", "均价", "现价", "持仓成本", "持仓市值", "浮动盈亏", "盈亏%"]],
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
    orders_display["日期"] = orders_display["order_time"].dt.date
    orders_display["price"] = orders_display["price"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    orders_display["amount"] = orders_display["amount"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")

    # 每日买卖汇总
    st.caption("每日汇总")
    daily_summary = orders_display.groupby(["日期", "direction"]).agg(
        笔数=("id", "count"), 总金额=("amount", lambda x: sum(float(v.replace(",", "")) for v in x))
    ).reset_index()
    st.dataframe(daily_summary, use_container_width=True, hide_index=True)

    # 详细订单
    show_cols = ["日期", "code", "direction", "price", "volume", "amount", "status"]
    if "note" in orders_display.columns:
        show_cols.append("note")
    st.caption(f"最近 {len(orders_display)} 笔委托")
    st.dataframe(
        orders_display[show_cols],
        use_container_width=True,
        hide_index=True,
    )

# ---- 已平仓交易（FIFO 匹配） ----
st.subheader("已平仓交易")
all_orders = get_orders(account_id)
if all_orders.empty:
    st.info("暂无已平仓交易")
else:
    all_orders["order_time"] = pd.to_datetime(all_orders["order_time"])
    completed = all_orders[all_orders["status"] == "filled"].sort_values("order_time")

    if completed.empty:
        st.info("暂无已成交订单")
    else:
        closed_trades = []
        for code, group in completed.groupby("code"):
            buys = []  # FIFO queue: [(price, volume, date)]
            for _, row in group.iterrows():
                vol = int(row["volume"])
                price = float(row["price"])
                if row["direction"] == "BUY":
                    buys.append((price, vol, row["order_time"]))
                else:
                    remaining = vol
                    while remaining > 0 and buys:
                        entry_price, entry_vol, entry_time = buys.pop(0)
                        matched = min(entry_vol, remaining)
                        pnl = (price - entry_price) * matched
                        cost = entry_price * matched
                        closed_trades.append({
                            "代码": code,
                            "买入日": entry_time.date(),
                            "卖出日": row["order_time"].date(),
                            "买入价": f"{entry_price:.2f}",
                            "卖出价": f"{price:.2f}",
                            "数量": matched,
                            "盈亏(元)": f"{pnl:,.0f}",
                            "盈亏%": f"{pnl / cost * 100:.2f}%" if cost > 0 else "-",
                        })
                        if entry_vol > matched:
                            buys.insert(0, (entry_price, entry_vol - matched, entry_time))
                        remaining -= matched

        if closed_trades:
            closed_df = pd.DataFrame(closed_trades)
            total_realized = sum(
                float(t["盈亏(元)"].replace(",", "")) for t in closed_trades
            )
            st.caption(f"共 {len(closed_df)} 笔已平仓交易，合计盈亏: {total_realized:,.0f} 元")
            st.dataframe(closed_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无已平仓交易（需先卖出持仓）")
