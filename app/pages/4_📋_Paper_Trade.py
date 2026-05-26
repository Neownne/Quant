import json
from datetime import date

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
def get_paper_daily_pnl(account_id: int) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT * FROM paper_daily_pnl WHERE account_id = :aid ORDER BY trade_date"),
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

    st.divider()
    st.subheader("运行模拟盘引擎")

    # 账户选择
    accts = get_accounts()
    if not accts.empty:
        acc_labels_all = [f"#{r['id']} {r['name']}" for _, r in accts.iterrows()]
        acc_idx = st.selectbox("选择账户", range(len(acc_labels_all)),
                               format_func=lambda i: acc_labels_all[i],
                               key="paper_run_account_idx")
        sel_account = accts.iloc[acc_idx]
        run_account_id = int(sel_account["id"])
        run_strategy_type = sel_account.get("strategy_type", "ml") or "ml"
        run_strategy_name = sel_account.get("strategy_name", "") or ""

        if run_strategy_type == "static":
            st.caption(f"策略: {run_strategy_name} (静态)")
        else:
            st.caption("策略: ML 集成 (XGBoost+LightGBM)")

        if st.button("🚀 运行模拟盘", use_container_width=True,
                     help="基于选中账户的策略类型运行模拟盘"):

            if run_strategy_type == "static":
                # ── 静态策略回放 ──
                with st.spinner(f"正在加载数据并回放 {run_strategy_name} 策略..."):
                    try:
                        import sys as _sys, os as _os
                        _PROJECT_ROOT = _os.path.dirname(
                            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                        )
                        if _PROJECT_ROOT not in _sys.path:
                            _sys.path.insert(0, _PROJECT_ROOT)

                        from strategies import get_all_strategies
                        from portfolio.paper_engine import StaticPaperEngine
                        from app.utils.account_manager import get_account
                        import json as _json

                        acc_info = get_account(run_account_id)
                        if not acc_info:
                            st.error("账户不存在")
                            st.stop()

                        sp = acc_info.get("strategy_params", {})
                        if isinstance(sp, str):
                            sp = _json.loads(sp)

                        all_s = get_all_strategies()
                        strat_cls = None
                        for name, cls in all_s.items():
                            if cls.__name__ == run_strategy_name:
                                strat_cls = cls
                                break

                        if strat_cls is None:
                            st.error(f"未找到策略类: {run_strategy_name}")
                            st.stop()

                        engine = get_engine()
                        codes = pd.read_sql(
                            "SELECT code FROM stock_basic WHERE is_st = FALSE "
                            "AND list_date <= CURRENT_DATE - INTERVAL '60 days' "
                            "ORDER BY code LIMIT 50",
                            engine,
                        )["code"].tolist()
                        engine.dispose()

                        static_engine = StaticPaperEngine(
                            account_id=run_account_id,
                            strategy_class=strat_cls,
                            strategy_params=sp,
                            codes=codes,
                            initial_capital=float(acc_info["initial_capital"]),
                            commission=float(acc_info.get("commission_rate", 0.00009)),
                            stamp_duty=float(acc_info.get("stamp_duty_rate", 0.0005)),
                            slippage=float(acc_info.get("slippage", 0.01)),
                            use_market_filter=bool(acc_info.get("use_market_filter", True)),
                        )
                        res = static_engine.run_replay("20180101", date.today().strftime("%Y%m%d"))
                        get_positions.clear()
                        get_orders.clear()
                        st.success(
                            f"静态策略回放完成！股票: {res['n_stocks']}只 | "
                            f"信号: {res['n_signals']}条 | "
                            f"持仓: {res['n_positions']}只 | "
                            f"现金: ¥{res['final_cash']:,.0f}"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"静态策略回放失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            else:
                # ── ML 策略 ──
                with st.spinner("正在加载数据并训练模型（约3-5分钟）..."):
                    try:
                        import sys as _sys, os as _os
                        _PROJECT_ROOT = _os.path.dirname(
                            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                        )
                        if _PROJECT_ROOT not in _sys.path:
                            _sys.path.insert(0, _PROJECT_ROOT)

                        from factors import ALL_FACTORS
                        from models.dataset import build_factor_dataset
                        from models.trainer import walk_forward_train_ensemble
                        from factors.screening import filter_factors_by_ic, select_orthogonal_factors
                        from portfolio.paper_engine import PaperEngine

                        engine = get_engine()

                        codes = pd.read_sql(
                            "SELECT code FROM stock_basic WHERE is_st = FALSE "
                            "AND list_date <= CURRENT_DATE - INTERVAL '60 days' "
                            "ORDER BY code LIMIT 500",
                            engine,
                        )
                        if codes.empty:
                            st.error("候选池为空，请先同步股票基本信息")
                            st.stop()

                        code_list = ",".join([f"'{c}'" for c in codes["code"].tolist()])
                        ohlcv = pd.read_sql(
                            f"SELECT code, trade_date, open, high, low, close, volume, amount, turnover "
                            f"FROM stock_daily WHERE code IN ({code_list}) "
                            f"AND trade_date >= CURRENT_DATE - INTERVAL '5 years' "
                            f"ORDER BY code, trade_date",
                            engine,
                        )
                        index_ohlcv = pd.read_sql(
                            "SELECT trade_date, close FROM index_daily "
                            "WHERE code = '000001' "
                            "AND trade_date >= CURRENT_DATE - INTERVAL '5 years' "
                            "ORDER BY trade_date",
                            engine,
                        )
                        engine.dispose()

                        if ohlcv.empty:
                            st.error("无行情数据，请先同步股票日线")
                            st.stop()

                        factor_names = list(ALL_FACTORS.keys())
                        dataset = build_factor_dataset(ohlcv, factor_names, label_mode="binary")
                        pv_names = [f for f in factor_names if not f.startswith("fin_")]
                        filtered = filter_factors_by_ic(dataset, pv_names, ret_col="ret_1d")
                        selected = select_orthogonal_factors(dataset, filtered, threshold=0.7)

                        results = walk_forward_train_ensemble(
                            dataset, selected, train_years=3, val_years=1,
                        )
                        if not results:
                            st.error("训练失败，无有效Walk-forward窗口")
                            st.stop()

                        latest_result = results[-1]
                        predictor = latest_result["ensemble"]

                        paper = PaperEngine(
                            account_id=run_account_id,
                            predictor=predictor,
                            factor_names=selected,
                            top_n=15,
                            rebalance_mode="ndrop",
                            ndrop_n=2,
                        )
                        latest_date = dataset["trade_date"].max()
                        today_factors = dataset[dataset["trade_date"] == latest_date]

                        industry_map = {}
                        engine = get_engine()
                        try:
                            ind_df = pd.read_sql(
                                "SELECT code, industry_sw1 FROM stock_industry", engine,
                            )
                            industry_map = dict(zip(ind_df["code"], ind_df["industry_sw1"]))
                        except Exception:
                            pass
                        finally:
                            engine.dispose()

                        res = paper.run_daily(
                            trade_date=latest_date,
                            factor_df=today_factors,
                            ohlcv_data=ohlcv,
                            industry_map=industry_map,
                            index_ohlcv=index_ohlcv if not index_ohlcv.empty else None,
                        )

                        if res is None:
                            st.error("模拟盘执行失败")
                        else:
                            get_positions.clear()
                            get_orders.clear()
                            st.success(
                                f"模拟盘执行完成！日期: {latest_date.date()} | "
                                f"选股: {res['n_selected']}只 | "
                                f"买单: {res['n_buy_orders']}笔 | "
                                f"卖单: {res['n_sell_orders']}笔 | "
                                f"总资产: {res['total_value']:,.0f}元"
                            )
                            if res.get("crash_warning"):
                                st.warning("⚠️ 指数崩盘触发，已清仓")
                            if res.get("stop_losses"):
                                st.warning(f"⚠️ 止损触发: {res['stop_losses']}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"模拟盘执行失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# ---- 主区域 ----
accounts = get_accounts()

if accounts.empty:
    st.info("暂无模拟账户，请在左侧创建")
    st.stop()

# 使用侧边栏选择的账户
if "paper_run_account_idx" in st.session_state:
    acc_idx = st.session_state.paper_run_account_idx
    if acc_idx >= len(accounts):
        acc_idx = 0
else:
    acc_idx = 0
account = accounts.iloc[acc_idx]
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
              delta=f"{total_pnl / max(float(account['initial_capital']), 1) * 100:.2f}%")

# ---- 权益曲线 + 策略信息 ----
st.divider()
ecol1, ecol2 = st.columns([2, 1])

with ecol1:
    st.subheader("📈 权益曲线")
    pnl_data = get_paper_daily_pnl(account_id)
    if not pnl_data.empty:
        import plotly.graph_objects as go
        pnl_data["trade_date"] = pd.to_datetime(pnl_data["trade_date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pnl_data["trade_date"], y=pnl_data["total_value"],
            mode="lines", name="总资产",
            fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
            line=dict(color="#2196f3", width=2),
        ))
        fig.add_hline(y=float(account["initial_capital"]), line_dash="dash",
                      line_color="gray", annotation_text="初始资金")
        # 回撤填充
        if "drawdown" in pnl_data.columns:
            peak = pnl_data["total_value"].expanding().max()
            dd_area = pnl_data.copy()
            dd_area["drawdown_val"] = peak - pnl_data["total_value"]
            fig.add_trace(go.Scatter(
                x=pnl_data["trade_date"], y=peak,
                mode="lines", name="峰值",
                line=dict(width=0.5, color="rgba(255,0,0,0.3)"),
                fill="tonexty", fillcolor="rgba(255,0,0,0.05)",
            ))
        fig.update_layout(
            height=350, template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无净值记录，运行模拟盘后生成")

with ecol2:
    st.subheader("⚙️ 策略信息")
    stype = account.get("strategy_type") or "ml"
    sname = account.get("strategy_name") or ""
    sp = account.get("strategy_params") or {}
    if isinstance(sp, str):
        try:
            sp = json.loads(sp)
        except Exception:
            sp = {}

    if stype == "static" and sname:
        st.caption(f"**类型**: 静态策略")
        st.caption(f"**策略**: {sname}")
        if sp:
            st.caption("**参数**:")
            for k, v in sp.items():
                st.caption(f"　{k}: {v}")
    elif stype == "ml" or not sname:
        st.caption("**类型**: ML 集成")
        st.caption("**模型**: XGBoost + LightGBM")
        st.caption("**选股**: Top-15 NDrop")
        st.caption("**训练**: Walk-Forward 3+1年")

    st.caption(f"**佣金**: {float(account.get('commission_rate', 0.00009))*10000:.1f}‱")
    st.caption(f"**印花税**: {float(account.get('stamp_duty_rate', 0.0005))*10000:.1f}‱")
    st.caption(f"**滑点**: {float(account.get('slippage', 0.01)):.2f} 元/股")
    st.caption(f"**大盘过滤器**: {'开启' if account.get('use_market_filter', True) else '关闭'}")

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

# ---- 持仓分布 ----
if not pos.empty and price_map:
    st.subheader("📊 持仓分布")
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        # 行业分布饼图
        engine = get_engine()
        try:
            ind_df = pd.read_sql("SELECT code, industry_sw1 FROM stock_industry", engine)
            ind_map = dict(zip(ind_df["code"], ind_df["industry_sw1"]))
        except Exception:
            ind_map = {}
        engine.dispose()

        pos_codes = pos["code"].tolist()
        ind_dist: dict[str, float] = {}
        for _, p_row in pos.iterrows():
            pc = str(p_row["code"])
            val = int(p_row["volume"]) * price_map.get(pc, float(p_row["avg_cost"]))
            ind = ind_map.get(pc, "其他")
            ind_dist[ind] = ind_dist.get(ind, 0) + val

        if ind_dist:
            import plotly.graph_objects as go
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(ind_dist.keys()),
                values=list(ind_dist.values()),
                hole=0.4,
                textinfo="label+percent",
            )])
            fig_pie.update_layout(
                height=300, margin=dict(l=0, r=0, t=5, b=0),
                showlegend=False,
            )
            st.caption("行业分布")
            st.plotly_chart(fig_pie, use_container_width=True)

    with dcol2:
        # 个股权重 Top-10
        weights = []
        total_mv = sum(
            int(p_row["volume"]) * price_map.get(str(p_row["code"]), float(p_row["avg_cost"]))
            for _, p_row in pos.iterrows()
        )
        for _, p_row in pos.iterrows():
            pc = str(p_row["code"])
            mv = int(p_row["volume"]) * price_map.get(pc, float(p_row["avg_cost"]))
            if total_mv > 0:
                weights.append({"code": pc, "weight": mv / total_mv * 100, "value": mv})
        weights.sort(key=lambda x: x["weight"], reverse=True)
        top10 = weights[:10]

        if top10:
            import plotly.graph_objects as go
            fig_bar = go.Figure(data=[go.Bar(
                x=[w["code"] for w in top10],
                y=[w["weight"] for w in top10],
                text=[f"{w['weight']:.1f}%" for w in top10],
                textposition="outside",
                marker_color="#2196f3",
            )])
            fig_bar.update_layout(
                height=300, template="plotly_white",
                margin=dict(l=0, r=0, t=5, b=0),
                yaxis_title="权重 (%)",
            )
            st.caption("个股权重 Top-10")
            st.plotly_chart(fig_bar, use_container_width=True)

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
