"""ML 策略监控页面：因子IC、模型表现、当日信号、模拟盘净值。"""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.db import get_engine, init_db

st.set_page_config(page_title="ML策略监控", page_icon="📊", layout="wide")
st.title("📊 ML 策略监控")

init_db()

# ── 缓存数据加载 ─────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_factor_ic_summary() -> pd.DataFrame | None:
    """计算最近一年的因子 IC 汇总。"""
    try:
        from factors.monitor import compute_ic_series, compute_ic_summary
        from factors import ALL_FACTORS
        from models.dataset import build_factor_dataset

        engine = get_engine()
        codes = pd.read_sql("SELECT code FROM stock_basic LIMIT 200", engine)
        code_list = ",".join([f"'{c}'" for c in codes["code"].tolist()])
        ohlcv = pd.read_sql(
            f"SELECT code, trade_date, open, high, low, close, volume, amount, turnover "
            f"FROM stock_daily WHERE code IN ({code_list}) "
            f"AND trade_date >= CURRENT_DATE - INTERVAL '400 days' ORDER BY code, trade_date",
            engine,
        )
        engine.dispose()

        if len(ohlcv) < 10000:
            return None

        factor_names = list(ALL_FACTORS.keys())
        dataset = build_factor_dataset(ohlcv, factor_names, label_mode="binary")
        valid = dataset[["ret_1d", "trade_date"] + factor_names].dropna(subset=["ret_1d", "trade_date"])
        if len(valid) < 100:
            return None

        ic_df = compute_ic_series(valid, factor_names, ret_col="ret_1d")
        return compute_ic_summary(ic_df)
    except Exception as e:
        st.warning(f"IC 数据加载失败: {e}")
        return None


@st.cache_data(ttl=1800)
def load_paper_accounts() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(text("SELECT * FROM paper_account ORDER BY id"), conn)
    engine.dispose()
    return df


@st.cache_data(ttl=30)
def load_paper_pnl(account_id: int) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT * FROM paper_daily_pnl WHERE account_id = :aid ORDER BY trade_date"),
            conn, params={"aid": account_id},
        )
    engine.dispose()
    return df


@st.cache_data(ttl=30)
def load_paper_positions(account_id: int) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT * FROM paper_positions WHERE account_id = :aid AND volume > 0"),
            conn, params={"aid": account_id},
        )
    engine.dispose()
    return df


# ── Tab 布局 ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 因子IC看板", "🧠 模型表现", "🎯 当日信号", "💰 模拟盘净值"]
)

# ── Tab 1: 因子IC看板 ───────────────────────────────────

with tab1:
    st.subheader("因子 RankIC 汇总（近一年）")

    ic_summary = load_factor_ic_summary()

    if ic_summary is None or ic_summary.empty:
        st.info("暂无因子 IC 数据，请确认数据同步已完成")
    else:
        # IC 柱状图
        top_n = st.slider("展示因子数", 10, min(50, len(ic_summary)), 20, key="ic_top_n")
        ic_display = ic_summary.copy()
        ic_display["ic_abs"] = ic_display["ic_mean"].abs()
        ic_display = ic_display.sort_values("ic_abs", ascending=False).head(top_n)

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ic_display.index,
            y=ic_display["ic_mean"],
            marker=dict(
                color=[
                    "green" if v > 0 else "red" for v in ic_display["ic_mean"]
                ],
            ),
            error_y=dict(
                type="data", array=ic_display["ic_std"].values,
                visible=True, color="gray",
            ),
        ))
        fig.add_hline(y=0.02, line_dash="dash", line_color="gray", annotation_text="|IC|=0.02")
        fig.add_hline(y=-0.02, line_dash="dash", line_color="gray")
        fig.update_layout(
            height=400, template="plotly_white",
            xaxis_title="因子", yaxis_title="IC Mean ± Std",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # IC 汇总表
        col1, col2, col3 = st.columns(3)
        with col1:
            passed = (ic_summary["ic_mean"].abs() > 0.02).sum()
            st.metric("|IC|>0.02 因子数", f"{passed}/{len(ic_summary)}")
        with col2:
            st.metric("平均 |IC|", f"{ic_summary['ic_mean'].abs().mean():.4f}")
        with col3:
            st.metric("IC 为正因子数", f"{(ic_summary['ic_mean'] > 0).sum()}/{len(ic_summary)}")

# ── Tab 2: 模型表现 ─────────────────────────────────────

with tab2:
    st.subheader("Walk-Forward 训练评估")

    st.info(
        "运行下方命令查看最新模型表现：\n\n"
        "```bash\n"
        "python scripts/run_ml_backtest.py --start 20180101 --end 20250101\n"
        "python scripts/run_ml_backtest.py --start 20180101 --end 20250101 --regime\n"
        "```"
    )

    if st.button("🧪 快速评估（小范围）", help="使用近3年数据跑一次快速评估"):
        with st.spinner("正在运行 walk-forward 训练评估..."):
            import subprocess
            result = subprocess.run(
                [
                    sys.executable, os.path.join(_PROJECT_ROOT, "scripts", "run_ml_backtest.py"),
                    "--start", "20220101", "--end", "20250101",
                ],
                capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=300,
            )
            st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
            if result.stderr:
                st.caption(f"stderr (尾部): {result.stderr[-500:]}")

# ── Tab 3: 当日信号 ─────────────────────────────────────

with tab3:
    st.subheader("当日 ML 选股信号")

    if st.button("🔮 生成当日预测", help="基于最新因子数据运行 ML 预测"):
        with st.spinner("正在加载模型和因子数据..."):
            try:
                from factors import ALL_FACTORS
                from models.dataset import build_factor_dataset
                from models.trainer import walk_forward_train_ensemble

                engine = get_engine()
                codes = pd.read_sql("SELECT code FROM stock_basic LIMIT 200", engine)
                code_list = ",".join([f"'{c}'" for c in codes["code"].tolist()])

                ohlcv = pd.read_sql(
                    f"SELECT code, trade_date, open, high, low, close, volume, amount, turnover "
                    f"FROM stock_daily WHERE code IN ({code_list}) "
                    f"AND trade_date >= CURRENT_DATE - INTERVAL '5 years' ORDER BY code, trade_date",
                    engine,
                )
                engine.dispose()

                factor_names = list(ALL_FACTORS.keys())
                dataset = build_factor_dataset(ohlcv, factor_names, label_mode="binary")

                from factors.screening import filter_factors_by_ic, select_orthogonal_factors
                filtered = filter_factors_by_ic(dataset, factor_names, ret_col="ret_1d")
                selected = select_orthogonal_factors(dataset, filtered, threshold=0.7)

                results = walk_forward_train_ensemble(
                    dataset, selected, train_years=3, val_years=1,
                )

                if not results:
                    st.warning("训练未产生有效窗口，请检查数据范围")
                else:
                    latest = results[-1]
                    ensemble = latest["ensemble"]

                    # 获取最新交易日的因子数据
                    latest_date = dataset["trade_date"].max()
                    today_factors = dataset[dataset["trade_date"] == latest_date].dropna(
                        subset=selected
                    )

                    if today_factors.empty:
                        st.warning(f"最新交易日 {latest_date.date()} 无有效因子数据")
                    else:
                        scores = ensemble.predict(today_factors)
                        top20 = scores.head(20)

                        st.success(f"评估日期: {latest_date.date()}，基于 {len(selected)} 个因子")

                        # 合并名称
                        basic_df = pd.read_sql(
                            f"SELECT code, name, industry FROM stock_basic WHERE code IN ({code_list})",
                            get_engine(),
                        )
                        display = top20.merge(
                            basic_df[["code", "name", "industry"]], on="code", how="left",
                        )
                        display = display[["code", "name", "industry", "score", "rank"]]
                        display["score"] = display["score"].round(4)
                        st.dataframe(display, use_container_width=True, hide_index=True)

                        # 行业分布
                        industry_count = display["industry"].value_counts()
                        if not industry_count.empty:
                            import plotly.express as px
                            fig_pie = px.pie(
                                values=industry_count.values,
                                names=industry_count.index,
                                title="Top-20 行业分布",
                            )
                            fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_pie, use_container_width=True)

            except Exception as e:
                st.error(f"预测失败: {e}")

# ── Tab 4: 模拟盘净值 ───────────────────────────────────

with tab4:
    st.subheader("模拟盘表现")

    accounts = load_paper_accounts()
    if accounts.empty:
        st.info("暂无模拟账户，请在「模拟盘」页面创建账户")
    else:
        acc_id = st.selectbox(
            "选择账户",
            accounts["id"].tolist(),
            format_func=lambda x: f"#{x} {accounts[accounts['id']==x]['name'].iloc[0]}",
        )

        pnl = load_paper_pnl(acc_id)
        positions = load_paper_positions(acc_id)

        if pnl.empty:
            st.info("暂无净值记录，请运行模拟盘引擎")
        else:
            col1, col2, col3, col4 = st.columns(4)
            total_return = (pnl["total_value"].iloc[-1] / pnl["total_value"].iloc[0] - 1) if len(pnl) > 1 else 0
            with col1:
                st.metric("累计收益", f"{total_return:.2%}")
            with col2:
                max_dd = pnl["drawdown"].max() if not pnl["drawdown"].isna().all() else 0
                st.metric("最大回撤", f"{max_dd:.2%}")
            with col3:
                if len(pnl) > 1:
                    rets = pnl["total_value"].pct_change().dropna()
                    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
                    st.metric("夏普比率", f"{sharpe:.2f}")
                else:
                    st.metric("夏普比率", "-")
            with col4:
                st.metric("交易天数", len(pnl))

            # 权益曲线
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pnl["trade_date"], y=pnl["total_value"],
                mode="lines", name="总资产",
                line=dict(color="#2196f3", width=2),
                fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
            ))
            init_val = accounts[accounts["id"] == acc_id]["initial_capital"].iloc[0]
            fig.add_hline(
                y=init_val, line_dash="dash", line_color="gray",
                annotation_text=f"初始 {init_val:,.0f}",
            )
            fig.update_layout(
                height=350, template="plotly_white",
                margin=dict(l=0, r=0, t=10, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        # 当前持仓
        st.subheader("当前持仓")
        if positions.empty:
            st.info("暂无持仓")
        else:
            pos_display = positions.copy()
            pos_display["volume"] = pos_display["volume"].astype(int)
            pos_display["avg_cost"] = pos_display["avg_cost"].apply(lambda x: f"{x:.2f}")
            st.dataframe(
                pos_display[["code", "volume", "avg_cost"]],
                use_container_width=True, hide_index=True,
            )
