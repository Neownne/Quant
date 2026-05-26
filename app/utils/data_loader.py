import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import text

from data.db import get_engine


# ---------- 资产类型检测 ----------

def detect_asset_type(code: str) -> str:
    """检测代码属于哪种资产类型。返回 "stock" / "unknown"。
    ETF/基金检测暂时禁用，需要恢复时取消下方注释即可。"""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM stock_basic WHERE code = :c"), {"c": code}
        ).first()
        if row:
            engine.dispose()
            return "stock"

        # -- ETF/基金检测（暂时禁用） --
        # row = conn.execute(
        #     text("SELECT 1 FROM etf_basic WHERE code = :c"), {"c": code}
        # ).first()
        # if row:
        #     engine.dispose()
        #     return "etf"
        #
        # row = conn.execute(
        #     text("SELECT 1 FROM fund_basic WHERE code = :c"), {"c": code}
        # ).first()
        # if row:
        #     engine.dispose()
        #     return "fund"

    engine.dispose()
    return "unknown"


# ---------- 数据查询 ----------

@st.cache_data(ttl=3600)
def get_stock_list() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT code, name FROM stock_basic WHERE is_st = FALSE ORDER BY code"), conn
        )
    engine.dispose()
    df["type"] = "stock"
    return df


@st.cache_data(ttl=3600)
def get_etf_list() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT code, name FROM etf_basic ORDER BY code"), conn
        )
    engine.dispose()
    df["type"] = "etf"
    return df


@st.cache_data(ttl=3600)
def get_fund_list() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT code, name FROM fund_basic ORDER BY code"), conn
        )
    engine.dispose()
    df["type"] = "fund"
    return df


@st.cache_data(ttl=3600)
def get_all_assets() -> pd.DataFrame:
    """股票合并列表，含 type 列。（ETF/基金合并暂时禁用）"""
    stocks = get_stock_list()
    # etfs = get_etf_list()   # -- 暂时禁用 --
    # funds = get_fund_list() # -- 暂时禁用 --
    return stocks  # pd.concat([stocks, etfs, funds], ignore_index=True)


def get_asset_name(code: str) -> str:
    """根据代码查名称（股票表）。ETF/基金查询暂时禁用。"""
    engine = get_engine()
    for table in ["stock_basic"]:  # , "etf_basic", "fund_basic" -- 暂时禁用
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT name FROM {table} WHERE code = :c"), {"c": code}
            ).first()
        if row:
            engine.dispose()
            return row[0]
    engine.dispose()
    return code


@st.cache_data(ttl=60)
def load_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    """加载 OHLCV 数据（股票 → stock_daily）。ETF/基金路由暂时禁用。"""
    at = detect_asset_type(code)
    engine = get_engine()

    # -- ETF/基金路由（暂时禁用） --
    # if at == "etf":
    #     sql = """SELECT trade_date, open, high, low, close, volume, amount FROM etf_daily ..."""
    # elif at == "fund":
    #     sql = """SELECT nav_date AS trade_date, ... FROM fund_nav ..."""
    # else:
    sql = """
        SELECT trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily
        WHERE code = :code AND trade_date BETWEEN :start AND :end
        ORDER BY trade_date
    """

    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(sql), conn, params={"code": code, "start": start, "end": end}
        )
    engine.dispose()

    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


@st.cache_data(ttl=60)
def get_latest_trade_date(code: str) -> str:
    """获取某资产的最新数据日期（自动识别表，ETF/基金暂时禁用）。"""
    at = detect_asset_type(code)
    engine = get_engine()

    # -- ETF/基金表路由（暂时禁用） --
    # if at == "etf": table, col = "etf_daily", "trade_date"
    # elif at == "fund": table, col = "fund_nav", "nav_date"
    table, col = "stock_daily", "trade_date"

    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT MAX({col}) FROM {table} WHERE code = :code"),
            {"code": code},
        ).scalar()
    engine.dispose()
    return str(row) if row else ""


# ---------- 批量行情（非交易时段回退） ----------

@st.cache_data(ttl=60)
def get_latest_daily_batch(codes: list[str]) -> pd.DataFrame:
    """获取一批资产最近交易/净值日的数据（自动分流，ETF/基金暂时禁用）。"""
    if not codes:
        return pd.DataFrame()

    engine = get_engine()
    stock_codes = []  # etf_codes, fund_codes 暂时禁用
    for c in codes:
        at = detect_asset_type(c)
        # -- ETF/基金分流（暂时禁用） --
        # if at == "etf": etf_codes.append(c)
        # elif at == "fund": fund_codes.append(c)
        if at == "stock":
            stock_codes.append(c)

    parts = []

    # 股票
    if stock_codes:
        ph = ",".join([f":s{i}" for i in range(len(stock_codes))])
        sql = f"""
            SELECT DISTINCT ON (code) code, trade_date, open, high, low, close, volume, amount
            FROM stock_daily
            WHERE code IN ({ph})
            ORDER BY code, trade_date DESC
        """
        params = {f"s{i}": c for i, c in enumerate(stock_codes)}
        with engine.connect() as conn:
            parts.append(pd.read_sql_query(text(sql), conn, params=params))

    # -- ETF 和 基金 批量查询（暂时禁用） --
    # if etf_codes:
    #     ... (etf_daily query)
    # if fund_codes:
    #     ... (fund_nav query)

    engine.dispose()
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame()


# ---------- 实时行情 ----------

@st.cache_data(ttl=5)
def get_realtime_quotes(codes: list[str]) -> pd.DataFrame:
    """批量获取实时行情。股票用 stock_zh_a_spot_em()。ETF 行情暂时禁用。"""
    if not codes:
        return pd.DataFrame()

    import akshare as ak

    # -- ETF 分类（暂时禁用） --
    stock_codes = list(codes)  # , etf_codes = [], []
    # for c in codes:
    #     at = detect_asset_type(c)
    #     if at == "etf": etf_codes.append(c)

    parts = []

    # 股票实时行情
    if stock_codes:
        try:
            raw = ak.stock_zh_a_spot_em()
            raw["代码"] = raw["代码"].astype(str)
            result = raw[raw["代码"].isin(stock_codes)].copy()
            if not result.empty:
                out = pd.DataFrame()
                out["code"] = result["代码"]
                out["name"] = result["名称"]
                out["price"] = pd.to_numeric(result["最新价"], errors="coerce")
                out["change_pct"] = pd.to_numeric(result["涨跌幅"], errors="coerce")
                out["volume"] = pd.to_numeric(result["成交量"], errors="coerce")
                out["amount"] = pd.to_numeric(result["成交额"], errors="coerce")
                out["high"] = pd.to_numeric(result["最高"], errors="coerce")
                out["low"] = pd.to_numeric(result["最低"], errors="coerce")
                out["open"] = pd.to_numeric(result["今开"], errors="coerce")
                out["pre_close"] = pd.to_numeric(result["昨收"], errors="coerce")
                parts.append(out)
        except Exception:
            pass

    # -- ETF 实时行情（暂时禁用） --
    # if etf_codes:
    #     try:
    #         raw = ak.fund_etf_spot_em()
    #         ...

    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame()


# ---------- 技术指标 ----------

def calc_ma(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    result = df.copy()
    for p in periods:
        result[f"ma_{p}"] = result["close"].rolling(window=p).mean()
    return result


def calc_ema(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    result = df.copy()
    for p in periods:
        result[f"ema_{p}"] = result["close"].ewm(span=p, adjust=False).mean()
    return result


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    result = df.copy()
    ema_fast = result["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow, adjust=False).mean()
    result["dif"] = ema_fast - ema_slow
    result["dea"] = result["dif"].ewm(span=signal, adjust=False).mean()
    result["macd_hist"] = 2 * (result["dif"] - result["dea"])
    return result


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    result = df.copy()
    delta = result["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi"] = 100 - (100 / (1 + rs))
    return result


def calc_bollinger(df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
    result = df.copy()
    result["boll_mid"] = result["close"].rolling(window=period).mean()
    rolling_std = result["close"].rolling(window=period).std()
    result["boll_upper"] = result["boll_mid"] + std * rolling_std
    result["boll_lower"] = result["boll_mid"] - std * rolling_std
    return result


# ---------- K线图 ----------

def build_kline_chart(df: pd.DataFrame, indicators: dict, is_fund: bool = False) -> "plotly.graph_objects.Figure":
    """构建多 pane K线图表。is_fund 参数保留兼容性但暂未使用。
    indicators 可包含: ma_periods, ema_periods, macd, rsi, bollinger"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    ma_periods = indicators.get("ma_periods") or []
    ema_periods = indicators.get("ema_periods") or []
    show_macd = indicators.get("macd")
    show_rsi = indicators.get("rsi")
    show_bollinger = indicators.get("bollinger")

    if ma_periods:
        df = calc_ma(df, ma_periods)
    if ema_periods:
        df = calc_ema(df, ema_periods)
    if show_macd:
        df = calc_macd(df)
    if show_rsi:
        df = calc_rsi(df)
    if show_bollinger:
        df = calc_bollinger(df)

    has_vol = "volume" in df.columns and df["volume"].notna().any()
    has_macd = "dif" in df.columns
    has_rsi = "rsi" in df.columns

    subplot_rows = 1
    row_heights = [0.5]
    if has_vol:
        subplot_rows += 1
        row_heights.insert(-1, 0.2)
    if has_macd:
        subplot_rows += 1
        row_heights.append(0.2)
    if has_rsi:
        subplot_rows += 1
        row_heights.append(0.15)

    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=subplot_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # ---- Pane 1: 主图 ----
    if is_fund:
        # 基金用净值折线 + 填充
        fig.add_trace(
            go.Scatter(
                x=df["trade_date"], y=df["close"],
                mode="lines", name="单位净值",
                fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
                line=dict(width=2, color="#2196f3"),
            ),
            row=1, col=1,
        )
        # 累计净值虚线
        if "high" in df.columns and df["high"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df["trade_date"], y=df["high"],
                    mode="lines", name="累计净值",
                    line=dict(width=1, dash="dot", color="#9e9e9e"),
                ),
                row=1, col=1,
            )
    else:
        fig.add_trace(
            go.Candlestick(
                x=df["trade_date"],
                open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                name="K线",
                increasing_line_color="#ef5350",
                decreasing_line_color="#26a69a",
            ),
            row=1, col=1,
        )

    # MA / EMA 叠加线
    ma_colors = {5: "#ff9800", 10: "#2196f3", 20: "#9c27b0", 60: "#4caf50"}
    for p in ma_periods:
        col = f"ma_{p}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["trade_date"], y=df[col],
                    mode="lines", name=f"MA{p}",
                    line=dict(width=1, color=ma_colors.get(p, "#888")),
                ),
                row=1, col=1,
            )

    ema_colors = {12: "#e91e63", 26: "#00bcd4"}
    for p in ema_periods:
        col = f"ema_{p}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["trade_date"], y=df[col],
                    mode="lines", name=f"EMA{p}",
                    line=dict(width=1, dash="dot", color=ema_colors.get(p, "#888")),
                ),
                row=1, col=1,
            )

    if show_bollinger and "boll_upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["trade_date"], y=df["boll_upper"],
                mode="lines", name="Boll Upper",
                line=dict(width=0.5, color="rgba(128,128,128,0.5)"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["trade_date"], y=df["boll_lower"],
                mode="lines", name="Boll Lower",
                line=dict(width=0.5, color="rgba(128,128,128,0.5)"),
                fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
            ),
            row=1, col=1,
        )

    current_row = 1

    # ---- Pane 2: 成交量 ----
    if has_vol:
        current_row += 1
        vol_colors = [
            "red" if c >= o else "green"
            for c, o in zip(df["close"], df["open"])
        ] if not is_fund else "#2196f3"
        fig.add_trace(
            go.Bar(
                x=df["trade_date"], y=df["volume"],
                name="成交量", marker_color=vol_colors if isinstance(vol_colors, list) else vol_colors,
                showlegend=False,
            ),
            row=current_row, col=1,
        )

    # ---- Pane 3: MACD ----
    if has_macd:
        current_row += 1
        fig.add_trace(
            go.Bar(
                x=df["trade_date"], y=df["macd_hist"],
                name="MACD柱", showlegend=False,
                marker_color=["#ef5350" if v >= 0 else "#26a69a" for v in df["macd_hist"]],
            ),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["trade_date"], y=df["dif"],
                mode="lines", name="DIF",
                line=dict(width=1, color="#2196f3"),
            ),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["trade_date"], y=df["dea"],
                mode="lines", name="DEA",
                line=dict(width=1, color="#ff9800"),
            ),
            row=current_row, col=1,
        )

    # ---- Pane 4: RSI ----
    if has_rsi:
        current_row += 1
        fig.add_trace(
            go.Scatter(
                x=df["trade_date"], y=df["rsi"],
                mode="lines", name="RSI",
                line=dict(width=1.5, color="#7c4dff"),
            ),
            row=current_row, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.3)", row=current_row)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.3)", row=current_row)

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    y_title = "净值" if is_fund else "价格"
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    if has_vol:
        fig.update_yaxes(title_text="成交量", row=2, col=1)
    if has_macd:
        fig.update_yaxes(title_text="MACD", row=current_row - (1 if has_rsi else 0), col=1)
    if has_rsi:
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)

    return fig
