"""
Universe data assembler — loads ~30 raw columns from 7 DB tables into one
unified (code, trade_date) DataFrame.

Usage:
    from factors.data_assembler import assemble_universe

    df = assemble_universe(engine, "2024-01-01", "2025-12-31")
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.settings import DBConfig


# ── output column order (matches spec) ──────────────────────────
_OUTPUT_COLS = [
    "code", "trade_date",
    "open", "high", "low", "close", "volume", "amount", "turnover",
    "mcap", "float_mcap", "pe", "pb", "total_share", "float_share",
    "roe", "gross_margin", "net_margin", "bps", "eps",
    "cashflow_ps", "ocf_ps", "goodwill_ratio", "debt_ratio",
    "adjusted_profit",
    "north_net", "north_buy", "north_sell",
    "cn_10y", "us_10y", "spread_cn_us",
    "usd_cny",
    "industry_sw1", "industry_sw2",
    "is_st", "list_date",
]


# ── internal helpers ────────────────────────────────────────────

def _read_sql(query: str, engine: Engine, params: dict | None = None) -> pd.DataFrame:
    """Execute a parameterized query and return a DataFrame."""
    return pd.read_sql(text(query), engine, params=params or {})


def _load_daily(engine: Engine, start: str, end: str) -> pd.DataFrame:
    """Load stock_daily (OHLCV + turnover) for the date range."""
    cols = ["code", "trade_date", "open", "high", "low", "close",
            "volume", "amount", "turnover"]
    sql = (
        "SELECT " + ", ".join(cols) + " "
        "FROM stock_daily "
        "WHERE trade_date BETWEEN :start AND :end "
        "ORDER BY code, trade_date"
    )
    df = _read_sql(sql, engine, {"start": start, "end": end})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _load_daily_extra(engine: Engine, start: str, end: str) -> pd.DataFrame:
    """Load stock_daily_extra (market cap, pe, pb, shares)."""
    cols = ["code", "trade_date", "market_cap", "float_market_cap",
            "pe", "pb", "total_share", "float_share"]
    sql = (
        "SELECT " + ", ".join(cols) + " "
        "FROM stock_daily_extra "
        "WHERE trade_date BETWEEN :start AND :end "
        "ORDER BY code, trade_date"
    )
    df = _read_sql(sql, engine, {"start": start, "end": end})
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # rename to match output spec
    df = df.rename(columns={
        "market_cap": "mcap",
        "float_market_cap": "float_mcap",
    })
    return df


def _load_financial(engine: Engine, start: str, end: str) -> pd.DataFrame:
    """Load stock_financial from buffer-start to end.

    Returns a DataFrame with code, report_date plus financial columns
    needed for both direct output and derivation.
    """
    cols = [
        "code", "report_date",
        # direct output
        "roe", "gross_margin", "net_margin", "bps", "eps",
        "adjusted_profit",
        # for derived columns
        "cash_flow", "operating_cash_flow", "goodwill",
        "total_assets", "total_liability",
    ]
    # only select columns that actually exist in the table
    sql = (
        "SELECT " + ", ".join(cols) + " "
        "FROM stock_financial "
        "WHERE report_date BETWEEN :start AND :end "
        "ORDER BY code, report_date"
    )
    df = _read_sql(sql, engine, {"start": start, "end": end})
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def _load_static(engine: Engine, table: str, cols: list[str]) -> pd.DataFrame:
    """Load a static (per-code) table."""
    sql = f"SELECT {', '.join(cols)} FROM {table}"
    return _read_sql(sql, engine)


def _load_macro(engine: Engine, table: str, cols: list[str],
                start: str, end: str) -> pd.DataFrame:
    """Load a macro (per-trade_date) table."""
    sql = (
        f"SELECT {', '.join(cols)} FROM {table} "
        "WHERE trade_date BETWEEN :start AND :end "
        "ORDER BY trade_date"
    )
    df = _read_sql(sql, engine, {"start": start, "end": end})
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _forward_fill_financial(
    daily_df: pd.DataFrame, fin_df: pd.DataFrame
) -> pd.DataFrame:
    """Forward-fill quarterly financial data to every trade_date.

    Uses merge_asof: for each (code, trade_date), picks the most recent
    report_date <= trade_date.

    Parameters
    ----------
    daily_df : DataFrame with at least [code, trade_date]
    fin_df   : DataFrame with at least [code, report_date, ...financial cols...]

    Returns
    -------
    DataFrame : daily_df with financial columns forward-filled.
    """
    if fin_df.empty:
        # no financial data — add NaN columns
        fin_cols = [c for c in fin_df.columns if c not in ("code", "report_date")]
        for c in fin_cols:
            daily_df[c] = np.nan
        return daily_df

    fin_cols = [c for c in fin_df.columns if c not in ("code", "report_date")]

    # sort by (date, code) so that the on-key (trade_date/report_date) is
    # globally monotonic — required by pandas >=3.0 merge_asof with `by`.
    daily_sorted = daily_df.sort_values(["trade_date", "code"]).reset_index(drop=True)
    fin_sorted = fin_df.sort_values(["report_date", "code"]).reset_index(drop=True)

    merged = pd.merge_asof(
        daily_sorted,
        fin_sorted,
        left_on="trade_date",
        right_on="report_date",
        by="code",
        direction="backward",  # report_date <= trade_date
    )

    # drop the report_date column that merge_asof brought in
    merged = merged.drop(columns=["report_date"], errors="ignore")
    return merged


def _derive_financial_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived financial columns after forward-fill and daily merge.

    Requires:
      - cash_flow (already per-share from DB → cashflow_ps)
      - operating_cash_flow (total) + total_share (from daily_extra) → ocf_ps
      - goodwill, total_assets → goodwill_ratio
      - total_liability, total_assets → debt_ratio
    """
    # cashflow_ps: cash_flow is already 每股经营现金流 (per share)
    if "cash_flow" in df.columns:
        df["cashflow_ps"] = df["cash_flow"]
    else:
        df["cashflow_ps"] = np.nan

    # ocf_ps: operating_cash_flow is total, divide by total_share
    if "operating_cash_flow" in df.columns and "total_share" in df.columns:
        total_share_safe = df["total_share"].replace(0, np.nan)
        df["ocf_ps"] = df["operating_cash_flow"] / total_share_safe
    else:
        df["ocf_ps"] = np.nan

    # goodwill_ratio = COALESCE(goodwill, 0) / NULLIF(total_assets, 0)
    goodwill = df.get("goodwill", pd.Series(0, index=df.index)).fillna(0)
    ta = df.get("total_assets", pd.Series(np.nan, index=df.index))
    df["goodwill_ratio"] = goodwill / ta.replace(0, np.nan)

    # debt_ratio = total_liability / NULLIF(total_assets, 0)
    tl = df.get("total_liability", pd.Series(np.nan, index=df.index))
    df["debt_ratio"] = tl / ta.replace(0, np.nan)

    return df


# ── main interface ──────────────────────────────────────────────

def assemble_universe(
    engine: Engine | None = None,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Load raw columns from 7 DB tables into a unified (code, trade_date) wide table.

    Parameters
    ----------
    engine : SQLAlchemy engine (auto-created via DBConfig if None).
    start  : Start date (YYYY-MM-DD).
    end    : End date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame with columns:
      code, trade_date,
      open, high, low, close, volume, amount, turnover,
      mcap, float_mcap, pe, pb, total_share, float_share,
      roe, gross_margin, net_margin, bps, eps,
      cashflow_ps, ocf_ps, goodwill_ratio, debt_ratio,
      adjusted_profit,
      north_net, north_buy, north_sell,
      cn_10y, us_10y, spread_cn_us,
      usd_cny,
      industry_sw1, industry_sw2,
      is_st, list_date

    Filters applied:
      - Non-ST stocks only.
      - Listed > 60 calendar days.
    """
    if engine is None:
        engine = DBConfig.create_engine()

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    fin_buffer_start = (start_dt - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    start_str = start_dt.strftime("%Y-%m-%d")

    # ── 1. stock_daily (OHLCV + turnover) ──
    daily_df = _load_daily(engine, start_str, end_str)
    if daily_df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # ── 2. stock_daily_extra (mcap, pe, pb, shares) ──
    extra_df = _load_daily_extra(engine, start_str, end_str)
    df = daily_df.merge(extra_df, on=["code", "trade_date"], how="left")

    # ── 3. stock_financial (forward-fill quarterly → daily) ──
    fin_df = _load_financial(engine, fin_buffer_start, end_str)
    df = _forward_fill_financial(df, fin_df)

    # ── 4. derive financial columns ──
    df = _derive_financial_cols(df)

    # ── 5. stock_industry (static, per code) ──
    industry_df = _load_static(engine, "stock_industry",
                               ["code", "industry_sw1", "industry_sw2"])
    df = df.merge(industry_df, on="code", how="left")

    # ── 6. stock_basic (static, per code) ──
    basic_df = _load_static(engine, "stock_basic", ["code", "is_st", "list_date"])
    if "list_date" in basic_df.columns:
        basic_df["list_date"] = pd.to_datetime(basic_df["list_date"])
    df = df.merge(basic_df, on="code", how="left")

    # ── 7. macro: north flow (broadcast per trade_date) ──
    north_df = _load_macro(
        engine, "market_north_flow",
        ["trade_date", "net_flow", "buy_amount", "sell_amount"],
        start_str, end_str,
    )
    north_df = north_df.rename(columns={
        "net_flow": "north_net",
        "buy_amount": "north_buy",
        "sell_amount": "north_sell",
    })
    df = df.merge(north_df, on="trade_date", how="left")

    # ── 8. macro: bond yields (broadcast per trade_date) ──
    bond_df = _load_macro(
        engine, "market_bond_yield",
        ["trade_date", "cn_10y", "us_10y", "spread_cn_us_10y"],
        start_str, end_str,
    )
    bond_df = bond_df.rename(columns={"spread_cn_us_10y": "spread_cn_us"})
    df = df.merge(bond_df, on="trade_date", how="left")

    # ── 9. macro: FX rate (broadcast per trade_date) ──
    fx_df = _load_macro(
        engine, "market_fx_rate",
        ["trade_date", "usd_cny"],
        start_str, end_str,
    )
    df = df.merge(fx_df, on="trade_date", how="left")

    # ── 10. filters ──
    # remove ST
    df = df[~df["is_st"].fillna(False)]
    # remove newly listed (listed ≤ 60 calendar days)
    df["list_days"] = (df["trade_date"] - df["list_date"]).dt.days
    df = df[df["list_days"] > 60]

    # ── 11. drop intermediate / raw columns ──
    drop_cols = [
        "cash_flow", "operating_cash_flow", "goodwill",
        "total_assets", "total_liability",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ── 12. sort & select output columns ──
    df = df.sort_values(["code", "trade_date"]).reset_index(drop=True)

    # only keep columns that exist in the DataFrame
    out_cols = [c for c in _OUTPUT_COLS if c in df.columns]
    df = df[out_cols]

    return df
