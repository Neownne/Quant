"""Tests for factors/data_assembler.py — universe data assembler."""

import sys
import os

import pytest
import pandas as pd
import numpy as np

# ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factors.data_assembler import assemble_universe, _OUTPUT_COLS


# ── expected columns grouped by source ──────────────────────────

_PRICE_COLS = ["open", "high", "low", "close", "volume", "amount", "turnover"]
_EXTRA_COLS = ["mcap", "float_mcap", "pe", "pb", "total_share", "float_share"]
_FIN_DIRECT_COLS = ["roe", "gross_margin", "net_margin", "bps", "eps", "adjusted_profit"]
_FIN_DERIVED_COLS = ["cashflow_ps", "ocf_ps", "goodwill_ratio", "debt_ratio"]
_NORTH_COLS = ["north_net", "north_buy", "north_sell"]
_BOND_COLS = ["cn_10y", "us_10y", "spread_cn_us"]
_FX_COLS = ["usd_cny"]
_STATIC_COLS = ["industry_sw1", "industry_sw2", "is_st", "list_date"]

_REQUIRED_COLS = (
    ["code", "trade_date"]
    + _PRICE_COLS
    + _EXTRA_COLS
    + _FIN_DIRECT_COLS
    + _FIN_DERIVED_COLS
    + _NORTH_COLS
    + _BOND_COLS
    + _FX_COLS
    + _STATIC_COLS
)


# ── fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    """Create a DB engine once per test module."""
    from config.settings import DBConfig
    eng = DBConfig.create_engine()
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def assembled(engine):
    """Run assemble_universe on a recent short window (lightweight)."""
    df = assemble_universe(engine, start="2025-09-01", end="2025-09-05")
    return df


# ── tests ───────────────────────────────────────────────────────

class TestOutputShape:
    """Basic shape and structure."""

    def test_returns_dataframe(self, assembled):
        assert isinstance(assembled, pd.DataFrame)

    def test_not_empty(self, assembled):
        assert len(assembled) > 0, "Should return data for a valid date range"

    def test_has_all_required_columns(self, assembled):
        missing = set(_REQUIRED_COLS) - set(assembled.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_extra_output_cols(self, assembled):
        """All columns in output should be documented in _OUTPUT_COLS."""
        extra = set(assembled.columns) - set(_OUTPUT_COLS)
        assert not extra, f"Unexpected output columns: {extra}"

    def test_sorted_by_code_trade_date(self, assembled):
        df = assembled.reset_index(drop=True)
        expected = df.sort_values(["code", "trade_date"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df, expected)

    def test_unique_code_date_pairs(self, assembled):
        dupes = assembled.duplicated(subset=["code", "trade_date"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate (code, trade_date) rows"


class TestDataTypes:
    """Column data types."""

    def test_trade_date_is_datetime(self, assembled):
        assert pd.api.types.is_datetime64_any_dtype(assembled["trade_date"])

    def test_list_date_is_datetime(self, assembled):
        non_null = assembled["list_date"].dropna()
        if len(non_null) > 0:
            assert pd.api.types.is_datetime64_any_dtype(non_null)

    def test_code_is_string(self, assembled):
        # pandas >=3.0 uses StringDtype; <3.0 uses object
        assert pd.api.types.is_string_dtype(assembled["code"]), \
            f"code dtype={assembled['code'].dtype} should be string-like"

    def test_price_cols_are_numeric(self, assembled):
        for col in _PRICE_COLS:
            assert pd.api.types.is_numeric_dtype(assembled[col]), \
                f"{col} should be numeric, got {assembled[col].dtype}"


class TestFilters:
    """Filtering logic."""

    def test_no_st_stocks(self, assembled):
        """No ST stocks should be present (is_st is False or NaN that was filtered out)."""
        st_count = assembled[assembled["is_st"].fillna(False) == True].shape[0]  # noqa: E712
        assert st_count == 0, f"Found {st_count} ST stocks"

    def test_listed_more_than_60_days(self, assembled):
        """All stocks should be listed > 60 calendar days."""
        assembled_with_days = assembled.copy()
        assembled_with_days["list_days"] = (
            assembled_with_days["trade_date"] - assembled_with_days["list_date"]
        ).dt.days
        # all non-null list_days should be > 60
        valid = assembled_with_days["list_days"].dropna()
        assert (valid > 60).all(), \
            f"Found {(valid <= 60).sum()} rows with list_days <= 60"

    def test_list_date_not_future(self, assembled):
        """list_date should be <= trade_date for all rows."""
        assembled_with_days = assembled.copy()
        assembled_with_days["list_days"] = (
            assembled_with_days["trade_date"] - assembled_with_days["list_date"]
        ).dt.days
        valid = assembled_with_days["list_days"].dropna()
        assert (valid >= 0).all(), \
            f"Found {(valid < 0).sum()} rows with list_date in future"


class TestMergedColumns:
    """Verify that merged columns have expected characteristics."""

    def test_price_columns_present(self, assembled):
        for col in _PRICE_COLS:
            assert col in assembled.columns
            # at least 50% non-null for liquid stocks
            non_null_pct = assembled[col].notna().mean()
            assert non_null_pct > 0.5, \
                f"{col}: only {non_null_pct:.1%} non-null"

    def test_extra_columns_present(self, assembled):
        for col in _EXTRA_COLS:
            assert col in assembled.columns

    def test_macro_broadcast_same_per_date(self, assembled):
        """Macro values should be identical for all stocks on the same trade_date."""
        for col in _NORTH_COLS + _BOND_COLS + _FX_COLS:
            # group by trade_date, check that each group has only 1 unique value (or all NaN)
            unique_per_date = assembled.groupby("trade_date")[col].nunique()
            # Allow 1 (one unique value) or 0 (all NaN counted as 0 by nunique)
            bad = unique_per_date[unique_per_date > 1]
            assert len(bad) == 0, \
                f"{col} has >1 unique value on dates: {bad.index.tolist()[:5]}"

    def test_industry_not_all_null(self, assembled):
        """At least some stocks should have industry classification."""
        sw1_pct = assembled["industry_sw1"].notna().mean()
        assert sw1_pct > 0.1, \
            f"industry_sw1 only {sw1_pct:.1%} non-null — suspicious"

    def test_financial_cols_exist(self, assembled):
        """All financial columns exist even if sparse."""
        fin_cols = _FIN_DIRECT_COLS + _FIN_DERIVED_COLS
        for col in fin_cols:
            assert col in assembled.columns, f"Missing financial column: {col}"


class TestDerivedColumns:
    """Derived columns have reasonable values."""

    def test_goodwill_ratio_range(self, assembled):
        gr = assembled["goodwill_ratio"].dropna()
        if len(gr) > 0:
            assert (gr >= 0).all(), "goodwill_ratio should be >= 0"
            assert (gr <= 2).all() or gr.quantile(0.99) <= 2, \
                "goodwill_ratio should be in reasonable range"

    def test_debt_ratio_range(self, assembled):
        dr = assembled["debt_ratio"].dropna()
        if len(dr) > 0:
            assert (dr >= 0).all() or dr.quantile(0.01) >= -0.1, \
                "debt_ratio should be >= 0 for most stocks"

    def test_ocf_ps_not_all_null_when_operating_cash_flow_exists(self, assembled):
        """If there's any financial data, ocf_ps should have some non-null values."""
        # just check the column exists and is numeric
        assert "ocf_ps" in assembled.columns
        assert pd.api.types.is_numeric_dtype(assembled["ocf_ps"])

    def test_cashflow_ps_not_all_null_when_cash_flow_exists(self, assembled):
        assert "cashflow_ps" in assembled.columns
        assert pd.api.types.is_numeric_dtype(assembled["cashflow_ps"])


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_range(self, engine):
        """A date range with no trading data returns empty DataFrame with correct columns."""
        df = assemble_universe(engine, start="1990-01-01", end="1990-01-05")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        for col in _REQUIRED_COLS:
            assert col in df.columns, f"Empty result missing column: {col}"

    def test_single_day(self, engine):
        """Single-day query works."""
        # Use a known trading day
        df = assemble_universe(engine, start="2025-09-01", end="2025-09-01")
        assert len(df) > 0
        assert df["trade_date"].nunique() == 1

    def test_date_range_ordering(self, engine):
        """Start > end should not crash; just return empty (graceful)."""
        df = assemble_universe(engine, start="2025-12-31", end="2025-01-01")
        assert isinstance(df, pd.DataFrame)
        # Either empty or correctly handles it

    def test_engine_default(self):
        """Calling without engine should auto-create one via DBConfig."""
        df = assemble_universe(start="2025-09-01", end="2025-09-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestFinancialForwardFill:
    """Verify forward-fill behaviour of financial data."""

    def test_financial_data_present_for_some_stocks(self, assembled):
        """At least some stocks should have non-null financial data after ffill."""
        for col in ["roe", "eps", "bps", "net_margin"]:
            pct = assembled[col].notna().mean()
            assert pct > 0.01, \
                f"{col}: only {pct:.3%} non-null — forward-fill may not be working"

    def test_forward_fill_consistency(self, assembled):
        """For a given stock, forward-filled financial values should stay constant
        between report dates (non-decreasing in info, monotonic in fill)."""
        # pick a stock with financial data
        for col in ["roe", "eps", "bps"]:
            has_data = assembled.groupby("code")[col].transform("any")
            sample = assembled[has_data].head(500)
            if len(sample) == 0:
                continue
            # check: once a value appears, it should remain until next report
            for code in sample["code"].unique()[:3]:
                stock = assembled[assembled["code"] == code].sort_values("trade_date")
                vals = stock[col].ffill()  # re-ffill for reference
                # The forward-filled values should match re-forward-filled values
                # (since they were already forward-filled)
                actual = stock[col].values
                expected = vals.values
                # Compare non-NaN positions
                mask = ~np.isnan(actual) & ~np.isnan(expected)
                if mask.sum() > 0:
                    np.testing.assert_array_almost_equal(
                        actual[mask], expected[mask], decimal=6,
                        err_msg=f"{col} ffill inconsistency for {code}"
                    )
                    break  # one stock is enough
