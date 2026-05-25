import pytest
import pandas as pd
from data.fetcher import fetch_financial_data

FINANCIAL_COLS = ["code", "report_date", "revenue", "net_profit", "gross_margin",
                   "net_margin", "roe", "total_assets", "total_liability",
                   "bps", "eps", "cash_flow"]


def test_fetch_financial_data_returns_dataframe():
    df = fetch_financial_data("000001")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "平安银行应该有财务数据"

    for col in FINANCIAL_COLS:
        assert col in df.columns, f"Missing column: {col}"

    assert all(df["code"] == "000001")

    # Verify data quality for 000001
    # report_date should be date objects
    assert all(hasattr(d, 'year') for d in df["report_date"].head()), \
        "report_date should be date objects"

    # At least some non-NaN values in key numeric columns
    numeric_cols = ["revenue", "net_profit", "net_margin", "roe", "bps", "eps", "cash_flow"]
    for col in numeric_cols:
        non_null = df[col].dropna()
        assert len(non_null) > 0, f"{col} should have non-NaN values for 000001"
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"

    # report_date should span at least 5 years
    dmin, dmax = df["report_date"].min(), df["report_date"].max()
    assert dmax.year - dmin.year >= 5, \
        f"report_date range too narrow: {dmin} to {dmax}"


def test_fetch_financial_data_invalid_code():
    df = fetch_financial_data("999999")
    assert isinstance(df, pd.DataFrame)
    for col in FINANCIAL_COLS:
        assert col in df.columns, f"Missing column: {col} (empty df should still have schema)"


def test_fetch_industry_classification():
    """行业分类应返回大量股票的 DataFrame。"""
    from data.fetcher import fetch_industry_classification
    df = fetch_industry_classification()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "行业分类不应为空"
    assert "code" in df.columns
    assert "industry_sw1" in df.columns
    assert "market" in df.columns
    # 至少覆盖 2000 只股票
    assert len(df) > 2000, f"行业表应有 >2000 条记录，实际: {len(df)}"
    # code 不应有重复
    assert df["code"].is_unique, "code 不应有重复"
    # 检查 market 值
    valid_markets = {"主板", "创业板", "科创板", "北交所", "未知"}
    actual_markets = set(df["market"].unique())
    assert actual_markets.issubset(valid_markets), f"无效 market 值: {actual_markets - valid_markets}"
