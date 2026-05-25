import pytest
import pandas as pd
from data.fetcher import fetch_financial_data


def test_fetch_financial_data_returns_dataframe():
    df = fetch_financial_data("000001")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "平安银行应该有财务数据"

    required = ["code", "report_date", "revenue", "net_profit", "gross_margin",
                "net_margin", "roe", "total_assets", "total_liability",
                "bps", "eps", "cash_flow"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

    assert all(df["code"] == "000001")


def test_fetch_financial_data_invalid_code():
    df = fetch_financial_data("999999")
    assert isinstance(df, pd.DataFrame)
    required = ["code", "report_date", "revenue", "net_profit", "gross_margin",
                "net_margin", "roe", "total_assets", "total_liability",
                "bps", "eps", "cash_flow"]
    for col in required:
        assert col in df.columns, f"Missing column: {col} (empty df should still have schema)"
