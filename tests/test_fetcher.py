import pytest
import pandas as pd
from data.fetcher import fetch_financial_data


def test_fetch_financial_data_returns_dataframe():
    df = fetch_financial_data("000001")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "平安银行应该有财务数据"
    assert "code" in df.columns
    assert "report_date" in df.columns
    assert "revenue" in df.columns
    assert "net_profit" in df.columns
    assert "roe" in df.columns
    assert "eps" in df.columns
    assert all(df["code"] == "000001")


def test_fetch_financial_data_invalid_code():
    df = fetch_financial_data("999999")
    assert isinstance(df, pd.DataFrame)
