import pytest
import pandas as pd
from data.quality import DataQualityChecker


class TestCoverageCheck:
    def test_coverage_pass_when_all_stocks_present(self):
        checker = DataQualityChecker(expected_stock_count=5000)
        df = pd.DataFrame({
            "code": [f"00000{i}" for i in range(5000)],
            "close": [10.0] * 5000,
            "volume": [1e6] * 5000,
            "change_pct": [1.0] * 5000,
        })
        result = checker.check_coverage(df, "2024-01-15")
        assert result["passed"] is True

    def test_coverage_fail_when_coverage_low(self):
        checker = DataQualityChecker(expected_stock_count=5000)
        df = pd.DataFrame({
            "code": ["000001"],
            "close": [10.0],
            "volume": [1e6],
            "change_pct": [1.0],
        })
        result = checker.check_coverage(df, "2024-01-15")
        assert result["passed"] is False

    def test_null_rate_detects_missing_close(self):
        checker = DataQualityChecker(expected_stock_count=100)
        df = pd.DataFrame({
            "code": [f"00000{i}" for i in range(100)],
            "close": [10.0] * 99 + [None],
            "volume": [1e6] * 100,
            "change_pct": [1.0] * 100,
        })
        result = checker.check_null_rate(df, "2024-01-15")
        assert result["passed"] is False

    def test_limit_freeze_detected(self):
        checker = DataQualityChecker(expected_stock_count=100)
        df = pd.DataFrame({
            "code": [f"00000{i}" for i in range(100)],
            "close": [10.0] * 100,
            "volume": [1e6] * 80 + [0] * 20,
            "change_pct": [1.0] * 100,
        })
        result = checker.check_frozen(df, "2024-01-15")
        assert not result["passed"]
