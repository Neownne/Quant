import pandas as pd
import numpy as np
from datetime import date


class DataQualityChecker:
    def __init__(self, expected_stock_count: int = 5000, coverage_threshold: float = 0.95):
        self.expected_stock_count = expected_stock_count
        self.coverage_threshold = coverage_threshold

    def run_all(self, df: pd.DataFrame, trade_date: date) -> list[dict]:
        results = []
        for method in [self.check_coverage, self.check_null_rate, self.check_frozen, self.check_jumps]:
            results.append(method(df, trade_date))
        return results

    def check_coverage(self, df: pd.DataFrame, trade_date: date) -> dict:
        actual = len(df)
        ratio = actual / self.expected_stock_count if self.expected_stock_count > 0 else 1.0
        passed = bool(ratio >= self.coverage_threshold)
        return {
            "check_name": "coverage",
            "trade_date": trade_date,
            "expected": str(self.expected_stock_count),
            "actual": str(actual),
            "passed": passed,
            "detail": f"覆盖率 {ratio:.2%}，阈值 {self.coverage_threshold:.0%}",
        }

    def check_null_rate(self, df: pd.DataFrame, trade_date: date) -> dict:
        null_close = df["close"].isna().sum()
        null_vol = df["volume"].isna().sum()
        bad = null_close + null_vol
        passed = bool(bad == 0)
        return {
            "check_name": "null_rate",
            "trade_date": trade_date,
            "expected": "0",
            "actual": str(bad),
            "passed": passed,
            "detail": f"close空{null_close}行, volume空{null_vol}行",
        }

    def check_frozen(self, df: pd.DataFrame, trade_date: date) -> dict:
        zero_vol = (df["volume"] == 0).sum()
        ratio = zero_vol / len(df) if len(df) > 0 else 0
        passed = bool(ratio < 0.05)
        return {
            "check_name": "frozen",
            "trade_date": trade_date,
            "expected": "<5%",
            "actual": f"{ratio:.1%}",
            "passed": passed,
            "detail": f"零成交量{zero_vol}只 ({ratio:.2%})",
        }

    def check_jumps(self, df: pd.DataFrame, trade_date: date) -> dict:
        if "change_pct" not in df.columns or df["change_pct"].dropna().empty:
            return {"check_name": "jumps", "trade_date": trade_date,
                    "expected": "no_extreme", "actual": "no_data", "passed": True, "detail": ""}
        extreme = (df["change_pct"].abs() > 20).sum()
        passed = bool(extreme == 0)
        return {
            "check_name": "jumps",
            "trade_date": trade_date,
            "expected": "0",
            "actual": str(extreme),
            "passed": passed,
            "detail": f"涨跌幅超20%的{extreme}只",
        }
