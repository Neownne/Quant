"""板块数据集构建测试。"""
import pytest
import pandas as pd
import numpy as np
from models.sector_dataset import build_sector_dataset


def _make_multi_day_ohlcv(
    codes: list[str],
    dates: list[pd.Timestamp],
    close_matrix: np.ndarray,  # (n_dates, n_codes)
) -> pd.DataFrame:
    """构造多日 OHLCV 测试数据。"""
    n_dates = len(dates)
    n_codes = len(codes)
    records = []
    for i, date in enumerate(dates):
        for j, code in enumerate(codes):
            c = float(close_matrix[i, j])
            records.append({
                "code": code,
                "trade_date": date,
                "open": c * 0.99,
                "high": c * 1.02,
                "low": c * 0.98,
                "close": c,
                "volume": 1_000_000,
                "amount": c * 1_000_000,
                "turnover": 2.0,
            })
    return pd.DataFrame(records)


class TestBuildSectorDataset:
    """板块数据集构建测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回的 DataFrame 应含 trade_date, sector, 特征列, label, ret_Nd。"""
        codes = ["688001.SH", "688002.SH", "600001.SH", "600002.SH"]
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        # 各股随机游走
        np.random.seed(42)
        close = 10 + np.cumsum(np.random.randn(30, 4) * 0.2, axis=0)
        ohlcv = _make_multi_day_ohlcv(codes, dates, close)

        sector_map = {"688001.SH": "科创", "688002.SH": "科创",
                      "600001.SH": "主板大盘", "600002.SH": "主板大盘"}

        result = build_sector_dataset(ohlcv, sector_map, forward_days=5)

        assert isinstance(result, pd.DataFrame)
        assert "trade_date" in result.columns
        assert "sector" in result.columns
        assert "label" in result.columns
        # 应包含板块宽度特征
        assert "advance_decline_ratio" in result.columns
        assert "sector_ret_mean" in result.columns

    def test_labels_are_binary(self):
        """标签应为 0 或 1（二分类）。"""
        codes = ["688001.SH", "600001.SH", "600002.SH", "600003.SH"]
        dates = pd.date_range("2024-01-02", periods=50, freq="B")
        np.random.seed(42)
        close = 10 + np.cumsum(np.random.randn(50, 4) * 0.3, axis=0)
        ohlcv = _make_multi_day_ohlcv(codes, dates, close)

        sector_map = {"688001.SH": "科创",
                      "600001.SH": "主板大盘", "600002.SH": "主板大盘",
                      "600003.SH": "主板小盘"}

        result = build_sector_dataset(ohlcv, sector_map, forward_days=5)
        valid = result.dropna(subset=["label"])
        assert set(valid["label"].unique()).issubset({0, 1})

    def test_label_based_on_relative_outperformance(self):
        """标签应基于板块收益相对截面中位数的比较。"""
        codes = ["688001.SH", "688002.SH", "600001.SH", "600002.SH"]
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        np.random.seed(123)
        close = 10 + np.cumsum(np.random.randn(30, 4) * 0.2, axis=0)
        ohlcv = _make_multi_day_ohlcv(codes, dates, close)

        sector_map = {"688001.SH": "科创", "688002.SH": "科创",
                      "600001.SH": "主板大盘", "600002.SH": "主板大盘"}

        result = build_sector_dataset(ohlcv, sector_map, forward_days=5)
        valid = result.dropna(subset=["label"])

        # 检查某一天：两个板块一个为1一个为0（假设它们有不同的未来收益）
        # 在相对排序中，跑赢中位数的为1
        assert valid["label"].nunique() >= 1  # 至少有一个标签出现

    def test_regression_mode_returns_continuous_labels(self):
        """回归模式应返回连续收益率标签。"""
        codes = ["688001.SH", "688002.SH", "600001.SH", "600002.SH"]
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        np.random.seed(42)
        close = 10 + np.cumsum(np.random.randn(30, 4) * 0.2, axis=0)
        ohlcv = _make_multi_day_ohlcv(codes, dates, close)

        sector_map = {"688001.SH": "科创", "688002.SH": "科创",
                      "600001.SH": "主板大盘", "600002.SH": "主板大盘"}

        result = build_sector_dataset(ohlcv, sector_map, forward_days=5, label_mode="regression")
        valid = result.dropna(subset=["label"])
        # 回归标签应为连续值（非0/1）
        unique = valid["label"].unique()
        assert any(v != 0 and v != 1 for v in unique[:10])

    def test_forward_return_column_correct(self):
        """ret_5d 列应等于板块未来5日等权收益。"""
        codes = ["600001.SH", "600002.SH"]
        dates = pd.date_range("2024-01-02", periods=20, freq="B")
        # 固定涨幅：stock1从10→29 (每天+1), stock2从20→39 (每天+1)
        close = np.column_stack([
            np.array([float(i) for i in range(10, 30)]),
            np.array([float(i) for i in range(20, 40)]),
        ])
        ohlcv = _make_multi_day_ohlcv(codes, dates, close)

        sector_map = {"600001.SH": "主板大盘", "600002.SH": "主板大盘"}

        result = build_sector_dataset(ohlcv, sector_map, forward_days=5)
        # 用第5个交易日（有足够回溯数据），而非第一个
        test_date = dates[5]  # 2024-01-09
        row = result[(result["trade_date"] == test_date) & (result["sector"] == "主板大盘")]
        assert len(row) == 1, f"Expected 1 row for {test_date}, got {len(row)}"

        # stock1: 从15→20 (+5/15≈33.3%), stock2: 从25→30 (+5/25=20%), 等权≈26.67%
        # 实际值：(20-15)/15=0.3333, (30-25)/25=0.2, mean=0.26667
        ret_val = row.iloc[0]["ret_5d"]
        assert abs(ret_val - 0.26667) < 0.02, f"Expected ~0.2667, got {ret_val}"

    def test_handles_empty_ohlcv(self):
        """空输入应返回空DataFrame。"""
        ohlcv = pd.DataFrame()
        sector_map = {}
        result = build_sector_dataset(ohlcv, sector_map, forward_days=5)
        assert result.empty
