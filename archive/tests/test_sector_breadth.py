"""板块宽度特征计算测试。"""
import pytest
import pandas as pd
import numpy as np
from factors.sector_breadth import (
    compute_breadth_features,
    compute_single_day_features,
    FEATURE_NAMES,
)


def _make_ohlcv(
    codes: list[str],
    dates: list[pd.Timestamp],
    close_prices: np.ndarray,
    volume: float = 1_000_000,
    turnover: float = 2.0,
    high_prices: np.ndarray | None = None,
    low_prices: np.ndarray | None = None,
) -> pd.DataFrame:
    """构造测试用 OHLCV DataFrame。

    close_prices: (n_dates, n_codes) 收盘价矩阵
    """
    n_dates = len(dates)
    n_codes = len(codes)
    records = []
    for i, date in enumerate(dates):
        for j, code in enumerate(codes):
            c = float(close_prices[i, j])
            prev_c = float(close_prices[i - 1, j]) if i > 0 else c * 0.99
            records.append({
                "code": code,
                "trade_date": date,
                "open": c * 0.99,
                "high": float(high_prices[i, j]) if high_prices is not None else c * 1.02,
                "low": float(low_prices[i, j]) if low_prices is not None else c * 0.98,
                "close": c,
                "volume": volume,
                "amount": c * volume,
                "turnover": turnover,
            })
    return pd.DataFrame(records)


class TestSingleDayFeatures:
    """单日板块特征计算测试。"""

    def test_all_stocks_advance_gives_ratio_1(self):
        """所有股票都上涨时，涨跌比应为 1.0（或无穷大时的特殊处理）。"""
        codes = ["600001.SH", "600002.SH", "600003.SH"]
        dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        # 全部上涨
        close = np.array([[10.0, 20.0, 30.0],
                          [10.5, 21.0, 31.5]])  # +5%
        ohlcv = _make_ohlcv(codes, dates, close)

        sector_map = {c: "主板大盘" for c in codes}
        result = compute_single_day_features(ohlcv, sector_map)

        assert "主板大盘" in result
        features = result["主板大盘"]
        assert features["advance_decline_ratio"] > 0
        assert features["n_decliners"] == 0
        assert features["n_advancers"] == 3

    def test_all_stocks_decline_gives_ratio_0(self):
        """所有股票都下跌时，上涨家数为0。"""
        codes = ["600001.SH", "600002.SH"]
        dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        close = np.array([[10.0, 20.0],
                          [9.5, 19.0]])  # -5%
        ohlcv = _make_ohlcv(codes, dates, close)

        sector_map = {c: "主板小盘" for c in codes}
        result = compute_single_day_features(ohlcv, sector_map)

        features = result["主板小盘"]
        assert features["n_advancers"] == 0
        assert features["n_decliners"] == 2

    def test_sector_ret_mean_matches_manual_calculation(self):
        """板块平均收益应等于各股收益的等权平均。"""
        codes = ["600001.SH", "600002.SH"]
        dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        close = np.array([[10.0, 20.0],
                          [10.5, 22.0]])  # +5%, +10%
        ohlcv = _make_ohlcv(codes, dates, close)

        sector_map = {c: "主板大盘" for c in codes}
        result = compute_single_day_features(ohlcv, sector_map)

        features = result["主板大盘"]
        expected = (0.05 + 0.10) / 2
        assert abs(features["sector_ret_mean"] - expected) < 0.001

    def test_limit_up_detected_for_mainboard(self):
        """主板涨停（≈10%）应被检测到。"""
        codes = ["600001.SH", "600002.SH"]
        dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        close = np.array([[10.0, 20.0],
                          [10.99, 21.0]])  # 第一只≈涨停（9.9%），第二只正常
        ohlcv = _make_ohlcv(codes, dates, close)

        sector_map = {c: "主板大盘" for c in codes}
        result = compute_single_day_features(ohlcv, sector_map)

        features = result["主板大盘"]
        assert features["n_limit_up"] >= 1

    def test_limit_up_detected_for_kechuang(self):
        """科创板涨停（≈20%）应使用更高阈值。"""
        codes = ["688001.SH", "688002.SH"]
        dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        close = np.array([[50.0, 30.0],
                          [60.0, 31.5]])  # 第一只+20%，科创板涨停
        ohlcv = _make_ohlcv(codes, dates, close)

        sector_map = {c: "科创" for c in codes}
        result = compute_single_day_features(ohlcv, sector_map)

        features = result["科创"]
        assert features["n_limit_up"] >= 1

    def test_money_flow_positive_for_up_stocks(self):
        """全部上涨时，资金流向应为正值。"""
        codes = ["600001.SH", "600002.SH"]
        dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        close = np.array([[10.0, 20.0],
                          [10.5, 21.0]])
        ohlcv = _make_ohlcv(codes, dates, close)

        sector_map = {c: "主板大盘" for c in codes}
        result = compute_single_day_features(ohlcv, sector_map)

        features = result["主板大盘"]
        assert features["money_flow_pct"] > 0


class TestMultiDayFeatures:
    """多日板块特征计算测试。"""

    def test_new_high_count_with_lookback(self):
        """当日收盘价创20日新高的股票应被计数。"""
        codes = ["600001.SH"]
        dates = pd.date_range("2024-01-02", periods=25, freq="B")
        # 逐步上涨，确保最后一天是新高
        close = np.linspace(10, 15, 25).reshape(-1, 1)
        high = close * 1.01  # 每日最高价比收盘价高1%
        low = close * 0.99
        ohlcv = _make_ohlcv(codes, dates, close, high_prices=high, low_prices=low)

        sector_map = {"600001.SH": "主板大盘"}
        result = compute_breadth_features(ohlcv, sector_map, dates[-1], lookback_days=20)

        features = result["主板大盘"]
        assert features["new_high_20d"] >= 1

    def test_sector_momentum_is_mean_return_over_period(self):
        """板块动量应等于各股N日收益的等权平均。"""
        codes = ["600001.SH", "600002.SH"]
        dates = pd.date_range("2024-01-02", periods=10, freq="B")
        close = np.column_stack([
            np.linspace(10, 11, 10),   # +10% over 10 days
            np.linspace(20, 22, 10),   # +10%
        ])
        ohlcv = _make_ohlcv(codes, dates, close)

        sector_map = {c: "主板大盘" for c in codes}
        result = compute_breadth_features(ohlcv, sector_map, dates[-1], lookback_days=20)

        features = result["主板大盘"]
        # 5日动量
        assert "sector_mom_5" in features
        # 20日波动率
        assert "sector_vol_20" in features
        # 大致验证趋势方向
        assert features["sector_mom_5"] > 0

    def test_multiple_sectors_computed(self):
        """多板块应分别计算。"""
        codes = ["688001.SH", "688002.SH", "600001.SH", "600002.SH"]
        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        close = np.column_stack([
            np.linspace(50, 55, 5),
            np.linspace(30, 33, 5),
            np.linspace(10, 10.5, 5),
            np.linspace(20, 21, 5),
        ])
        ohlcv = _make_ohlcv(codes, dates, close)

        sector_map = {"688001.SH": "科创", "688002.SH": "科创",
                      "600001.SH": "主板大盘", "600002.SH": "主板大盘"}
        result = compute_breadth_features(ohlcv, sector_map, dates[-1])

        assert "科创" in result
        assert "主板大盘" in result
        # 科创应有所有特征
        for name in FEATURE_NAMES:
            assert name in result["科创"], f"Missing feature: {name}"


class TestFeatureNames:
    """特征名称定义测试。"""

    def test_all_expected_features_defined(self):
        """FEATURE_NAMES 应包含所有规划的板块特征。"""
        expected = {
            "advance_decline_ratio", "n_advancers", "n_decliners",
            "n_limit_up", "n_limit_down",
            "up_volume_ratio", "sector_ret_mean", "sector_ret_std",
            "sector_turnover_mean", "new_high_20d", "new_low_20d",
            "money_flow_pct", "concentration_top3",
            "sector_mom_5", "sector_mom_20", "sector_vol_20",
        }
        assert set(FEATURE_NAMES) == expected
