"""因子筛选测试。"""
import pytest
import pandas as pd
import numpy as np
from factors.screening import compute_factor_correlation, select_orthogonal_factors, filter_factors_by_ic


class TestICGate:
    def test_filter_by_ic_removes_noise_factor(self):
        """强信号因子通过 IC 门禁，噪声因子被淘汰。"""
        np.random.seed(42)
        n_dates = 300
        n_stocks = 100
        dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
        rows = []
        for i in range(n_stocks):
            # 每只股票的基准值不同，产生截面区分度
            base = np.random.randn()
            for j, d in enumerate(dates):
                rows.append({
                    "trade_date": d,
                    "good_factor": base + np.random.randn() * 0.01,
                    "noise_factor": np.random.randn(),
                    "ret_1d": base * 0.8 + np.random.randn() * 0.01,
                })
        df = pd.DataFrame(rows)
        passed = filter_factors_by_ic(
            df, ["good_factor", "noise_factor"],
            ret_col="ret_1d", ic_threshold=0.05, t_threshold=2.0,
        )
        assert "good_factor" in passed
        assert "noise_factor" not in passed

    def test_filter_empty_factor_list(self):
        """空列表应返回空列表。"""
        result = filter_factors_by_ic(pd.DataFrame(), [])
        assert result == []


class TestScreening:
    def test_compute_correlation_matrix(self):
        """应返回因子间相关性矩阵。"""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        })
        corr = compute_factor_correlation(df, ["f1", "f2", "f3"])
        assert corr.shape == (3, 3)
        assert abs(corr.loc["f1", "f1"] - 1.0) < 0.001

    def test_select_orthogonal_drops_highly_correlated(self):
        """相关性 > 0.9 的因子应被剔除。"""
        np.random.seed(42)
        n = 200
        base = np.random.randn(n)
        df = pd.DataFrame({
            "f1": base,
            "f2": base + np.random.randn(n) * 0.05,  # 与 f1 高相关
            "f3": np.random.randn(n),                  # 独立
        })
        selected = select_orthogonal_factors(df, ["f1", "f2", "f3"], threshold=0.9)
        # f1 与 f2 高相关，方差更大的 f2 被选入，f1 被剔除；f3 独立保留
        assert "f2" in selected
        assert "f3" in selected
        assert len(selected) == 2
        assert "f1" not in selected

    def test_empty_factor_list(self):
        """空列表应返回空列表。"""
        result = select_orthogonal_factors(pd.DataFrame(), [], threshold=0.7)
        assert result == []
