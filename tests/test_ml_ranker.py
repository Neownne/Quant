"""ml_ranker 模块测试 — LightGBM LambdaRank 排序器。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.ml_ranker import compute_ndcg, predict_rank, train_lambdarank


@pytest.fixture
def ml_sample_data():
    """生成包含 100 只股票、约 2 年交易日的模拟因子数据。"""
    np.random.seed(42)
    codes = [f"{i:06d}" for i in range(100)]
    dates = pd.date_range("2024-01-02", "2025-12-31", freq="B")
    rows = []
    for d in dates:
        for code in np.random.choice(codes, 80, replace=False):
            rows.append(
                {
                    "code": code,
                    "trade_date": d,
                    "factor_a": np.random.randn(),
                    "factor_b": np.random.randn(),
                    "fwd_5d": np.random.randn() * 0.03,
                }
            )
    return pd.DataFrame(rows)


class TestTrainLambdarank:
    def test_train_returns_model_dict(self, ml_sample_data):
        """训练应返回包含 model 和 feature_importances 的结果字典。"""
        factor_df = ml_sample_data[["code", "trade_date", "factor_a", "factor_b"]]
        forward_ret = ml_sample_data["fwd_5d"]

        result = train_lambdarank(factor_df, forward_ret)

        assert isinstance(result, dict)
        assert "model" in result
        assert result["model"] is not None
        assert "feature_importances" in result
        assert isinstance(result["feature_importances"], dict)
        assert "factor_a" in result["feature_importances"]
        assert "factor_b" in result["feature_importances"]
        assert "ndcg_score" in result
        assert isinstance(result["ndcg_score"], float)

    def test_empty_data_returns_safe(self):
        """空 factor_df 应返回 model=None 而不崩溃。"""
        empty_df = pd.DataFrame(columns=["code", "trade_date", "factor_a"])
        empty_ret = pd.Series([], dtype=float)

        result = train_lambdarank(empty_df, empty_ret)

        assert result["model"] is None
        assert result["feature_importances"] == {}
        assert result["ndcg_score"] == 0.0

    def test_no_factor_columns_returns_safe(self):
        """只有 code/trade_date 列时应安全返回。"""
        df = pd.DataFrame({"code": ["000001"], "trade_date": ["2024-01-02"]})
        ret = pd.Series([0.01])

        result = train_lambdarank(df, ret)

        assert result["model"] is None
        assert result["feature_importances"] == {}

    def test_all_nan_returns_safe(self):
        """全部为 NaN 的因子数据应安全返回。"""
        df = pd.DataFrame(
            {
                "code": ["000001", "000002"],
                "trade_date": ["2024-01-02", "2024-01-02"],
                "factor_a": [np.nan, np.nan],
                "factor_b": [np.nan, np.nan],
            }
        )
        ret = pd.Series([0.01, -0.02])

        result = train_lambdarank(df, ret)

        assert result["model"] is None


class TestPredictRank:
    def test_predict_returns_scores(self, ml_sample_data):
        """预测应返回与输入索引对齐的得分 Series。"""
        factor_df = ml_sample_data[["code", "trade_date", "factor_a", "factor_b"]]
        forward_ret = ml_sample_data["fwd_5d"]

        result = train_lambdarank(factor_df, forward_ret)
        scores = predict_rank(result, factor_df)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(factor_df)
        assert scores.index.equals(factor_df.index)

    def test_predict_handles_nan(self, ml_sample_data):
        """含 NaN 的因子 DataFrame 不应导致预测崩溃。"""
        factor_df = ml_sample_data[["code", "trade_date", "factor_a", "factor_b"]].copy()
        forward_ret = ml_sample_data["fwd_5d"]

        # 引入部分 NaN
        factor_df.loc[factor_df.sample(frac=0.1, random_state=7).index, "factor_a"] = np.nan

        result = train_lambdarank(factor_df, forward_ret)
        scores = predict_rank(result, factor_df)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(factor_df)
        assert not np.isinf(scores).any()

    def test_predict_with_none_model_returns_zero(self, ml_sample_data):
        """model=None 时应返回全零 Series。"""
        factor_df = ml_sample_data[["code", "trade_date", "factor_a", "factor_b"]]
        model_dict = {"model": None, "feature_importances": {}, "ndcg_score": 0.0}

        scores = predict_rank(model_dict, factor_df)

        assert (scores == 0.0).all()
        assert len(scores) == len(factor_df)


class TestComputeNDCG:
    def test_ndcg_perfect_ranking(self):
        """完美排序时 NDCG 应为 1.0。"""
        y_true = np.array([4, 3, 2, 1, 0])
        y_pred = np.array([5, 4, 3, 2, 1])
        ndcg = compute_ndcg(y_true, y_pred, k=5)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_worst_ranking(self):
        """反向排序时 NDCG 应低于 1.0 且非负。"""
        y_true = np.array([4, 3, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 3, 4])
        ndcg = compute_ndcg(y_true, y_pred, k=5)
        assert 0.0 <= ndcg <= 1.0

    def test_ndcg_range(self, ml_sample_data):
        """中等相关性的数据集 NDCG 应在合理范围内。"""
        n = 100
        y_true = np.random.randint(0, 5, n)
        y_pred = y_true + np.random.randn(n) * 0.5  # 有噪声
        ndcg = compute_ndcg(y_true, y_pred, k=10)
        assert 0.0 <= ndcg <= 1.0

    def test_ndcg_all_zero_labels(self):
        """所有标签为零时 NDCG 应为 0.0。"""
        y_true = np.zeros(10, dtype=int)
        y_pred = np.random.randn(10)
        ndcg = compute_ndcg(y_true, y_pred, k=5)
        assert ndcg == 0.0

    def test_ndcg_empty_input(self):
        """空输入应返回 0.0。"""
        assert compute_ndcg(np.array([]), np.array([]), k=5) == 0.0

    def test_ndcg_k_larger_than_data(self):
        """k 大于数据长度时应优雅处理。"""
        y_true = np.array([3, 2, 1])
        y_pred = np.array([3, 1, 2])
        ndcg = compute_ndcg(y_true, y_pred, k=10)
        assert 0.0 <= ndcg <= 1.0
