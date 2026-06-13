"""RL 预测器测试。"""
import pytest
import numpy as np
import pandas as pd
import torch
from rl.models import StockScoreNet
from rl.predictor import RLPredictor


class TestRLPredictor:
    """RLPredictor 接口测试。"""

    @pytest.fixture
    def predictor(self):
        """创建一个预训练的 RLPredictor。"""
        net = StockScoreNet(n_factors=3, n_context=4)
        net.eval()
        factor_names = ["factor_0", "factor_1", "factor_2"]
        return RLPredictor(policy_net=net, factor_names=factor_names, device="cpu")

    @pytest.fixture
    def sample_factor_df(self):
        """构造测试用因子 DataFrame。"""
        np.random.seed(42)
        n_stocks = 20
        return pd.DataFrame({
            "code": [f"{i:06d}.SH" for i in range(n_stocks)],
            "factor_0": np.random.randn(n_stocks),
            "factor_1": np.random.randn(n_stocks),
            "factor_2": np.random.randn(n_stocks),
        })

    def test_predict_returns_dataframe_with_code_score_rank(self, predictor, sample_factor_df):
        """predict() 应返回 [code, score, rank] 格式的 DataFrame。"""
        result = predictor.predict(sample_factor_df)

        assert "code" in result.columns
        assert "score" in result.columns
        assert "rank" in result.columns
        assert len(result) == len(sample_factor_df)

    def test_predict_sorted_by_score_descending(self, predictor, sample_factor_df):
        """预测结果应按 score 降序排列。"""
        result = predictor.predict(sample_factor_df)

        scores = result["score"].values
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Not sorted: idx={i}"

    def test_rank_starts_from_one(self, predictor, sample_factor_df):
        """rank 应从 1 开始。"""
        result = predictor.predict(sample_factor_df)
        assert result["rank"].min() == 1
        assert result["rank"].max() == len(sample_factor_df)

    def test_scores_in_valid_range(self, predictor, sample_factor_df):
        """评分应在 [0, 1] 范围内。"""
        result = predictor.predict(sample_factor_df)
        assert result["score"].min() >= 0.0
        assert result["score"].max() <= 1.0

    def test_deterministic_in_eval_mode(self, predictor, sample_factor_df):
        """评估模式下两次预测应一致。"""
        result1 = predictor.predict(sample_factor_df)
        result2 = predictor.predict(sample_factor_df)
        np.testing.assert_array_almost_equal(result1["score"].values, result2["score"].values)

    def test_handles_missing_factors(self, sample_factor_df):
        """缺失因子时应填充0且不崩溃。"""
        net = StockScoreNet(n_factors=3, n_context=4)
        net.eval()
        predictor = RLPredictor(policy_net=net, factor_names=["factor_0", "factor_1", "factor_2"], device="cpu")

        # 去掉一个因子列
        df_missing = sample_factor_df.drop(columns=["factor_1"])
        result = predictor.predict(df_missing)
        assert len(result) == len(df_missing)
        assert "score" in result.columns

    def test_empty_dataframe(self, predictor):
        """空 DataFrame 应返回空结果。"""
        df = pd.DataFrame(columns=["code", "factor_0", "factor_1", "factor_2"])
        result = predictor.predict(df)
        assert result.empty
