"""RL 训练器测试。"""
import pytest
import numpy as np
import pandas as pd
from rl.trainer import walk_forward_train_rl
from rl.predictor import RLPredictor


def _make_dataset(n_days: int = 400, n_stocks: int = 30, n_factors: int = 5) -> pd.DataFrame:
    """构造带标签的因子数据集（模拟 build_factor_dataset 输出）。"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    codes = [f"{i:06d}.SH" for i in range(n_stocks)]

    records = []
    for date in dates:
        for code in codes:
            row = {"trade_date": date, "code": code}
            for j in range(n_factors):
                row[f"factor_{j}"] = np.random.randn()
            # 标签：部分依赖于因子值
            signal = row["factor_0"] * 0.3 + row["factor_1"] * 0.2 + np.random.normal(0, 0.5)
            row["label"] = int(signal > 0)
            row["ret_1d"] = np.random.normal(0.001, 0.02)
            records.append(row)

    return pd.DataFrame(records)


class TestWalkForwardTrainRL:
    """walk_forward_train_rl 测试。"""

    def test_returns_list_of_window_results(self):
        """应返回每窗口结果列表。"""
        df = _make_dataset(n_days=300, n_stocks=20)
        factor_cols = [f"factor_{i}" for i in range(5)]

        results = walk_forward_train_rl(
            df, factor_cols,
            train_years=1, val_years=1,
            total_timesteps=200,  # 小步数加快测试
        )
        assert isinstance(results, list)

    def test_each_window_has_predictor(self):
        """每窗口结果应含 RLPredictor。"""
        df = _make_dataset(n_days=300, n_stocks=20)
        factor_cols = [f"factor_{i}" for i in range(5)]

        results = walk_forward_train_rl(
            df, factor_cols,
            train_years=1, val_years=1,
            total_timesteps=200,
        )

        if results:
            for r in results:
                assert "ensemble" in r
                assert isinstance(r["ensemble"], RLPredictor)
                assert "active_cols" in r
                assert "train_end" in r
                assert "val_end" in r

    def test_predictor_produces_valid_output(self):
        """训练后的 RLPredictor 应能正确预测。"""
        df = _make_dataset(n_days=300, n_stocks=20)
        factor_cols = [f"factor_{i}" for i in range(5)]

        results = walk_forward_train_rl(
            df, factor_cols,
            train_years=1, val_years=1,
            total_timesteps=200,
        )

        if results:
            predictor = results[-1]["ensemble"]
            # 用一条日数据测试
            one_day = df[df["trade_date"] == df["trade_date"].iloc[-1]]
            pred = predictor.predict(one_day)
            assert "code" in pred.columns
            assert "score" in pred.columns
            assert "rank" in pred.columns

    def test_handles_empty_data(self):
        """空数据应返回空列表。"""
        results = walk_forward_train_rl(pd.DataFrame(), [], total_timesteps=100)
        assert results == []
