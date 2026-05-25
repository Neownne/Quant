"""ML 模型测试。"""
import pytest
import pandas as pd
import numpy as np
from models.dataset import (
    build_factor_dataset,
    make_labels,
    walk_forward_split,
)


class TestDataset:
    def test_make_labels_binary(self):
        """make_labels 应为每个交易日计算 T+1 涨跌标签。"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "code": ["000001"] * n,
            "trade_date": pd.date_range("2020-01-02", periods=n, freq="B"),
            "close": 10 + np.cumsum(np.random.randn(n) * 0.5),
        })
        result = make_labels(df, forward_days=1, mode="binary")
        assert "label" in result.columns
        # 最后一天没有未来数据，标签应为 NaN
        assert pd.isna(result["label"].iloc[-1])
        # 前面应有 0/1 标签
        labels = result["label"].dropna()
        assert labels.isin([0, 1]).all()

    def test_make_labels_regression(self):
        """回归模式应返回连续收益率。"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "code": ["000001"] * n,
            "trade_date": pd.date_range("2020-01-02", periods=n, freq="B"),
            "close": 10 + np.cumsum(np.random.randn(n) * 0.5),
        })
        result = make_labels(df, forward_days=1, mode="regression")
        assert "label" in result.columns
        assert result["label"].iloc[-2] is not np.nan  # second to last has 1 forward day

    def test_walk_forward_split(self):
        """walk_forward_split 应生成 (train, val) 对的迭代器。"""
        dates = pd.date_range("2018-01-02", "2023-12-29", freq="B")
        df = pd.DataFrame({"trade_date": dates, "value": range(len(dates))})

        splits = list(walk_forward_split(df, train_years=3, val_years=1))
        assert len(splits) >= 2  # 至少 2 个窗口 (2018-20→2021, 2019-21→2022)
        train, val = splits[0]
        assert len(train) > len(val)
        # train 和 val 的时间不应重叠
        assert train["trade_date"].max() < val["trade_date"].min()

    def test_build_factor_dataset_smoke(self):
        """端到端冒烟测试：从少量股票的 OHLCV 构建因子数据集。"""
        # 构造 20 只股票 × 200 个交易日的模拟数据
        np.random.seed(42)
        dates = pd.date_range("2020-01-02", periods=200, freq="B")
        codes = [f"{i:06d}" for i in range(20)]
        rows = []
        for code in codes:
            close = 10 + np.cumsum(np.random.randn(200) * 0.5)
            for i, d in enumerate(dates):
                rows.append({
                    "code": code,
                    "trade_date": d.date(),
                    "open": close[i] * (1 + np.random.randn() * 0.01),
                    "high": close[i] * (1 + abs(np.random.randn()) * 0.02),
                    "low": close[i] * (1 - abs(np.random.randn()) * 0.02),
                    "close": close[i],
                    "volume": np.random.randint(100000, 1000000),
                })
        ohlcv = pd.DataFrame(rows)

        result = build_factor_dataset(
            ohlcv,
            factor_names=["rsi_14", "mom_20", "vol_20"],
            label_mode="binary",
        )

        assert "label" in result.columns
        assert "rsi_14" in result.columns
        assert "mom_20" in result.columns
        assert "vol_20" in result.columns
        assert "code" in result.columns
        assert "trade_date" in result.columns
        # 应有足够有效行（200天 × 20只，warmup 后）
        assert len(result.dropna()) > 1000


class TestTrainer:
    def test_train_xgboost_binary(self):
        """用模拟数据训练 XGBoost 二分类器，应返回模型和指标。"""
        from models.trainer import train_xgboost

        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        })
        y = (X["f1"] + X["f2"] * 0.5 > 0).astype(int)

        model, metrics = train_xgboost(X, y, X, y)
        assert model is not None
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5  # better than random

    def test_train_xgboost_returns_feature_importance(self):
        """应返回特征重要性 DataFrame。"""
        from models.trainer import train_xgboost

        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
        })
        y = (X["f1"] > 0).astype(int)

        model, metrics = train_xgboost(X, y, X, y)
        assert "feature_importance" in metrics
        fi = metrics["feature_importance"]
        assert "f1" in fi.index
        assert "f2" in fi.index

    def test_train_lightgbm(self):
        """LightGBM 也应能训练并返回模型。"""
        from models.trainer import train_lightgbm

        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
        })
        y = (X["f1"] + X["f2"] * 0.3 > 0).astype(int)

        model, metrics = train_lightgbm(X, y, X, y)
        assert model is not None
        assert "accuracy" in metrics


class TestPredictor:
    def test_predict_returns_scores(self):
        """预测器应返回每只股票的得分和排名。"""
        from models.predictor import DailyPredictor
        from models.trainer import train_xgboost
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 300
        X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        y = (X["f1"] * 0.7 + X["f2"] * 0.3 > 0).astype(int)

        model, _ = train_xgboost(X, y, X, y)
        predictor = DailyPredictor(model, factor_names=["f1", "f2"])

        # 模拟今日横截面
        today_data = pd.DataFrame({
            "code": [f"{i:06d}" for i in range(50)],
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
        })

        result = predictor.predict(today_data)
        assert "code" in result.columns
        assert "score" in result.columns
        assert "rank" in result.columns
        assert result["rank"].min() == 1
        assert result["rank"].max() == 50

    def test_predict_handles_missing_factors(self):
        """缺失因子列时应报清晰错误。"""
        from models.predictor import DailyPredictor
        import pytest

        predictor = DailyPredictor(None, factor_names=["f1", "f2"])
        bad_data = pd.DataFrame({"code": ["000001"], "f1": [1.0]})  # missing f2

        with pytest.raises(KeyError):
            predictor.predict(bad_data)
