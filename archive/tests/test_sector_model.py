"""板块打分模型测试。"""
import pytest
import pandas as pd
import numpy as np
from models.sector_model import SectorScoringModel, walk_forward_train_sectors


def _make_sector_data(n_days: int = 100, n_sectors: int = 5) -> pd.DataFrame:
    """构造板块特征数据集。"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    sectors = ["科创", "北证", "红利", "主板大盘", "主板小盘"][:n_sectors]

    records = []
    for date in dates:
        for sector in sectors:
            row = {"trade_date": date, "sector": sector}
            # 模拟板块特征
            row["advance_decline_ratio"] = np.random.uniform(0.3, 3.0)
            row["n_advancers"] = np.random.randint(10, 100)
            row["n_decliners"] = np.random.randint(5, 80)
            row["n_limit_up"] = np.random.randint(0, 10)
            row["n_limit_down"] = np.random.randint(0, 5)
            row["up_volume_ratio"] = np.random.uniform(0.3, 0.7)
            row["sector_ret_mean"] = np.random.normal(0.001, 0.02)
            row["sector_ret_std"] = np.random.uniform(0.005, 0.03)
            row["sector_turnover_mean"] = np.random.uniform(0.5, 5.0)
            row["new_high_20d"] = np.random.randint(0, 20)
            row["new_low_20d"] = np.random.randint(0, 10)
            row["money_flow_pct"] = np.random.normal(0, 0.1)
            row["concentration_top3"] = np.random.uniform(0.2, 0.6)
            row["sector_mom_5"] = np.random.normal(0, 0.03)
            row["sector_mom_20"] = np.random.normal(0, 0.05)
            row["sector_vol_20"] = np.random.uniform(0.01, 0.04)

            # 标签：部分依赖于 sector_ret_mean（加入噪声）
            signal = row["sector_ret_mean"] + row["sector_mom_5"] * 2 + np.random.normal(0, 0.01)
            row["label"] = int(signal > 0)
            records.append(row)

    return pd.DataFrame(records)


FEATURE_COLS = [
    "advance_decline_ratio", "n_advancers", "n_decliners",
    "n_limit_up", "n_limit_down", "up_volume_ratio",
    "sector_ret_mean", "sector_ret_std", "sector_turnover_mean",
    "new_high_20d", "new_low_20d", "money_flow_pct",
    "concentration_top3", "sector_mom_5", "sector_mom_20", "sector_vol_20",
]


class TestSectorScoringModel:
    """SectorScoringModel 接口测试。"""

    def test_predict_returns_dataframe_with_sector_score_rank(self):
        """predict() 应返回 [sector, score, rank] 格式的 DataFrame。"""
        import xgboost as xgb

        df = _make_sector_data(n_days=50, n_sectors=3)
        X = df[FEATURE_COLS].fillna(0)
        y = df["label"].fillna(0)

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        sector_model = SectorScoringModel(
            model=model,
            feature_names=FEATURE_COLS,
            sector_col="sector",
        )

        # 预测单日数据（按天调用是预期用法）
        one_day = df[df["trade_date"] == df["trade_date"].iloc[0]]
        pred = sector_model.predict(one_day)

        assert "sector" in pred.columns
        assert "score" in pred.columns
        assert "rank" in pred.columns
        assert len(pred) == one_day["sector"].nunique()

    def test_predict_sorted_by_score_descending(self):
        """预测结果应按 score 降序排列。"""
        import xgboost as xgb

        df = _make_sector_data(n_days=50, n_sectors=5)
        X = df[FEATURE_COLS].fillna(0)
        y = df["label"].fillna(0)

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        sector_model = SectorScoringModel(model, FEATURE_COLS)
        one_day = df[df["trade_date"] == df["trade_date"].iloc[0]]
        pred = sector_model.predict(one_day)

        scores = pred["score"].values
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Not sorted: {scores[i]} < {scores[i+1]} at index {i}"

    def test_rank_starts_from_one(self):
        """排名应从1开始。"""
        import xgboost as xgb

        df = _make_sector_data(n_days=50, n_sectors=5)
        X = df[FEATURE_COLS].fillna(0)
        y = df["label"].fillna(0)

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        sector_model = SectorScoringModel(model, FEATURE_COLS)
        one_day = df[df["trade_date"] == df["trade_date"].iloc[0]]
        pred = sector_model.predict(one_day)

        assert pred["rank"].min() == 1
        assert pred["rank"].max() == len(pred)


class TestWalkForwardTrainSectors:
    """板块模型 walk-forward 训练测试。"""

    def test_returns_list_of_window_results(self):
        """应返回每窗口结果列表。"""
        df = _make_sector_data(n_days=600, n_sectors=5)
        results = walk_forward_train_sectors(
            df,
            feature_cols=FEATURE_COLS,
            train_years=1,
            val_years=1,
        )
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_each_window_has_model_and_metrics(self):
        """每窗口结果应含模型和评估指标。"""
        df = _make_sector_data(n_days=600, n_sectors=5)
        results = walk_forward_train_sectors(
            df,
            feature_cols=FEATURE_COLS,
            train_years=1,
            val_years=1,
        )

        for r in results:
            assert "model" in r, f"Missing 'model' key in window: {list(r.keys())}"
            assert "metrics" in r, f"Missing 'metrics' key in window"
            assert isinstance(r["model"], SectorScoringModel)

    def test_model_can_predict_after_training(self):
        """训练后的模型应能正确预测。"""
        df = _make_sector_data(n_days=600, n_sectors=5)
        results = walk_forward_train_sectors(
            df,
            feature_cols=FEATURE_COLS,
            train_years=1,
            val_years=1,
        )

        # 用最后窗口的模型预测（单日数据）
        last_model = results[-1]["model"]
        one_day = df[df["trade_date"] == df["trade_date"].iloc[-1]]
        pred = last_model.predict(one_day)
        assert len(pred) == df["sector"].nunique()
        assert pred["rank"].min() == 1

    def test_handles_empty_data(self):
        """空数据应返回空列表。"""
        df = pd.DataFrame()
        results = walk_forward_train_sectors(df, feature_cols=FEATURE_COLS)
        assert results == []
