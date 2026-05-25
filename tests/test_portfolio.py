"""组合优化测试。"""
import pytest
import pandas as pd
import numpy as np
from portfolio.selector import select_top_n, filter_stocks
from portfolio.allocator import equal_weight, volatility_inverse_weight


class TestSelector:
    def test_select_top_n(self):
        """select_top_n 应从排序结果中选出得分最高的 N 只。"""
        scores = pd.DataFrame({
            "code": ["000001", "000002", "000003", "000004", "000005"],
            "score": [0.9, 0.7, 0.5, 0.3, 0.1],
            "rank": [1, 2, 3, 4, 5],
        })
        selected = select_top_n(scores, n=3)
        assert len(selected) == 3
        assert selected.iloc[0]["code"] == "000001"

    def test_filter_stocks_excludes_st(self):
        """应排除 ST 股票。"""
        stocks = pd.DataFrame({
            "code": ["000001", "000002", "000003"],
            "name": ["平安银行", "ST瑞德", "深振业"],
            "score": [0.9, 0.8, 0.7],
        })
        filtered = filter_stocks(stocks, exclude_st=True)
        assert "000002" not in filtered["code"].values

    def test_filter_stocks_excludes_new_listings(self):
        """应排除上市不足 60 天的次新股。"""
        stocks = pd.DataFrame({
            "code": ["000001", "000002"],
            "score": [0.9, 0.8],
            "list_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2026-05-01")],
        })
        ref_date = pd.Timestamp("2026-05-25")
        filtered = filter_stocks(stocks, ref_date=ref_date, min_list_days=60)
        assert "000002" not in filtered["code"].values


class TestAllocator:
    def test_equal_weight(self):
        """等权分配：N 只股票每只 1/N。"""
        result = equal_weight(["000001", "000002", "000003", "000004"], cash=1_000_000)
        assert len(result) == 4
        assert abs(result["weight"].sum() - 1.0) < 0.001
        assert result.iloc[0]["weight"] == 0.25

    def test_volatility_inverse_weight(self):
        """波动率倒数加权：低波动股票权重大。"""
        returns = pd.DataFrame({
            "000001": np.random.randn(100) * 0.01,
            "000002": np.random.randn(100) * 0.03,
        })
        result = volatility_inverse_weight(["000001", "000002"], returns, cash=1_000_000)
        assert len(result) == 2
        assert abs(result["weight"].sum() - 1.0) < 0.01
        # 000001 波动率更低，权重应更大
        assert result[result["code"] == "000001"]["weight"].iloc[0] > \
               result[result["code"] == "000002"]["weight"].iloc[0]
