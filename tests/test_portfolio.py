"""组合优化测试。"""
import pytest
import pandas as pd
import numpy as np
from portfolio.selector import select_top_n, select_topk_ndrop, filter_stocks
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


class TestNDrop:
    """TopK + NDrop 增量调仓测试。"""

    @staticmethod
    def _make_scores(codes: list[str]) -> pd.Series:
        """按给定顺序构造降序 scores，第一条得分最高。"""
        return pd.Series(
            [1.0 - i * 0.01 for i in range(len(codes))],
            index=codes,
        )

    def test_first_day_buys_top_k(self):
        """首次建仓：无持仓时买入得分最高的 K 只。"""
        scores = self._make_scores(["A", "B", "C", "D", "E"])
        new_holdings, to_buy, to_sell = select_topk_ndrop(
            scores, current_holdings=set(), K=3, N=1
        )
        assert new_holdings == {"A", "B", "C"}
        assert to_buy == {"A", "B", "C"}
        assert to_sell == set()

    def test_first_day_none_holdings(self):
        """current_holdings=None 等同于空集合。"""
        scores = self._make_scores(["A", "B", "C", "D"])
        new_holdings, to_buy, to_sell = select_topk_ndrop(
            scores, current_holdings=None, K=3, N=1
        )
        assert new_holdings == {"A", "B", "C"}
        assert to_buy == {"A", "B", "C"}

    def test_keep_top_holdings_swap_worst(self):
        """持仓 A/B/C (scores: A=0.9, B=0.5, C=0.3)，D 得分 0.8 高于 B/C。
        K=3, N=1: 保留 A(0.9)，卖出最低的 C(0.3)，买入 D(0.8)。
        """
        scores = pd.Series(
            [0.95, 0.85, 0.60, 0.40],
            index=["D", "A", "B", "C"],
        )
        current = {"A", "B", "C"}
        new_holdings, to_buy, to_sell = select_topk_ndrop(
            scores, current_holdings=current, K=3, N=1
        )
        # A 和 B 得分最高（在持仓中），C 得分最低被替换
        assert "A" in new_holdings
        assert "B" in new_holdings
        assert "C" in to_sell
        assert "D" in to_buy
        assert len(new_holdings) == 3

    def test_no_change_when_holdings_top(self):
        """持仓恰好是得分最高的 K 只 → NDrop 仍会替换最差的 N 只。
        K=3, N=1: 持仓 {A,B,C} 是 top3，但仍卖出最差的 C，买入 D。"""
        scores = self._make_scores(["A", "B", "C", "D", "E"])
        current = {"A", "B", "C"}
        new_holdings, to_buy, to_sell = select_topk_ndrop(
            scores, current_holdings=current, K=3, N=1
        )
        # K-N=2: 保留 A, B; 卖出 C; 买入 D
        assert "A" in new_holdings
        assert "B" in new_holdings
        assert "C" in to_sell
        assert "D" in to_buy
        assert len(new_holdings) == 3
        assert len(to_buy) == 1
        assert len(to_sell) == 1

    def test_drops_delisted_holdings(self):
        """持仓中有已退市股票（不在 scores 中）→ 自动清掉并补位。"""
        scores = self._make_scores(["A", "B", "C", "D", "E"])
        current = {"A", "X", "Y"}  # X, Y 已退市
        new_holdings, to_buy, to_sell = select_topk_ndrop(
            scores, current_holdings=current, K=3, N=1
        )
        assert "X" in to_sell
        assert "Y" in to_sell
        assert len(new_holdings) == 3
        # A 保留（K-N=2, A 在 alive 中得分最高），B 和 C 补位
        assert "A" in new_holdings
        assert len(to_buy) == 2

    def test_small_candidate_pool(self):
        """候选池不足 K 只时，有多少买多少。"""
        scores = self._make_scores(["A", "B"])
        current = {"B"}
        new_holdings, to_buy, to_sell = select_topk_ndrop(
            scores, current_holdings=current, K=5, N=2
        )
        assert new_holdings == {"A", "B"}
        assert to_buy == {"A"}
        assert to_sell == set()

    def test_n_equals_k_replaces_all(self):
        """N=K 时等同于全量换仓。"""
        scores = self._make_scores(["D", "E", "F", "A", "B", "C"])
        current = {"A", "B", "C"}
        new_holdings, to_buy, to_sell = select_topk_ndrop(
            scores, current_holdings=current, K=3, N=3
        )
        assert new_holdings == {"D", "E", "F"}
        assert to_buy == {"D", "E", "F"}
        assert to_sell == {"A", "B", "C"}

    def test_multiple_days_simulation(self):
        """模拟多日连续调仓，验证持仓规模始终为 K。"""
        all_codes = [f"S{i:03d}" for i in range(20)]
        rng = np.random.default_rng(42)
        holdings = set()
        K, N = 8, 1

        for day in range(50):
            # 随机打乱得分
            shuffled = all_codes.copy()
            rng.shuffle(shuffled)
            scores = pd.Series(
                np.linspace(1.0, 0.0, len(shuffled)),
                index=shuffled,
            )
            holdings, to_buy, to_sell = select_topk_ndrop(
                scores, current_holdings=holdings, K=K, N=N
            )
            # 持仓数不应超过 K
            assert len(holdings) <= K
            # 持仓都应在候选池中
            assert holdings.issubset(set(all_codes))
            # 非首日后，每日最多替换 N 只（除非有持仓缺失）
            if day > 0:
                assert len(to_buy) <= N or len(to_sell) <= N

        # 50 天后持仓应为 K
        assert len(holdings) == K


class TestRisk:
    def test_stop_loss_triggers(self):
        """跌幅超过阈值应触发止损。"""
        from portfolio.risk import apply_stop_loss
        positions = pd.DataFrame({"code": ["000001", "000002"]})
        prices = {"000001": 92.0, "000002": 105.0}
        cost_basis = {"000001": 100.0, "000002": 100.0}  # 000001 -8%

        result = apply_stop_loss(positions, prices, cost_basis, stop_pct=0.08)
        assert "000001" in result["code"].values
        assert "000002" not in result["code"].values

    def test_drawdown_limit(self):
        """回撤超限应触发预警。"""
        from portfolio.risk import check_drawdown_limit
        assert check_drawdown_limit(75.0, 100.0, 0.25)  # 25% drawdown → True
        assert not check_drawdown_limit(80.0, 100.0, 0.25)  # 20% → False


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
