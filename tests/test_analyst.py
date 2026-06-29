"""analyst 模块测试 — 7维深度分析报告生成器。"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from factors.analyst import (
    LEAF_DATA_SOURCE,
    _classify_leaf,
    _leaf_usage,
    _operator_usage,
    _tree_depth,
    _factor_correlation,
    generate_analysis_report,
    load_suggestions,
    apply_suggestions,
    save_analysis_report,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_data():
    """Generate a panel dataframe with stock OHLCV + industry + fwd_ret."""
    np.random.seed(42)
    codes = [f"{i:06d}" for i in range(50)]
    dates = pd.date_range("2025-01-02", "2025-03-31", freq="B")
    rows = []
    for d in dates:
        for code in np.random.choice(codes, 30, replace=False):
            rows.append({
                "code": code,
                "trade_date": d,
                "close": 10 + np.random.randn() * 2,
                "fwd_5d": np.random.randn() * 0.03,
                "industry_sw1": np.random.choice(["科技", "周期", "消费", "金融"]),
                "mcap": np.random.choice([10, 50, 200, 500]),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_population():
    """4-factor population with one duplicate."""
    return [
        ["close", "20", "ts_rank", "rank"],
        ["volume", "10", "ts_mean", "mcap", "zscore", "add"],
        ["close", "5", "ts_pct", "roe", "zscore", "mul", "rank"],
        ["close", "20", "ts_rank", "rank"],  # duplicate of f1
        ["north_net", "10", "ts_mean", "rank"],
    ]


@pytest.fixture
def sample_results():
    """4 factor results with correlated series (f1 and f4 share tokens)."""
    np.random.seed(42)
    base = pd.Series(np.random.randn(1000))
    base2 = pd.Series(np.random.randn(1000))
    return [
        {
            "name": "f1",
            "tokens": ["close", "20", "ts_rank", "rank"],
            "ic": 0.06,
            "series": base.copy(),
        },
        {
            "name": "f2",
            "tokens": ["volume", "10", "ts_mean", "mcap", "zscore", "add"],
            "ic": 0.04,
            "series": base2.copy(),
        },
        {
            "name": "f3",
            "tokens": ["close", "5", "ts_pct", "roe", "zscore", "mul", "rank"],
            "ic": 0.03,
            "series": pd.Series(np.random.randn(1000)),
        },
        {
            "name": "f4",
            "tokens": ["close", "20", "ts_rank", "rank"],
            "ic": 0.055,
            # f4 uses same tokens as f1; correlate with base but add some noise
            "series": base * 0.95 + pd.Series(np.random.randn(1000)) * 0.05,
        },
    ]


@pytest.fixture
def sample_db():
    """Minimal factor DB with 3 past rounds."""
    return {
        "rounds": 3,
        "history": [
            {
                "round": 1,
                "results": [{"ic": 0.03}, {"ic": -0.02}, {"ic": 0.05}],
            },
            {
                "round": 2,
                "results": [{"ic": 0.04}, {"ic": 0.06}, {"ic": -0.01}],
            },
            {
                "round": 3,
                "results": [{"ic": 0.055}, {"ic": 0.045}, {"ic": 0.035}],
            },
        ],
    }


@pytest.fixture
def sample_ml_result():
    """Mock ML result from train_lambdarank."""
    return {
        "model": None,
        "feature_importances": {
            "factor_a": 0.45,
            "factor_b": 0.30,
            "factor_c": 0.25,
        },
        "ndcg_score": 0.72,
        "backtest": {
            "bt_annual": 0.18,
            "bt_max_dd": -0.15,
            "bt_sharpe": 1.2,
            "bt_win_rate": 0.55,
            "bt_n_trades": 42,
            "bt_fitness": 0.65,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyLeaf:
    def test_known_leaf(self):
        assert _classify_leaf("close") == "price_volume"
        assert _classify_leaf("volume") == "price_volume"
        assert _classify_leaf("mcap") == "valuation"
        assert _classify_leaf("roe") == "financial"
        assert _classify_leaf("north_net") == "macro"
        assert _classify_leaf("bar0_ret") == "intraday"

    def test_prebuilt_factor(self):
        assert _classify_leaf("@my_factor") == "prebuilt_factor"
        assert _classify_leaf("@test") == "prebuilt_factor"

    def test_unknown_leaf(self):
        assert _classify_leaf("something_new") == "other"
        assert _classify_leaf("random_col") == "other"


class TestTreeDepth:
    def test_simple_rank(self):
        # rank is arity 1: pops 1, pushes 1
        assert _tree_depth(["close", "rank"]) == 1

    def test_ts_mean(self):
        # ts_mean is arity 2 (1 subtree + 1 int param)
        assert _tree_depth(["close", "20", "ts_mean"]) == 1

    def test_nested(self):
        # rank(ts_mean(close, 20))
        tokens = ["close", "20", "ts_mean", "rank"]
        assert _tree_depth(tokens) == 2

    def test_deep_nested(self):
        # rank(mul(ts_pct(close, 5), zscore(roe)))
        tokens = ["close", "5", "ts_pct", "roe", "zscore", "mul", "rank"]
        assert _tree_depth(tokens) == 3

    def test_add_two_branches(self):
        # add(zscore(close), zscore(volume))
        tokens = ["close", "zscore", "volume", "zscore", "add"]
        assert _tree_depth(tokens) == 2

    def test_ts_corr(self):
        # ts_corr has arity 3 (2 data + 1 int)
        tokens = ["close", "volume", "20", "ts_corr"]
        assert _tree_depth(tokens) == 1

    def test_empty(self):
        assert _tree_depth([]) == 0

    def test_leaf_only(self):
        assert _tree_depth(["close"]) == 0


class TestLeafUsage:
    def test_counts(self, sample_population):
        lu = _leaf_usage(sample_population)
        # Leaves: close appears 3x (f1, f3, f4), 20 appears 2x (f1, f4)
        assert lu["close"] == 3
        assert "20" in lu
        assert lu["20"] == 2
        # operators should NOT be in leaf usage
        assert "rank" not in lu
        assert "ts_mean" not in lu

    def test_empty_population(self):
        lu = _leaf_usage([])
        assert lu == {}


class TestOperatorUsage:
    def test_counts(self, sample_population):
        ou = _operator_usage(sample_population)
        # ts_rank appears in f1 and f4 (the duplicate): 2 times
        assert ou.get("ts_rank", 0) >= 2
        assert "close" not in ou  # close is a leaf

    def test_empty_population(self):
        ou = _operator_usage([])
        assert ou == {}


class TestFactorCorrelation:
    def test_detects_redundancy(self, sample_results):
        corr = _factor_correlation(sample_results)
        assert "redundant_pairs" in corr
        assert "high_corr_count" in corr
        # f1 and f4 share identical tokens and their series are highly correlated
        # (base * 0.95 + noise * 0.05)
        assert corr["high_corr_count"] >= 1

    def test_single_factor(self):
        r = [{"name": "solo", "tokens": ["close"], "ic": 0.05, "series": pd.Series([1, 2, 3])}]
        corr = _factor_correlation(r)
        assert corr["high_corr_count"] == 0
        assert corr["redundant_pairs"] == []

    def test_no_series(self):
        r = [
            {"name": "a", "ic": 0.05},
            {"name": "b", "ic": 0.03},
        ]
        corr = _factor_correlation(r)
        assert corr["high_corr_count"] == 0


class TestGenerateReport:
    def test_generates_all_dimensions(
        self, sample_data, sample_population, sample_results, sample_ml_result, sample_db
    ):
        report = generate_analysis_report(
            sample_data, sample_population, sample_results,
            sample_ml_result, sample_db, 4
        )
        dims = {
            "data_source_coverage", "factor_structure", "ic_decay",
            "regime_sensitivity", "factor_redundancy",
            "ml_feature_importance", "backtest_diagnostics",
        }
        assert set(report.keys()) == dims

    def test_data_source_coverage_counts(
        self, sample_data, sample_population, sample_results, sample_ml_result, sample_db
    ):
        report = generate_analysis_report(
            sample_data, sample_population, sample_results,
            sample_ml_result, sample_db, 4
        )
        dsc = report["data_source_coverage"]
        # source_pct should sum near 100
        total_pct = sum(dsc["source_pct"].values())
        assert abs(total_pct - 100.0) < 1.0
        # top_leaves should not be empty
        assert len(dsc["top_leaves"]) > 0

    def test_empty_population_no_crash(
        self, sample_data, sample_ml_result, sample_db
    ):
        """Empty population should produce a valid report without crashing."""
        report = generate_analysis_report(
            sample_data, [], [], sample_ml_result, sample_db, 1
        )
        assert "data_source_coverage" in report
        assert "factor_structure" in report
        # factor_structure should have zero depth
        assert report["factor_structure"]["avg_depth"] == 0.0
        assert report["factor_structure"]["avg_nodes"] == 0.0

    def test_handles_none_ml_result(
        self, sample_data, sample_population, sample_results, sample_db
    ):
        """None ml_result should not crash."""
        report = generate_analysis_report(
            sample_data, sample_population, sample_results,
            None, sample_db, 4
        )
        assert report["ml_feature_importance"]["ndcg_score"] == 0.0
        assert report["ml_feature_importance"]["importances"] == {}
        assert report["backtest_diagnostics"]["annual_return"] == 0.0

    def test_handles_no_industry_col(
        self, sample_population, sample_results, sample_ml_result, sample_db
    ):
        """df without industry_sw1 should note it."""
        df = pd.DataFrame({
            "code": ["000001"],
            "trade_date": [pd.Timestamp("2025-01-02")],
            "close": [10.0],
            "fwd_5d": [0.01],
        })
        report = generate_analysis_report(
            df, sample_population, sample_results,
            sample_ml_result, sample_db, 4
        )
        rs = report["regime_sensitivity"]
        assert "note" in rs["industry_ic"]
        assert "not available" in rs["industry_ic"]["note"]

    def test_ic_decay_trend(
        self, sample_data, sample_population, sample_results,
        sample_ml_result, sample_db
    ):
        report = generate_analysis_report(
            sample_data, sample_population, sample_results,
            sample_ml_result, sample_db, 4
        )
        trend = report["ic_decay"]["trend"]
        assert len(trend) == 3  # 3 history entries
        assert all("round" in t and "avg_abs_ic" in t for t in trend)


class TestSuggestions:
    def test_load_suggestions_not_found(self):
        result = load_suggestions("nonexistent_file.json")
        assert result is None

    def test_load_suggestions_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json {[[")
        result = load_suggestions(str(bad))
        assert result is None

    def test_apply_suggestions_force_data_source(self):
        pool = ["close", "volume", "roe", "north_net", "mcap"]
        probs = {"rank": 0.3, "zscore": 0.3, "ts_mean": 0.2, "add": 0.2}

        suggestions = {"force_data_source": ["financial", "macro"]}
        new_pool, new_probs = apply_suggestions(suggestions, pool, probs)

        # close & volume are price_volume -> not boosted
        # roe is financial -> boosted 3x extra (=4x total)
        # north_net is macro -> boosted 3x extra (=4x total)
        # mcap is valuation -> not boosted
        assert new_pool.count("roe") > 1
        assert new_pool.count("north_net") > 1
        assert new_pool.count("close") == 1  # unchanged
        assert new_pool.count("volume") == 1  # unchanged
        assert new_pool.count("mcap") == 1  # unchanged
        # probs unchanged
        assert new_probs == probs

    def test_apply_suggestions_kill_operator(self):
        pool = ["close", "volume"]
        probs = {"rank": 0.25, "zscore": 0.25, "ts_corr": 0.25, "add": 0.25}

        suggestions = {"kill_operator": ["ts_corr"]}
        new_pool, new_probs = apply_suggestions(suggestions, pool, probs)

        assert "ts_corr" not in new_probs
        assert abs(sum(new_probs.values()) - 1.0) < 1e-9
        # pool unchanged
        assert new_pool == pool

    def test_apply_suggestions_boost_operator(self):
        pool = ["close", "volume"]
        probs = {"rank": 0.3, "zscore": 0.5, "add": 0.2}

        suggestions = {"boost_operator": ["rank"]}
        new_pool, new_probs = apply_suggestions(suggestions, pool, probs)

        # rank doubled from 0.3 to 0.6
        assert new_probs["rank"] == 0.6
        assert new_probs["zscore"] == 0.5
        assert new_probs["add"] == 0.2

    def test_apply_suggestions_boost_leaf_prob(self):
        pool = ["close", "volume", "roe", "close"]
        probs = {"rank": 1.0}

        suggestions = {"boost_leaf_prob": {"close": 2.0}}
        new_pool, new_probs = apply_suggestions(suggestions, pool, probs)

        # close appears 2x in original -> 2 * 2 = 4 extra copies, total 6
        assert new_pool.count("close") == 6

    def test_apply_suggestions_all_combined(self):
        pool = ["close", "volume", "roe", "mcap"]
        probs = {"rank": 0.3, "zscore": 0.3, "ts_corr": 0.2, "add": 0.2}

        suggestions = {
            "force_data_source": ["financial"],
            "boost_operator": ["rank"],
            "kill_operator": ["ts_corr"],
        }
        new_pool, new_probs = apply_suggestions(suggestions, pool, probs)

        # force_data_source: roe is financial -> boosted 3x extra
        assert new_pool.count("roe") == 4  # 1 original + 3 extra

        # boost_operator: rank doubled
        assert new_probs.get("rank", 0) > 0.3

        # kill_operator: ts_corr removed
        assert "ts_corr" not in new_probs
        assert abs(sum(new_probs.values()) - 1.0) < 1e-9


class TestSaveReport:
    def test_save_and_reload_report(self, tmp_path, monkeypatch):
        """Roundtrip save -> load preserves data."""
        report = {
            "data_source_coverage": {"source_pct": {"price_volume": 60.0}, "top_leaves": {"close": 10}},
            "factor_structure": {"avg_depth": 2.0},
        }
        # Save to tmp_path/data/analysis_round_0001.json
        save_dir = tmp_path / "data"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Patch cwd-like behavior: save_analysis_report writes relative to cwd
        # We'll use monkeypatch to change os.getcwd and also patch os.makedirs
        # Actually, easier: just save and check contents directly
        filepath = str(save_dir / "analysis_round_0001.json")
        from factors.analyst import save_analysis_report
        # We need to redirect the save location. Let's monkeypatch the function
        # to use tmp_path.
        original_save = save_analysis_report

        def patched_save(report, round_num):
            path = str(save_dir / f"analysis_round_{round_num:04d}.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            return path

        monkeypatch.setattr("factors.analyst.save_analysis_report", patched_save)

        saved_path = save_analysis_report(report, 1)
        assert os.path.exists(saved_path)

        with open(saved_path) as f:
            loaded = json.load(f)
        assert loaded["data_source_coverage"]["source_pct"] == {"price_volume": 60.0}
        assert loaded["factor_structure"]["avg_depth"] == 2.0


class TestReusedFunctions:
    """Verify that reused helper functions work correctly with report integration."""

    def test_full_report_has_depth_vs_ic(
        self, sample_data, sample_population, sample_results, sample_ml_result, sample_db
    ):
        report = generate_analysis_report(
            sample_data, sample_population, sample_results,
            sample_ml_result, sample_db, 4
        )
        dvi = report["factor_structure"]["depth_vs_ic"]
        assert isinstance(dvi, list)
        assert len(dvi) > 0
        for entry in dvi:
            assert "name" in entry
            assert "depth" in entry
            assert "ic" in entry

    def test_redundancy_pairs_contain_names(
        self, sample_data, sample_population, sample_results, sample_ml_result, sample_db
    ):
        report = generate_analysis_report(
            sample_data, sample_population, sample_results,
            sample_ml_result, sample_db, 4
        )
        pairs = report["factor_redundancy"]["redundant_pairs"]
        for p in pairs:
            assert "f1" in p
            assert "f2" in p
            assert "spearman_corr" in p
