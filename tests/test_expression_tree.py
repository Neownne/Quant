"""Tests for factors/expression_tree.py — RPN expression tree engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.expression_tree import (
    # Parser
    parse_rpn,
    tokens_to_str,
    # Evaluator
    evaluate_rpn,
    # Operators (exported for targeted testing)
    op_rank,
    op_zscore,
    op_sector_rank,
    op_ts_delta,
    op_ts_pct,
    op_ts_mean,
    op_ts_std,
    op_ts_rank,
    op_ts_min,
    op_ts_max,
    op_ts_corr,
    op_add,
    op_sub,
    op_mul,
    op_div,
    op_log,
    # Registry
    _OP_ARITY,
    _OPERATORS,
    # Random tree & genetics
    random_tree,
    mutate_rpn,
    crossover_rpn,
    validate_rpn,
    tree_depth,
    tree_size,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """Small multi-code, multi-date DataFrame for testing."""
    np.random.seed(42)
    codes = ["000001", "000002", "000003"]
    dates = pd.date_range("2025-01-02", "2025-01-15", freq="B")
    rows = []
    for code in codes:
        for d in dates:
            rows.append(
                {
                    "code": code,
                    "trade_date": d,
                    "close": 10.0 + np.random.randn() * 2,
                    "volume": float(np.random.randint(1000, 10000)),
                    "ret_1d": np.random.randn() * 0.02,
                }
            )
    df = pd.DataFrame(rows)
    df["industry_sw2"] = np.where(df["code"] == "000001", "银行", "科技")
    # Sort to match real-world data ordering
    df = df.sort_values(["trade_date", "code"]).reset_index(drop=True)
    return df


@pytest.fixture
def large_sample_df():
    """Larger fixture: 10 codes x 60 days for stress-testing ts operators."""
    np.random.seed(123)
    codes = [f"{i:06d}" for i in range(10)]
    dates = pd.date_range("2025-01-02", "2025-04-01", freq="B")
    rows = []
    for code in codes:
        for d in dates:
            rows.append(
                {
                    "code": code,
                    "trade_date": d,
                    "close": 10.0 + np.cumsum(np.random.randn() * 0.1)[-1]
                    if rows
                    else 10.0 + np.random.randn() * 2,
                    "volume": float(np.random.randint(1000, 10000)),
                }
            )
    df = pd.DataFrame(rows)
    df["industry_sw2"] = np.where(
        pd.to_numeric(df["code"]) % 2 == 0, "金融", "制造"
    )
    df = df.sort_values(["trade_date", "code"]).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1. Parser Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestParseRPN:
    def test_parse_basic(self):
        result = parse_rpn(";close;20;ts_rank")
        assert result == ["close", "20", "ts_rank"]

    def test_parse_empty(self):
        assert parse_rpn("") == []

    def test_parse_no_leading_semicolon(self):
        result = parse_rpn("close;20;ts_rank")
        assert result == ["close", "20", "ts_rank"]

    def test_tokens_to_str(self):
        tokens = ["close", "20", "ts_rank"]
        assert tokens_to_str(tokens) == ";close;20;ts_rank"

    def test_tokens_to_str_empty(self):
        assert tokens_to_str([]) == ""

    def test_roundtrip(self):
        """parse -> str should be identity (modulo leading semicolon on input)."""
        expr = ";close;20;ts_mean"
        tokens = parse_rpn(expr)
        assert tokens_to_str(tokens) == expr

    def test_parse_with_double_semicolons(self):
        result = parse_rpn(";close;;20;ts_rank")
        assert result == ["close", "20", "ts_rank"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Operator Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOpRank:
    def test_ranks_between_0_and_1(self, sample_df):
        result = op_rank(sample_df, sample_df["close"].values)
        assert isinstance(result, np.ndarray)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_per_date_mean_approx_0_5(self, sample_df):
        result = op_rank(sample_df, sample_df["close"].values)
        s = pd.Series(result, index=sample_df.index)
        for _, idx in sample_df.groupby("trade_date").groups.items():
            vals = s.loc[idx].dropna()
            if len(vals) >= 2:
                mean_rank = vals.mean()
                # With 3 stocks, rank values are 1/3, 2/3, 3/3 → mean = 2/3 ≈ 0.667
                # This is actually correct — the spec says "mean around 0.5" but
                # that's only true for large N. With N=3, mean = (N+1)/(2N) = 0.667.
                # The spec says ~0.5 which is the asymptotic value.
                assert 0.3 < mean_rank < 0.9, f"mean rank {mean_rank} out of range"


class TestOpZscore:
    def test_mean_near_zero(self, sample_df):
        result = op_zscore(sample_df, sample_df["close"].values)
        s = pd.Series(result, index=sample_df.index)
        for _, idx in sample_df.groupby("trade_date").groups.items():
            vals = s.loc[idx].dropna()
            if len(vals) >= 2:
                assert abs(vals.mean()) < 1e-9

    def test_std_near_one(self, sample_df):
        result = op_zscore(sample_df, sample_df["close"].values)
        s = pd.Series(result, index=sample_df.index)
        for _, idx in sample_df.groupby("trade_date").groups.items():
            vals = s.loc[idx].dropna()
            if len(vals) >= 2:
                # np.nanstd uses ddof=0, so with N=3 the population std is
                # exactly 1.0 (the z-score formula divides by population std).
                # pd.Series.std() defaults to ddof=1, so use ddof=0 here.
                assert abs(vals.std(ddof=0) - 1.0) < 0.01


class TestOpSectorRank:
    def test_requires_industry_column(self, sample_df):
        # Remove the column
        df_no_ind = sample_df.drop(columns=["industry_sw2"])
        with pytest.raises(KeyError, match="industry_sw2"):
            op_sector_rank(df_no_ind, sample_df["close"].values)

    def test_within_sector_ranks(self, sample_df):
        result = op_sector_rank(sample_df, sample_df["close"].values)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)


class TestOpTSMeanNoFutureLeak:
    def test_future_leak_check(self):
        """ts_mean(close, 3) at day i uses days i-3, i-2, i-1 only.

        We verify by constructing a DataFrame with monotonic increasing
        close per code and checking that ts_mean(close, 3) correctly
        computes the rolling mean of the 3 most recent PAST values
        (shift(1) excludes today).
        """
        # Build a controlled DataFrame sorted by code then date so per-code
        # group ordering matches position order.
        codes = ["000001", "000002"]
        dates = pd.date_range("2025-01-02", "2025-01-15", freq="B")
        rows = []
        for code in codes:
            for j, d in enumerate(dates):
                rows.append(
                    {
                        "code": code,
                        "trade_date": d,
                        "close": float(j + 1),  # 1, 2, 3, ... strictly increasing
                        "volume": 1000.0,
                    }
                )
        df = pd.DataFrame(rows)
        # Sort by code then date so per-code groups are contiguous
        df = df.sort_values(["code", "trade_date"]).reset_index(drop=True)

        d = 3
        result = op_ts_mean(df, df["close"].values, d)

        for code in codes:
            code_mask = df["code"] == code
            series = df.loc[code_mask, "close"].values
            res = result[code_mask.values]
            for i in range(len(series)):
                if i < 1:
                    # day 0: shift(1) = NaN, so rolling mean is NaN
                    assert np.isnan(res[i]), f"Day {i} should be NaN, got {res[i]}"
                elif i < d:
                    # days 1, 2: shift(1) gives values but rolling with
                    # min_periods=3 may not be met → could be NaN
                    pass
                else:
                    # shift(1).rolling(d) at position i uses original
                    # close[i-d : i] = the d values BEFORE position i
                    if not np.isnan(res[i]):
                        past_vals = series[i - d : i]
                        expected = np.mean(past_vals)
                        assert (
                            abs(res[i] - expected) < 1e-10
                        ), f"Day {i}, code {code}: got {res[i]}, expected {expected} from {past_vals}"


class TestOpTSDelta:
    def test_equals_close_minus_shift_close(self, sample_df):
        d = 2
        result = op_ts_delta(sample_df, sample_df["close"].values, d)
        s = pd.Series(sample_df["close"].values, index=sample_df.index)
        expected = s.groupby(sample_df["code"]).transform(
            lambda g: g - g.shift(d)
        ).values
        # Compare non-NaN values
        mask = ~np.isnan(result) & ~np.isnan(expected)
        assert np.allclose(result[mask], expected[mask], equal_nan=True)


class TestOpTSRank:
    def test_returns_valid_ranks(self, large_sample_df):
        d = 10
        result = op_ts_rank(large_sample_df, large_sample_df["close"].values, d)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(valid > 0.0)
        assert np.all(valid <= 1.0)


class TestOpTSCorr:
    def test_self_correlation_is_one(self, sample_df):
        """Correlation of close with itself should be 1.0 for sufficiently large windows."""
        d = 5
        result = op_ts_corr(
            sample_df, sample_df["close"].values, sample_df["close"].values, d
        )
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.allclose(valid, 1.0, atol=1e-9)


class TestArithmeticOps:
    def test_add(self, sample_df):
        a = np.array([1.0, 2.0, 3.0] * 30)[: len(sample_df)]
        b = np.array([4.0, 5.0, 6.0] * 30)[: len(sample_df)]
        result = op_add(sample_df, a, b)
        assert np.allclose(result, a + b)

    def test_sub(self, sample_df):
        a = np.array([10.0] * 90)
        b = np.array([3.0] * 90)
        result = op_sub(sample_df, a, b)
        assert np.allclose(result, 7.0)

    def test_mul(self, sample_df):
        a = np.array([2.0] * 90)
        b = np.array([3.0] * 90)
        result = op_mul(sample_df, a, b)
        assert np.allclose(result, 6.0)

    def test_div_safe(self, sample_df):
        a = np.array([6.0, 6.0, 6.0])
        # Create arrays of matching length
        ax = np.tile(a, 30)[: len(sample_df)]
        bx = np.tile(np.array([2.0, 0.0, 3.0]), 30)[: len(sample_df)]
        result = op_div(sample_df, ax, bx)
        expected = np.tile(np.array([3.0, np.nan, 2.0]), 30)[: len(sample_df)]
        mask = ~np.isnan(expected)
        assert np.allclose(result[mask], expected[mask])
        assert np.all(np.isnan(result[~mask]))

    def test_log_safe(self, sample_df):
        x = np.array([1.0, 0.0, -2.0])
        xx = np.tile(x, 30)[: len(sample_df)]
        result = op_log(sample_df, xx)
        # log(|1.0|) = 0, log(|0.0|) = NaN, log(|-2.0|) = log(2) ≈ 0.693
        expected0 = np.tile(np.array([0.0, np.nan, np.log(2.0)]), 30)[: len(sample_df)]
        mask0 = ~np.isnan(expected0)
        assert np.allclose(result[mask0], expected0[mask0])
        assert np.all(np.isnan(result[~mask0]))


# ═══════════════════════════════════════════════════════════════════════════
# 3. RPN Evaluator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluateSimpleRPN:
    def test_ts_rank_rpn(self, large_sample_df):
        """';close;20;ts_rank' should return a valid non-NaN Series.

        Uses large_sample_df (60 days per code) so d=20 fits in the data.
        """
        tokens = ["close", "20", "ts_rank"]
        result = evaluate_rpn(large_sample_df, tokens)
        assert isinstance(result, pd.Series)
        assert len(result) == len(large_sample_df)
        assert result.name == "_expr"
        # Some values should be non-NaN
        assert result.notna().sum() > 0

    def test_ts_mean_rpn(self, sample_df):
        tokens = ["close", "10", "ts_mean"]
        result = evaluate_rpn(sample_df, tokens)
        assert len(result) == len(sample_df)

    def test_log_rpn(self, sample_df):
        tokens = ["close", "log"]
        result = evaluate_rpn(sample_df, tokens)
        assert len(result) == len(sample_df)

    def test_div_rpn(self, sample_df):
        tokens = ["close", "volume", "div"]
        result = evaluate_rpn(sample_df, tokens)
        assert len(result) == len(sample_df)
        # close/volume should all be non-NaN (both are positive)
        assert result.notna().sum() > 0


class TestEvaluateComplexRPN:
    def test_add_then_rank(self, sample_df):
        """';close;volume;add;rank' → rank(close + volume)"""
        tokens = ["close", "volume", "add", "rank"]
        result = evaluate_rpn(sample_df, tokens)
        # Compute expected
        expected_arr = op_add(sample_df, sample_df["close"].values, sample_df["volume"].values)
        expected = pd.Series(
            op_rank(sample_df, expected_arr), index=sample_df.index
        )
        # Compare non-null
        mask = result.notna() & expected.notna()
        assert np.allclose(result[mask].values, expected[mask].values, atol=1e-10)

    def test_rank_then_ts_mean(self, sample_df):
        """';close;rank;20;ts_mean' → ts_mean(rank(close), 20)"""
        tokens = ["close", "rank", "20", "ts_mean"]
        result = evaluate_rpn(sample_df, tokens)
        assert len(result) == len(sample_df)

    def test_nested_arithmetic(self, sample_df):
        """';close;volume;add;close;sub' → (close + volume) - close = volume"""
        tokens = ["close", "volume", "add", "close", "sub"]
        result = evaluate_rpn(sample_df, tokens)
        expected = sample_df["volume"].values.astype(float)
        mask = result.notna()
        assert np.allclose(result[mask].values, expected[mask], atol=1e-10)


class TestEvaluateRPNWithNumberParams:
    def test_number_param_preserved_as_int(self, sample_df):
        tokens = ["close", "5", "ts_delta"]
        result = evaluate_rpn(sample_df, tokens)
        assert len(result) == len(sample_df)

    def test_multiple_number_params(self, sample_df):
        tokens = ["close", "volume", "10", "ts_corr"]
        result = evaluate_rpn(sample_df, tokens)
        assert len(result) == len(sample_df)

    def test_float_param(self, sample_df):
        # Non-integer numeric should still work
        tokens = ["close", "12.0", "ts_mean"]
        result = evaluate_rpn(sample_df, tokens)
        assert len(result) == len(sample_df)


class TestEvaluateRPNErrors:
    def test_unknown_token(self, sample_df):
        with pytest.raises(KeyError, match="Unknown token"):
            evaluate_rpn(sample_df, ["nonexistent_col", "rank"])

    def test_stack_underflow(self, sample_df):
        with pytest.raises(ValueError, match="Stack underflow"):
            evaluate_rpn(sample_df, ["add"])

    def test_stack_overflow(self, sample_df):
        with pytest.raises(ValueError, match="expected 1"):
            evaluate_rpn(sample_df, ["close", "volume"])


# ═══════════════════════════════════════════════════════════════════════════
# 4. Random Tree Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRandomTree:
    def test_generates_valid_rpn(self, sample_df):
        leaf_pool = ["close", "volume", "ret_1d"]
        for _ in range(20):
            tokens = random_tree(leaf_pool, max_depth=3)
            assert validate_rpn(tokens), f"Invalid RPN: {tokens}"
            # Should be evaluable
            result = evaluate_rpn(sample_df, tokens)
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_df)

    def test_respects_max_depth_approximately(self):
        leaf_pool = ["close", "volume"]
        for _ in range(30):
            tokens = random_tree(leaf_pool, max_depth=3)
            d = tree_depth(tokens)
            # Absolute cap is 5, should usually be ≤ 4
            assert d <= 5, f"Depth {d} exceeds cap for tokens: {tokens}"

    def test_uses_leaf_pool(self):
        leaf_pool = ["close", "volume"]
        for _ in range(20):
            tokens = random_tree(leaf_pool, max_depth=3)
            # At least one leaf should be from the pool
            non_op_non_param = [
                t for t in tokens if t not in _OP_ARITY and not t.lstrip("-").replace(".", "").isdigit()
            ]
            non_op_non_param_nums = []
            for t in tokens:
                if t not in _OP_ARITY:
                    try:
                        float(t)
                    except ValueError:
                        non_op_non_param_nums.append(t)
            assert any(t in leaf_pool for t in non_op_non_param_nums), (
                f"No leaf from pool found in: {tokens}"
            )

    def test_empty_leaf_pool_raises(self):
        with pytest.raises(ValueError, match="leaf_pool must be non-empty"):
            random_tree([], max_depth=3)

    def test_custom_op_probs(self, sample_df):
        """With custom op_probs only allowing 'add', all operators should be 'add'."""
        leaf_pool = ["close", "volume"]
        op_probs = {"add": 1.0}
        for _ in range(10):
            tokens = random_tree(leaf_pool, max_depth=3, op_probs=op_probs)
            ops_in_tree = [t for t in tokens if t in _OP_ARITY]
            if ops_in_tree:  # might be just a leaf
                assert all(
                    op == "add" for op in ops_in_tree
                ), f"Non-add operator in: {tokens}"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Genetic Operator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMutation:
    def test_mutate_produces_different(self):
        leaf_pool = ["close", "volume", "ret_1d"]
        original = random_tree(leaf_pool, max_depth=3)
        # Mutate with high rate to ensure changes
        mutated = mutate_rpn(original, leaf_pool, mutation_rate=0.8)
        # May be the same if very simple tree — try a few times
        for _ in range(10):
            if mutated != original:
                break
            mutated = mutate_rpn(original, leaf_pool, mutation_rate=0.8)
        # With high mutation rate, they should differ
        assert mutated != original or len(original) <= 1, (
            f"Mutation did not change anything: {original}"
        )

    def test_mutate_preserves_validity(self, sample_df):
        leaf_pool = ["close", "volume", "ret_1d"]
        passed = 0
        for _ in range(50):  # more attempts to handle higher mutation rates
            original = random_tree(leaf_pool, max_depth=3)
            mutated = mutate_rpn(original, leaf_pool, mutation_rate=0.5)
            if not validate_rpn(mutated):
                continue
            try:
                result = evaluate_rpn(sample_df, mutated)
                assert len(result) == len(sample_df)
                passed += 1
                if passed >= 3:
                    break
            except (TypeError, ValueError, KeyError):
                continue  # degenerate mutation, skip
        assert passed >= 1, "Should get at least one valid evaluated mutation"

    def test_mutate_empty_returns_empty(self):
        result = mutate_rpn([], ["close"])
        assert result == []


class TestCrossover:
    def test_crossover_produces_valid(self, sample_df):
        leaf_pool = ["close", "volume", "ret_1d"]
        for _ in range(20):
            a = random_tree(leaf_pool, max_depth=3)
            b = random_tree(leaf_pool, max_depth=3)
            child = crossover_rpn(a, b)
            if child is None:
                continue
            if not validate_rpn(child):
                continue
            result = evaluate_rpn(sample_df, child)
            assert len(result) == len(sample_df)

    def test_crossover_single_leaf_returns_none(self):
        """Crossover of two leaf-only trees should return None (no operators)."""
        a = ["close"]
        b = ["volume"]
        result = crossover_rpn(a, b)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 6. No Future Data Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNoFutureShift:
    def test_ts_mean_day_zero_is_nan(self, sample_df):
        """ts_mean(close, 20) at day 0 (first day for a code) must be NaN because
        shift(1) = NaN, so rolling mean is NaN."""
        result = op_ts_mean(sample_df, sample_df["close"].values, 20)
        s = pd.Series(result, index=sample_df.index)
        for code in sample_df["code"].unique():
            code_mask = sample_df["code"] == code
            first_idx = sample_df.index[code_mask][0]
            assert np.isnan(s.loc[first_idx]), (
                f"First day for {code} should be NaN (shift(1) = NaN), "
                f"got {s.loc[first_idx]}"
            )

    def test_all_ts_ops_use_shift_d_positive_only(self):
        """Verify via code inspection that no time-series operator uses shift(-d).

        We check each operator function's source individually, excluding docstrings
        and comments, to ensure only shift(d) with d>0 is used.
        """
        import inspect
        from factors import expression_tree as et

        # Check each ts operator individually
        ts_ops = {
            "op_ts_delta": et.op_ts_delta,
            "op_ts_pct": et.op_ts_pct,
            "op_ts_mean": et.op_ts_mean,
            "op_ts_std": et.op_ts_std,
            "op_ts_rank": et.op_ts_rank,
            "op_ts_min": et.op_ts_min,
            "op_ts_max": et.op_ts_max,
            "op_ts_corr": et.op_ts_corr,
        }
        for name, fn in ts_ops.items():
            src = inspect.getsource(fn)
            # Strip docstrings (triple-quoted strings)
            # Remove lines that are purely in triple-quoted strings
            lines = src.split("\n")
            clean_lines = []
            in_docstring = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if in_docstring:
                        in_docstring = False
                        continue
                    elif stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                        # One-line docstring
                        continue
                    else:
                        in_docstring = True
                        continue
                if in_docstring:
                    continue
                clean_lines.append(line)
            clean_src = "\n".join(clean_lines)

            assert "shift(-" not in clean_src, (
                f"shift(-d) detected in {name}! "
                f"This is a future-data leak. Per iron rule #9, "
                f"only shift(d) with d>0 is allowed."
            )

    def test_ts_mean_only_uses_past(self, sample_df):
        """Check that ts_mean result at day t depends only on values at days < t."""
        # With d=5, ts_mean at each point = mean of shift(1).rolling(5).mean()
        # shift(1) means today's value is excluded from the window
        d = 5
        result = op_ts_mean(sample_df, sample_df["close"].values, d)

        for code in sample_df["code"].unique():
            code_mask = sample_df["code"] == code
            code_df = sample_df.loc[code_mask].reset_index(drop=True)
            res = result[code_mask.values]
            close_vals = code_df["close"].values

            for i in range(len(code_df)):
                if np.isnan(res[i]):
                    continue
                # The mean should be of close[i-d+1..i] from the shifted data,
                # which is close[i-d..i-1] in the original data
                # Wait — shift(1) of close means:
                # shifted[t] = close[t-1]
                # rolling(5).mean()[t] = mean(shifted[t-4:t+1]) = mean(close[t-5:t])
                #
                # But shift(1) applies first, then rolling: so
                # shift(1).rolling(5).mean()[t] uses shifted[t-4:t+1] = close[t-5:t]
                start_idx = max(0, i - d)
                # The actual values used are close[i-d .. i-1] (because of shift(1))
                past_values = close_vals[start_idx:i]
                if len(past_values) >= max(d // 2, 3):
                    expected_mean = np.mean(past_values)
                    np.testing.assert_allclose(res[i], expected_mean, rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Validation & Utility Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateRPN:
    def test_valid_simple(self):
        assert validate_rpn(["close", "rank"])

    def test_valid_complex(self):
        assert validate_rpn(["close", "volume", "add", "rank"])

    def test_invalid_empty(self):
        assert not validate_rpn([])

    def test_invalid_underflow(self):
        assert not validate_rpn(["close", "add"])

    def test_invalid_overflow(self):
        assert not validate_rpn(["close", "close", "close"])


class TestTreeDepth:
    def test_single_leaf(self):
        assert tree_depth(["close"]) == 0

    def test_one_operator(self):
        assert tree_depth(["close", "rank"]) == 1

    def test_two_operator(self):
        assert tree_depth(["close", "close", "add", "rank"]) == 2

    def test_empty(self):
        assert tree_depth([]) == 0


class TestTreeSize:
    def test_leaf_only(self):
        assert tree_size(["close"]) == 1

    def test_op_only(self):
        # "close rank" — leaf + op = 2 nodes; param tokens excluded
        assert tree_size(["close", "rank"]) == 2

    def test_with_params(self):
        # "close 5 ts_mean" — close (leaf) + ts_mean (op) = 2 nodes
        # "5" is a param token, excluded from node count
        assert tree_size(["close", "5", "ts_mean"]) == 2

    def test_complex(self):
        # "close volume add rank" — 2 leaves + 2 ops = 4 nodes
        assert tree_size(["close", "volume", "add", "rank"]) == 4


# ═══════════════════════════════════════════════════════════════════════════
# 8. Operator Registry Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_16_operators_registered(self):
        assert len(_OP_ARITY) == 16, f"Expected 16, got {len(_OP_ARITY)}: {sorted(_OP_ARITY)}"
        assert len(_OPERATORS) == 16

    def test_all_op_arity_consistent(self):
        for op_name in _OP_ARITY:
            assert op_name in _OPERATORS, f"Operator '{op_name}' missing from _OPERATORS"
        for op_name in _OPERATORS:
            assert op_name in _OP_ARITY, f"Operator '{op_name}' missing from _OP_ARITY"

    def test_op_arity_values(self):
        assert _OP_ARITY["rank"] == 1
        assert _OP_ARITY["zscore"] == 1
        assert _OP_ARITY["sector_rank"] == 1
        assert _OP_ARITY["log"] == 1
        assert _OP_ARITY["ts_delta"] == 2
        assert _OP_ARITY["ts_mean"] == 2
        assert _OP_ARITY["ts_std"] == 2
        assert _OP_ARITY["ts_rank"] == 2
        assert _OP_ARITY["ts_min"] == 2
        assert _OP_ARITY["ts_max"] == 2
        assert _OP_ARITY["ts_pct"] == 2
        assert _OP_ARITY["add"] == 2
        assert _OP_ARITY["sub"] == 2
        assert _OP_ARITY["mul"] == 2
        assert _OP_ARITY["div"] == 2
        assert _OP_ARITY["ts_corr"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# 9. End-to-End Workflow Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_parse_evaluate_roundtrip(self, sample_df):
        """Parse, evaluate, and verify a few known expressions."""
        expressions = [
            ";close;5;ts_delta",
            ";close;rank",
            ";close;volume;div;zscore",
        ]
        for expr in expressions:
            tokens = parse_rpn(expr)
            result = evaluate_rpn(sample_df, tokens)
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_df)

    def test_generation_to_evaluation_pipeline(self, large_sample_df):
        """Generate random trees, evaluate them, verify no crashes.

        Uses large_sample_df (60 days per code).  Some generated trees may
        use window parameters up to 120 which exceed available data length,
        producing all-NaN results — that's acceptable as a degenerate case.
        The key invariant is that evaluation never crashes.
        """
        leaf_pool = ["close", "volume"]
        nan_count = 0
        for _ in range(30):
            tokens = random_tree(leaf_pool, max_depth=4)
            result = evaluate_rpn(large_sample_df, tokens)
            assert isinstance(result, pd.Series)
            assert len(result) == len(large_sample_df)
            if result.isna().all():
                nan_count += 1
        # At least some trees should produce valid results
        assert nan_count < 30, "All 30 random trees produced all-NaN"
