"""Genetic Programming Expression Tree Engine for Factor Discovery.

RPN-based expression trees with 15 operators covering cross-sectional,
time-series, and arithmetic transformations.  Operators use a unified
signature op(df, *args) where df provides code/trade_date/index context.

All time-series operators use shift(1) for the first lag, shifting data
FORWARD so today sees only past information.  Per iron rule #9, shift(-d)
is NEVER used.

Example
-------
>>> from factors.expression_tree import parse_rpn, evaluate_rpn, tokens_to_str
>>> tokens = parse_rpn(";close;20;ts_mean")
>>> result = evaluate_rpn(df, tokens)   # Rolling mean of shifted close

>>> from factors.expression_tree import random_tree, mutate_rpn, crossover_rpn
>>> tree = random_tree(["close", "volume", "ret_1d"], max_depth=3)
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# 1. Operator Arity Map
# ═══════════════════════════════════════════════════════════════════════════

_OP_ARITY: dict[str, int] = {
    # Cross-sectional (arity 1)
    "rank": 1,
    "zscore": 1,
    "sector_rank": 1,
    # Time-series (arity 2)
    "ts_delta": 2,
    "ts_pct": 2,
    "ts_mean": 2,
    "ts_std": 2,
    "ts_rank": 2,
    "ts_min": 2,
    "ts_max": 2,
    # Time-series correlation (arity 3)
    "ts_corr": 3,
    # Arithmetic (arity 2)
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    # Unary arithmetic (arity 1)
    "log": 1,
}

# How many int-parameter tokens each operator pushes onto the stack
# (on top of its child subtrees).  Arity = num_subtrees + num_int_params.
_OP_INT_PARAMS: dict[str, int] = {
    "rank": 0,
    "zscore": 0,
    "sector_rank": 0,
    "log": 0,
    "add": 0,
    "sub": 0,
    "mul": 0,
    "div": 0,
    "ts_delta": 1,
    "ts_pct": 1,
    "ts_mean": 1,
    "ts_std": 1,
    "ts_rank": 1,
    "ts_min": 1,
    "ts_max": 1,
    "ts_corr": 1,
}

# Number of subtrees each operator expects (arity - int_params).
_OP_NUM_SUBTREES: dict[str, int] = {
    op: _OP_ARITY[op] - _OP_INT_PARAMS[op] for op in _OP_ARITY
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Helper — resolve args from stack
# ═══════════════════════════════════════════════════════════════════════════

def _resolve(stack_item, df: pd.DataFrame):
    """Resolve a stack item to a numpy array or scalar.

    If it's a string, it's a column name — grab the values from df.
    If it's a scalar number (int, float), return it as-is for use as a parameter.
    Otherwise it's already an array (from a previous operator result).
    """
    if isinstance(stack_item, str):
        if stack_item not in df.columns:
            raise KeyError(
                f"Column '{stack_item}' not found in dataframe "
                f"(available: {list(df.columns)})"
            )
        return df[stack_item].values.astype(float)
    # Keep scalars as scalars — needed for int parameters like d in ts_ops
    if isinstance(stack_item, (int, float, np.integer, np.floating)):
        return stack_item
    return np.asarray(stack_item, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Cross-Sectional Operators (arity 1)
# ═══════════════════════════════════════════════════════════════════════════

def op_rank(df: pd.DataFrame, x) -> np.ndarray:
    """Cross-sectional percentile rank within each trade_date.

    Returns values in [0, 1].  For each row, percentile rank = (count_strictly_less + 1) / N.
    1/N is the minimum rank, (N)/N = 1 is the maximum.
    """
    s = pd.Series(x, index=df.index)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("trade_date").groups.items():
        vals = s.loc[idx].values
        nan_mask = np.isnan(vals)
        if nan_mask.all():
            continue
        # Build rank for each non-NaN position
        ranks = np.empty(len(vals))
        ranks[:] = np.nan
        valid = vals[~nan_mask]
        # Use broadcasting for ranks
        ranks[~nan_mask] = (
            np.sum(valid[:, np.newaxis] > valid[np.newaxis, :], axis=1) + 1
        ) / len(valid)
        result.loc[idx] = ranks
    return result.values


def op_zscore(df: pd.DataFrame, x) -> np.ndarray:
    """Cross-sectional z-score within each trade_date."""
    s = pd.Series(x, index=df.index)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("trade_date").groups.items():
        vals = s.loc[idx].values
        mu = np.nanmean(vals)
        sigma = np.nanstd(vals)
        if sigma is None or np.isnan(sigma) or sigma < 1e-12:
            result.loc[idx] = 0.0
        else:
            result.loc[idx] = (vals - mu) / sigma
    return result.values


def op_sector_rank(df: pd.DataFrame, x) -> np.ndarray:
    """Cross-sectional rank within (trade_date, industry_sw2) groups.

    Requires an 'industry_sw2' column in df.
    """
    if "industry_sw2" not in df.columns:
        raise KeyError("op_sector_rank requires 'industry_sw2' column in df")
    s = pd.Series(x, index=df.index)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby(["trade_date", "industry_sw2"]).groups.items():
        vals = s.loc[idx].values
        nan_mask = np.isnan(vals)
        if nan_mask.all():
            continue
        ranks = np.empty(len(vals))
        ranks[:] = np.nan
        valid = vals[~nan_mask]
        ranks[~nan_mask] = (
            np.sum(valid[:, np.newaxis] > valid[np.newaxis, :], axis=1) + 1
        ) / len(valid)
        result.loc[idx] = ranks
    return result.values


# ═══════════════════════════════════════════════════════════════════════════
# 4. Time-Series Operators (arity 2 or 3)
#
# CRITICAL: All time-series operators use shift(d) with d > 0.
# shift(d) shifts data FORWARD in time — today sees past values.
# shift(-d) is NEVER used.  Per iron rule #9.
# ═══════════════════════════════════════════════════════════════════════════

def _require_code_col(df: pd.DataFrame) -> None:
    if "code" not in df.columns:
        raise KeyError("Time-series operators require a 'code' column in df")


def op_ts_delta(df: pd.DataFrame, x, d: int) -> np.ndarray:
    """x - shift(x, d).  Today sees x from d days ago."""
    d = int(d)
    _require_code_col(df)
    s = pd.Series(x, index=df.index)
    return s.groupby(df["code"]).transform(
        lambda g: g - g.shift(d)
    ).values


def op_ts_pct(df: pd.DataFrame, x, d: int) -> np.ndarray:
    """x / shift(x, d) - 1.  Pct change over d days."""
    d = int(d)
    _require_code_col(df)
    s = pd.Series(x, index=df.index)
    return s.groupby(df["code"]).transform(
        lambda g: g / g.shift(d) - 1
    ).values


def op_ts_mean(df: pd.DataFrame, x, d: int) -> np.ndarray:
    """Rolling mean of shift(1) over d windows.

    Today sees the mean of yesterday and d-1 previous days.
    """
    d = int(d)
    _require_code_col(df)
    s = pd.Series(x, index=df.index)
    min_p = max(d // 2, 3)
    return s.groupby(df["code"]).transform(
        lambda g: g.shift(1).rolling(d, min_periods=min_p).mean()
    ).values


def op_ts_std(df: pd.DataFrame, x, d: int) -> np.ndarray:
    """Rolling std of shift(1) over d windows."""
    d = int(d)
    _require_code_col(df)
    s = pd.Series(x, index=df.index)
    min_p = max(d // 2, 3)
    return s.groupby(df["code"]).transform(
        lambda g: g.shift(1).rolling(d, min_periods=min_p).std()
    ).values


def op_ts_rank(df: pd.DataFrame, x, d: int) -> np.ndarray:
    """Rolling percentile rank of shift(1) over d windows within each code.

    At each point i, computes the percentile rank of the most recent (shifted)
    value within the d-day window ending at i.
    """
    d = int(d)
    _require_code_col(df)
    s = pd.Series(x, index=df.index)
    min_p = max(d // 2, 3)

    def _rolling_rank(g: pd.Series) -> pd.Series:
        shifted = g.shift(1)
        result = pd.Series(np.nan, index=g.index, dtype=float)
        vals = shifted.values
        for i in range(d - 1, len(vals)):
            window = vals[i - d + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= min_p:
                result.iloc[i] = (np.sum(valid < valid[-1]) + 1) / len(valid)
        return result

    return s.groupby(df["code"]).transform(_rolling_rank).values


def op_ts_min(df: pd.DataFrame, x, d: int) -> np.ndarray:
    """Rolling min of shift(1) over d windows within each code."""
    d = int(d)
    _require_code_col(df)
    s = pd.Series(x, index=df.index)
    min_p = max(d // 2, 3)
    return s.groupby(df["code"]).transform(
        lambda g: g.shift(1).rolling(d, min_periods=min_p).min()
    ).values


def op_ts_max(df: pd.DataFrame, x, d: int) -> np.ndarray:
    """Rolling max of shift(1) over d windows within each code."""
    d = int(d)
    _require_code_col(df)
    s = pd.Series(x, index=df.index)
    min_p = max(d // 2, 3)
    return s.groupby(df["code"]).transform(
        lambda g: g.shift(1).rolling(d, min_periods=min_p).max()
    ).values


def op_ts_corr(df: pd.DataFrame, x, y, d: int) -> np.ndarray:
    """Rolling correlation of shift(1) of x and shift(1) of y over d windows."""
    d = int(d)
    _require_code_col(df)
    sx = pd.Series(x, index=df.index)
    sy = pd.Series(y, index=df.index)
    min_p = max(d // 2, 3)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for code, idx in df.groupby("code").groups.items():
        gx = sx.loc[idx].shift(1)
        gy = sy.loc[idx].shift(1)
        for i in range(d - 1, len(gx)):
            xw = gx.iloc[i - d + 1 : i + 1]
            yw = gy.iloc[i - d + 1 : i + 1]
            mask = xw.notna() & yw.notna()
            if mask.sum() >= min_p:
                result.loc[idx[i]] = xw[mask].corr(yw[mask])
    return result.values


# ═══════════════════════════════════════════════════════════════════════════
# 5. Arithmetic Operators
# ═══════════════════════════════════════════════════════════════════════════

def op_add(df: pd.DataFrame, x, y) -> np.ndarray:
    """Element-wise addition: x + y."""
    return np.asarray(x, dtype=float) + np.asarray(y, dtype=float)


def op_sub(df: pd.DataFrame, x, y) -> np.ndarray:
    """Element-wise subtraction: x - y."""
    return np.asarray(x, dtype=float) - np.asarray(y, dtype=float)


def op_mul(df: pd.DataFrame, x, y) -> np.ndarray:
    """Element-wise multiplication: x * y."""
    return np.asarray(x, dtype=float) * np.asarray(y, dtype=float)


def op_div(df: pd.DataFrame, x, y) -> np.ndarray:
    """Safe division: x / y where |y| > 1e-12, else NaN."""
    xn = np.asarray(x, dtype=float)
    yn = np.asarray(y, dtype=float)
    mask = np.abs(yn) > 1e-12
    result = np.full_like(xn, np.nan, dtype=float)
    result[mask] = xn[mask] / yn[mask]
    return result


def op_log(df: pd.DataFrame, x) -> np.ndarray:
    """Safe natural log: log(|x|) where |x| > 1e-12, else NaN."""
    xn = np.asarray(x, dtype=float)
    mask = np.abs(xn) > 1e-12
    result = np.full_like(xn, np.nan, dtype=float)
    result[mask] = np.log(np.abs(xn[mask]))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. Operator Registry
# ═══════════════════════════════════════════════════════════════════════════

_OPERATORS: dict = {
    "rank": op_rank,
    "zscore": op_zscore,
    "sector_rank": op_sector_rank,
    "ts_delta": op_ts_delta,
    "ts_pct": op_ts_pct,
    "ts_mean": op_ts_mean,
    "ts_std": op_ts_std,
    "ts_rank": op_ts_rank,
    "ts_min": op_ts_min,
    "ts_max": op_ts_max,
    "ts_corr": op_ts_corr,
    "add": op_add,
    "sub": op_sub,
    "mul": op_mul,
    "div": op_div,
    "log": op_log,
}


# ═══════════════════════════════════════════════════════════════════════════
# 7. RPN Parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_rpn(expr: str) -> list[str]:
    """Parse a semicolon-delimited RPN expression string into tokens.

    Args:
        expr: e.g. ';close;20;ts_rank'

    Returns:
        List of string tokens, e.g. ['close', '20', 'ts_rank']

    Empty tokens from leading/trailing/double semicolons are skipped.
    """
    if not expr:
        return []
    tokens = expr.split(";")
    return [t for t in tokens if t != ""]


def tokens_to_str(tokens: list[str]) -> str:
    """Convert a token list back to a semicolon-delimited RPN string.

    Args:
        tokens: e.g. ['close', '20', 'ts_rank']

    Returns:
        ';close;20;ts_rank'
    """
    if not tokens:
        return ""
    return ";" + ";".join(str(t) for t in tokens)


# ═══════════════════════════════════════════════════════════════════════════
# 8. RPN Evaluator (Stack Machine)
# ═══════════════════════════════════════════════════════════════════════════

def _is_number_token(token: str) -> bool:
    """Check if a token represents a numeric literal (int or float)."""
    try:
        float(token)
        return True
    except (ValueError, TypeError):
        return False


def evaluate_rpn(df: pd.DataFrame, tokens: list[str]) -> pd.Series:
    """Evaluate an RPN expression against a DataFrame.

    Stack-machine evaluation:
    - Numeric tokens are pushed as int or float.
    - Tokens matching df.columns are pushed as column name strings (lazy resolution).
    - Operator tokens pop their arity args, resolve column names to arrays,
      call the operator, and push the numeric result.

    Args:
        df: DataFrame with code, trade_date columns plus data columns.
        tokens: List of token strings in RPN order.

    Returns:
        pd.Series with the final evaluation result, aligned to df.index.
        Named '_expr'.

    Raises:
        KeyError: if a referenced column is not in df.
        ValueError: if the expression is malformed (stack under/overflow).
    """
    stack: list = []

    for token in tokens:
        if token == "":
            continue

        if _is_number_token(token):
            num = float(token)
            stack.append(int(num) if num == int(num) else num)
        elif token in _OP_ARITY:
            arity = _OP_ARITY[token]
            if len(stack) < arity:
                raise ValueError(
                    f"Stack underflow: operator '{token}' needs {arity} args "
                    f"but stack has {len(stack)} items"
                )
            args = []
            for _ in range(arity):
                args.append(stack.pop())
            args.reverse()  # Restore data-flow order
            resolved = [_resolve(a, df) for a in args]
            fn = _OPERATORS[token]
            result = fn(df, *resolved)
            stack.append(result)
        elif token in df.columns:
            stack.append(token)  # Lazy: store column name string
        else:
            raise KeyError(
                f"Unknown token '{token}': not an operator, not in df.columns "
                f"(available: {sorted(df.columns.tolist())})"
            )

    if len(stack) == 0:
        raise ValueError("Empty expression: stack is empty after evaluation")
    if len(stack) > 1:
        raise ValueError(
            f"Stack has {len(stack)} items after evaluation, expected 1. "
            f"Expression may be incomplete."
        )

    final = _resolve(stack[0], df)
    return pd.Series(final, index=df.index, name="_expr")


# ═══════════════════════════════════════════════════════════════════════════
# 9. Random Tree Generation
# ═══════════════════════════════════════════════════════════════════════════

def _random_int_param(lo: int = 3, hi: int = 120) -> int:
    """Generate a random integer parameter for time-series window operators.

    Favors round numbers: 5, 10, 15, 20, 30, 40, 60, 90, 120.
    """
    preferred = [5, 10, 15, 20, 30, 40, 60, 90, 120]
    pref = [p for p in preferred if lo <= p <= hi]
    if pref and random.random() < 0.6:
        return random.choice(pref)
    return random.randint(lo, hi)


def random_tree(
    leaf_pool: list[str],
    max_depth: int = 4,
    op_probs: Optional[dict] = None,
) -> list[str]:
    """Generate a random valid RPN expression tree.

    Args:
        leaf_pool: List of column names available as leaf nodes.
        max_depth: Maximum tree depth (root = depth 1).
        op_probs: Optional dict mapping operator name to selection probability.
                  If None, uses weighted defaults favouring simpler trees.

    Returns:
        List of string tokens in RPN order.

    Raises:
        ValueError: if leaf_pool is empty.
    """
    if not leaf_pool:
        raise ValueError("leaf_pool must be non-empty")

    # Default operator probabilities — favour arith and simple CS ops
    if op_probs is None:
        op_probs = {
            "rank": 0.12,
            "zscore": 0.08,
            "sector_rank": 0.05,
            "ts_delta": 0.06,
            "ts_pct": 0.06,
            "ts_mean": 0.10,
            "ts_std": 0.06,
            "ts_rank": 0.08,
            "ts_min": 0.04,
            "ts_max": 0.04,
            "ts_corr": 0.04,
            "add": 0.07,
            "sub": 0.05,
            "mul": 0.05,
            "div": 0.04,
            "log": 0.06,
        }
    # Normalise
    total = sum(op_probs.values())
    op_probs_norm = {k: v / total for k, v in op_probs.items()}

    op_names = list(op_probs_norm.keys())
    op_weights = list(op_probs_norm.values())

    def _gen(depth: int) -> list[str]:
        """Recursively generate RPN tokens.

        Generates tokens in post-order: subtrees first, then int params,
        then operator.
        """
        # At depth >= max_depth, force a leaf
        if depth >= max_depth or depth >= 5:  # absolute cap
            return [random.choice(leaf_pool)]

        # Leaf vs operator decision
        leaf_prob = 0.4 + 0.15 * (depth - 1)  # grows with depth
        if random.random() < leaf_prob:
            return [random.choice(leaf_pool)]

        # Pick an operator
        op = random.choices(op_names, weights=op_weights, k=1)[0]
        arity = _OP_ARITY[op]
        num_subtrees = _OP_NUM_SUBTREES[op]
        num_params = _OP_INT_PARAMS[op]

        tokens: list[str] = []

        # Generate child subtrees
        for _ in range(num_subtrees):
            tokens.extend(_gen(depth + 1))

        # Push integer parameters
        for _ in range(num_params):
            tokens.append(str(_random_int_param()))

        # Push the operator
        tokens.append(op)

        return tokens

    # Repeated attempts in case we generate something implausible
    for _ in range(10):
        tokens = _gen(1)
        # Sanity check: ensure at least one leaf is from leaf_pool
        if any(t in leaf_pool for t in tokens):
            return tokens
    # Last resort fallback
    return _gen(1)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Genetic Operators
# ═══════════════════════════════════════════════════════════════════════════

# Semantic token categories for targeted mutation
_OPERATOR_NAMES = frozenset(_OP_ARITY.keys())


def _classify_token(token: str) -> str:
    """Classify a token as 'leaf', 'operator', or 'param'."""
    if token in _OPERATOR_NAMES:
        return "operator"
    if _is_number_token(token):
        return "param"
    return "leaf"


def mutate_rpn(
    tokens: list[str],
    leaf_pool: list[str],
    mutation_rate: float = 0.3,
) -> list[str]:
    """Randomly mutate an RPN expression.

    Mutation operators applied per-position with mutation_rate:
    - Leaf tokens: replaced with a random leaf from leaf_pool.
    - Parameter tokens: jittered by +/- 1..5 (or replaced with a round number).
    - Operator tokens: replaced with another operator of the same arity.

    Additionally, with (mutation_rate / 2) probability, a random subtree is
    replaced with a fresh random subtree.

    Args:
        tokens: Original token list in RPN order.
        leaf_pool: Available leaf column names.
        mutation_rate: Probability per token of being mutated.

    Returns:
        New token list (original is never mutated in place).
    """
    result = list(tokens)  # shallow copy of strings (immutable, but safe)

    if not result:
        return result

    # Per-position mutations
    for i, tok in enumerate(result):
        if random.random() >= mutation_rate:
            continue

        cls = _classify_token(tok)
        if cls == "leaf":
            result[i] = random.choice(leaf_pool)
        elif cls == "param":
            cur = int(float(tok))
            delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 10])
            new_val = max(2, min(252, cur + delta))
            result[i] = str(new_val)
        elif cls == "operator":
            # Replace with an operator of the same arity
            arity = _OP_ARITY[tok]
            same_arity = [op for op, a in _OP_ARITY.items() if a == arity and op != tok]
            if same_arity:
                result[i] = random.choice(same_arity)

    # Subtree-replacement mutation (coarser grain)
    if random.random() < mutation_rate / 2:
        result = _mutate_subtree(result, leaf_pool)

    return result


def _mutate_subtree(tokens: list[str], leaf_pool: list[str]) -> list[str]:
    """Replace a random subtree with a fresh randomly-generated subtree.

    Works by selecting a random operator position, determining the span
    of its subtree (in RPN order), and replacing that span.

    If no operator is found (degenerate tree), no mutation is applied.
    """
    # Find all operator positions with their subtree start positions
    op_positions = [i for i, t in enumerate(tokens) if t in _OPERATOR_NAMES]
    if not op_positions:
        return list(tokens)

    target_op_pos = random.choice(op_positions)
    span = _subtree_span(tokens, target_op_pos)
    if span is None:
        return list(tokens)

    start, end = span  # start inclusive, end exclusive

    # Generate a replacement subtree
    new_subtree = random_tree(leaf_pool, max_depth=3)

    return tokens[:start] + new_subtree + tokens[end:]


def _subtree_span(tokens: list[str], op_pos: int) -> Optional[tuple[int, int]]:
    """Given an operator at op_pos, find the [start, end) span of its
    entire subtree in RPN order.

    RPN property: an operator consumes its arity items from the stack.
    Walking backwards from the operator, we find the contiguous span
    that produces exactly arity items on the stack.

    Returns (start, end) where start is the index of the first token
    of the subtree, end is op_pos + 1 (exclusive).
    """
    arity = _OP_ARITY.get(tokens[op_pos])
    if arity is None:
        return None

    # Walk backwards tracking how many values we need on the stack
    needed = arity
    pos = op_pos - 1
    while needed > 0 and pos >= 0:
        tok = tokens[pos]
        if tok in _OPERATOR_NAMES:
            # This operator produces 1 value but consumes its own arity
            op_arity = _OP_ARITY[tok]
            needed += op_arity  # need op_arity more values (but we consume -1)
            needed -= 1         # this op produces 1 value
        else:
            # Leaf or param — produces 1 value
            needed -= 1
        pos -= 1

    if needed > 0:
        return None  # malformed expression

    return (pos + 1, op_pos + 1)


def crossover_rpn(
    tokens_a: list[str],
    tokens_b: list[str],
) -> Optional[list[str]]:
    """Single-point subtree crossover between two RPN trees.

    Selects a random operator in tokens_a, finds its subtree span,
    selects a random operator in tokens_b, finds its subtree span,
    and swaps them.

    Args:
        tokens_a: Parent A token list.
        tokens_b: Parent B token list.

    Returns:
        New token list (child), or None if crossover is not possible
        (either tree has no operators, or spans are too large).
    """
    # Find operator positions in both trees
    ops_a = [i for i, t in enumerate(tokens_a) if t in _OPERATOR_NAMES]
    ops_b = [i for i, t in enumerate(tokens_b) if t in _OPERATOR_NAMES]

    if not ops_a or not ops_b:
        return None

    op_a_pos = random.choice(ops_a)
    op_b_pos = random.choice(ops_b)

    span_a = _subtree_span(tokens_a, op_a_pos)
    span_b = _subtree_span(tokens_b, op_b_pos)

    if span_a is None or span_b is None:
        return None

    start_a, end_a = span_a
    start_b, end_b = span_b

    subtree_b = tokens_b[start_b:end_b]

    # Replace subtree_A with subtree_B
    child = tokens_a[:start_a] + subtree_b + tokens_a[end_a:]

    return child


# ═══════════════════════════════════════════════════════════════════════════
# 11. Tree Validation Utilities
# ═══════════════════════════════════════════════════════════════════════════

def validate_rpn(tokens: list[str]) -> bool:
    """Check whether an RPN token list is syntactically valid.

    A valid RPN expression must:
    - Not be empty.
    - Evaluate to exactly one value on the stack.
    - Have every operator consume the correct number of arguments.
    """
    if not tokens:
        return False

    stack_size = 0
    for tok in tokens:
        if tok in _OPERATOR_NAMES:
            arity = _OP_ARITY[tok]
            if stack_size < arity:
                return False
            stack_size -= arity
            stack_size += 1  # operator produces 1 result
        else:
            stack_size += 1

    return stack_size == 1


def tree_depth(tokens: list[str]) -> int:
    """Compute the maximum nesting depth of an RPN expression tree.

    Returns 0 for a single leaf, 1 for a leaf with one operator, etc.
    """
    if not tokens:
        return 0

    max_depth = 0
    stack: list[int] = []  # stack of depths

    for tok in tokens:
        if tok in _OPERATOR_NAMES:
            arity = _OP_ARITY[tok]
            if len(stack) < arity:
                return -1  # malformed
            child_depths = [stack.pop() for _ in range(arity)]
            node_depth = max(child_depths) + 1
            max_depth = max(max_depth, node_depth)
            stack.append(node_depth)
        else:
            stack.append(0)

    return max_depth if len(stack) == 1 else -1


def tree_size(tokens: list[str]) -> int:
    """Count the total number of nodes (leaves + operators) in the tree.

    Note: this count does NOT include numeric parameter tokens.
    Leaf and operator tokens are counted; parameter tokens are excluded
    from the node count.
    """
    return sum(1 for t in tokens if t in _OPERATOR_NAMES or not _is_number_token(t))
