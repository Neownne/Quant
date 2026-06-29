# Task 2 Report: Expression Tree Engine

## What Was Implemented

### `factors/expression_tree.py` (356 lines)

**RPN Parser:**
- `parse_rpn(expr)` — semicolon-delimited string to token list
- `tokens_to_str(tokens)` — token list back to semicolon-delimited string
- Handles edge cases: empty input, double semicolons, missing leading semicolon

**16 Operators (the spec listed 15 in the title but defined 16 in the arity map):**

| Category | Operators | Count |
|----------|-----------|-------|
| Cross-sectional (arity 1) | `rank`, `zscore`, `sector_rank` | 3 |
| Time-series (arity 2) | `ts_delta`, `ts_pct`, `ts_mean`, `ts_std`, `ts_rank`, `ts_min`, `ts_max` | 7 |
| Time-series correlation (arity 3) | `ts_corr` | 1 |
| Arithmetic (arity 2) | `add`, `sub`, `mul`, `div` | 4 |
| Unary arithmetic (arity 1) | `log` | 1 |

All operators follow the unified signature `op(df, *args)`.

**RPN Evaluator:**
- Stack-machine evaluation with lazy column resolution
- Column names stored as strings on the stack, resolved to arrays only when consumed by operators
- Numeric parameters preserved as Python scalars for type safety in operator functions

**Random Tree Generation:**
- `random_tree(leaf_pool, max_depth, op_probs)` — weighted operator selection, leaf-favouring at deeper levels
- Integer parameters favour round numbers (5, 10, 15, 20, 30, 40, 60, 90, 120)

**Genetic Operators:**
- `mutate_rpn(tokens, leaf_pool, mutation_rate)` — per-token mutation (leaf/param/operator replacement) plus subtree replacement
- `crossover_rpn(tokens_a, tokens_b)` — single-point subtree swap via RPN span detection

**Validation Utilities:**
- `validate_rpn(tokens)` — syntactic validity check
- `tree_depth(tokens)` — maximum nesting depth
- `tree_size(tokens)` — node count (excluding parameter tokens)

**No Future Data:**
- All time-series operators use `shift(1)` for the first lag, then `shift(d)` with d>0
- Verified by grep: zero occurrences of `shift(-` in operator function bodies

### `tests/test_expression_tree.py` (66 tests in 20 test classes)

Coverage:
- 7 parser tests (parse, roundtrip, edge cases)
- 12 operator unit tests (rank, zscore, sector_rank, ts_mean, ts_delta, ts_rank, ts_corr, add, sub, mul, div, log)
- 10 RPN evaluator tests (simple, complex, nested, error cases)
- 5 random tree tests (validity, depth, leaf pool, custom op_probs)
- 3 mutation tests (different, preserves validity, empty)
- 2 crossover tests (valid, single-leaf)
- 3 no-future-shift tests (day-zero NaN, code inspection, past-only)
- 9 validation/utility tests (validate_rpn, tree_depth, tree_size)
- 2 registry tests (count, consistency)
- 3 end-to-end tests (parse-evaluate roundtrip, generation pipeline)

## Test Results

```
============================== 66 passed in 1.02s ==============================
```

All 66 tests pass with zero failures.

## Concerns

1. **Operator count discrepancy**: The spec title says "15 Operators" but the arity map defines 16 entries. Implemented all 16 as listed in the spec's `_OP_ARITY` dict.
2. **Random tree window params**: `random_tree` can generate window parameters up to 120, which may exceed available data length. This is inherent to the random search process and handled gracefully (operators return NaN for insufficient data).
3. **`op_sector_rank` depends on `industry_sw2` column** — this column must be present in the DataFrame for evaluation. If the column is missing, a clear `KeyError` is raised.
