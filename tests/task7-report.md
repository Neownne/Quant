# Task 7: Deep Analysis Report Generator — Completion Report

## Status: PASS (34/34 tests)

## Files Created

### `factors/analyst.py` (17.6 KB)

Seven-dimension deep analysis report generator integrated with the Genetic Programming factor evolution pipeline.

**Components:**

1. **Data Source Classification** — `LEAF_DATA_SOURCE` dict maps 30+ leaf tokens to 7 categories (price_volume, valuation, financial, macro, intraday, prebuilt_factor, other). `_classify_leaf()` dispatches on `@`-prefix for prebuilt factors, falling back to the lookup table.

2. **Analysis Helpers:**
   - `_leaf_usage()` — Counts non-operator token frequency across population
   - `_operator_usage()` — Counts operator frequency; leverages `_OP_ARITY` from `expression_tree.py` for operator identification
   - `_tree_depth()` — Simulates RPN stack execution to compute expression tree depth; each operator pop N children and push max(depths)+1; leaves push 0
   - `_factor_correlation()` — Pairwise Spearman correlation on top-20 factors (by abs IC); pairs with |r| > 0.7 flagged as redundant

3. **Main Report Generator** (`generate_analysis_report`) — 7 dimensions:
   - `data_source_coverage` — Source breakdown (%), unused known leaves, top-20 leaves by count
   - `factor_structure` — Depth histogram, operator usage table, avg depth/nodes, depth-vs-IC scatter
   - `ic_decay` — Per-round avg_abs_IC trend (last 50 rounds), current round avg IC
   - `regime_sensitivity` — Industry-spearman IC (if industry_sw1 col present), MCap-quintile IC (if mcap col present); notes field when columns absent
   - `factor_redundancy` — Delegates to `_factor_correlation`
   - `ml_feature_importance` — LightGBM importances sorted descending, NDCG@5 score
   - `backtest_diagnostics` — Annual return, max DD, Sharpe, win rate, n_trades, fitness score; reads `bt_*` prefixed keys with fallback

4. **Suggestion System:**
   - `load_suggestions(path)` — Returns None for missing/invalid files
   - `apply_suggestions(suggestions, leaf_pool, operator_probs)` — 4 operations:
     - `force_data_source`: 3x duplicate matching leaves (4x probability)
     - `boost_leaf_prob`: Nx duplicate specific leaves
     - `boost_operator`: 2x operator probability
     - `kill_operator`: remove + renormalize
   - `save_analysis_report(report, round_num)` — Writes to `data/analysis_round_NNNN.json`

**Edge Cases Handled:**
- Empty population: all-zero depth/nodes, empty source coverage
- Missing `industry_sw1`: "note" field in regime_sensitivity
- Missing `mcap`: "note" field in regime_sensitivity
- None ml_result: all defaults (0.0 scores, empty importances)
- Missing `series` in results: skipped in correlation
- Fewer than 10 common indices: correlation pair skipped
- Malformed RPN tokens: fallback to depth 0

### `tests/test_analyst.py` (14.5 KB)

34 tests across 8 test classes:

| Class | Tests | Coverage |
|-------|-------|----------|
| TestClassifyLeaf | 3 | known/unknown/prebuilt classification |
| TestTreeDepth | 8 | simple, nested, ts_ops, empty, leaf-only |
| TestLeafUsage | 2 | counts + empty population |
| TestOperatorUsage | 2 | counts + empty population |
| TestFactorCorrelation | 3 | redundancy detection, single factor, no series |
| TestGenerateReport | 6 | all dimensions, edge cases (empty, None ML, no industry) |
| TestSuggestions | 7 | missing/invalid load, force/boost/kill/combined |
| TestSaveReport | 1 | roundtrip save->reload |
| TestReusedFunctions | 2 | depth_vs_ic structure, redundancy pair format |
