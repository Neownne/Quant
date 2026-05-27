#!/bin/bash
# Run all 6 strategies with 2020-2026 window (6-year)
# 1 static (震荡网格) + 5 ML (含动态反馈闭环)
set -e
cd "$(dirname "$0")/.."

START="20200101"
END="20260501"
N=30
REBAL=5  # 周度调仓（5个交易日）
UNIVERSE=200  # 候选池大小（0=全市场）

echo "=========================================="
echo " Phase 1: Static Strategy"
echo "=========================================="

python scripts/run_static_backtest.py --strategy "震荡网格(高抛低吸)" --top-n $N --start $START --end $END

echo ""
echo "=========================================="
echo " Phase 2: ML Strategies (5)"
echo "=========================================="

# 1. ML-默认集成: momentum+reversal ensemble, ret_1d
echo ">>> ML-默认集成: momentum+reversal ensemble"
python scripts/run_ml_backtest.py \
    --strategy "ML-默认集成" \
    --factor-preset "+momentum+reversal" \
    --forward-days 1 \
    --top-n $N \
    --rebalance-freq $REBAL \
    --universe-size $UNIVERSE \
    --start $START --end $END

# 2. ML-动量精选: momentum only, XGBoost, ret_5d
echo ">>> ML-动量精选: momentum, XGBoost, ret_5d"
python scripts/run_ml_backtest.py \
    --strategy "ML-动量精选" \
    --factor-preset momentum \
    --forward-days 5 \
    --model xgboost \
    --top-n $N \
    --rebalance-freq $REBAL \
    --universe-size $UNIVERSE \
    --start $START --end $END

# 3. ML-反转精选: reversal only, LightGBM, ret_1d
echo ">>> ML-反转精选: reversal, LightGBM, ret_1d"
python scripts/run_ml_backtest.py \
    --strategy "ML-反转精选" \
    --factor-preset reversal \
    --forward-days 1 \
    --model lightgbm \
    --top-n $N \
    --rebalance-freq $REBAL \
    --universe-size $UNIVERSE \
    --start $START --end $END

# 4. ML-全量因子测试: all factors, ensemble
echo ">>> ML-全量因子测试: all factors, ensemble"
python scripts/run_ml_backtest.py \
    --strategy "ML-全量因子测试" \
    --factor-preset all \
    --forward-days 1 \
    --top-n $N \
    --rebalance-freq $REBAL \
    --universe-size $UNIVERSE \
    --start $START --end $END

# 5. ML-动态多因子: all factors + 归因->健康度->调参 闭环
echo ">>> ML-动态多因子: all factors + dynamic feedback loop"
python scripts/run_ml_backtest.py \
    --strategy "ML-动态多因子" \
    --factor-preset all \
    --forward-days 1 \
    --dynamic \
    --top-n $N \
    --rebalance-freq $REBAL \
    --universe-size $UNIVERSE \
    --start $START --end $END

echo ""
echo "=========================================="
echo " All 6 strategies complete"
echo "=========================================="
