# 策略文档 (v1.10)

## 策略架构

所有策略通过 `strategies/__init__.py` 中的 `get_all_strategies()` 注册，支持静态策略（backtrader）和 ML 策略（XGBoost/LightGBM 集成）。

回测引擎统一配置（`config/settings.py:TradingConfig`）：
- **初始资金**：100 万
- **佣金**：万 0.9（买卖双向）
- **印花税**：万 5（卖出单向）
- **滑点**：0.1%

---

## 当前策略

### 静态策略（1 个）

#### 震荡网格 (高抛低吸)
- **文件**：`grid_shock.py`
- **引擎**：backtrader Cerebro + 等权聚合
- **逻辑**：基于均线偏离的网格交易，高抛低吸
- **参数**：size=500, buy_step=0.02, sell_step=0.02, ma_period=30, max_positions=5

### ML 策略（5 个）

所有 ML 策略共享 v1.10 管线：
```
全市场非ST (5238只) → 因子计算 → IC门禁 → 正交筛选 → Walk-forward训练
→ ML打分 → 排雷过滤(8项,≤3违规) → NDrop调仓(K=15,N=2) → 等权持有
→ 每日止损(-8%) → 组合回撤(-20%减仓,-25%清仓) → 指数大跌空仓
```

| 策略 | 因子池 | 模型 | 标签 | 特点 |
|---|---|---|---|---|
| ML-默认集成 | 动量+反转 | XGB+LGBM 集成 | ret_1d | 双模型集成，更稳健 |
| ML-动量精选 | 动量/趋势(17个) | XGBoost | ret_5d | 周度标签，换手低 |
| ML-反转精选 | 反转(20个) | LightGBM | ret_1d | 日频反转信号 |
| ML-全量因子测试 | 全部 76 因子 | XGB+LGBM 集成 | ret_1d | 最大因子覆盖 |
| ML-动态多因子 | 全部 76 因子 | XGB+LGBM 集成 | ret_1d | 动态反馈闭环 |

运行：
```bash
bash scripts/run_all_backtests.sh    # 一键运行全部 6 个策略
# 或单独运行
python scripts/run_ml_backtest.py --strategy "ML-默认集成" --factor-preset "+momentum+reversal"
python scripts/run_static_backtest.py --strategy "震荡网格(高抛低吸)" --top-n 30
```

---

## 已废弃策略

以下策略已移除（收益趋近于零，不适合 A 股市场）：
- ~~双均线交叉 (SMACross)~~ — `sma_cross.py`
- ~~MACD 金叉死叉 (MACDStrategy)~~ — `macd_strategy.py`
- ~~RSI 超买超卖 (RSIStrategy)~~ — `rsi_strategy.py`

---

## 版本历史

### v1.10 (2026-05-28)
- 全市场选股（修复仅选深市股票偏差）
- 排雷过滤 + NDrop 增量调仓
- 组合级风控（回撤减仓/清仓/指数空仓）
- 全部策略版本统一为 v1.10

### v1.00 (2026-05)
- 初始 ML 管线：因子→IC→正交→训练→Top-N 选股
- RSI/MACD/双均线策略移除
