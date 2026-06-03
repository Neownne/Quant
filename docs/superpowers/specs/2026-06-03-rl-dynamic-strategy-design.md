# RL-Dynamic 新策略 — 设计文档

## Context

舞策略的 Regime 自适应是 5 个固定状态 → 5 套固定参数，但市场远比 5 种复杂。建议文档指出头部机构已转向 RL 驱动的动态因子权重。构建完全独立的新策略，与舞策略并行运行对比。

## Goals

- RL（PPO）学习动态因子权重，替代人工 Regime 分类
- 完全独立于舞策略（独立目录/账户/模型/选股）
- 共享底层数据（stock_daily qfq）
- 交易参数与舞策略一致（TradingConfig）

## Architecture

```
StateBuilder（市场特征向量）
  ├── 市场宽度: 涨跌比/涨停数/跌停数/资金流
  ├── 板块动量: 5板块的20日收益
  ├── 波动率:  指数20日波动
  ├── 恐贪指标: 综合情绪分数
  └── 因子IC:   各因子近20日RankIC
       │
       ▼
RLPolicy（PPO, MPS GPU）
  输入: ~50维状态向量
  输出: N个因子的权重(softmax归一化)
       │
       ▼
Scorer
  因子值 × RL权重 → 综合得分 → NDrop选股
       │
       ▼
Reward = 组合日收益或 Sharpe
```

## 组件

### rl_dynamic/state_builder.py
- `build_market_state(ohlcv, index_df, sector_map, date)` → np.array(~50维)
- 包含: 市场宽度(8)、板块动量(5)、波动率(3)、恐贪(5)、因子IC(N)

### rl_dynamic/factor_pool.py  
- 维护因子池: 初始全量 86+ 因子
- `compute_factor_matrix(ohlcv, extra_data)` → DataFrame
- 每个 episode 可以选择不同因子子集

### rl_dynamic/policy_net.py
- `FactorWeightNet`: 状态 → 因子权重
- 架构: Linear(50,128) → ReLU → Linear(128,64) → Softmax(N_factors)

### rl_dynamic/env.py
- Gymnasium 环境
- 状态: build_market_state 输出
- 动作: 连续向量(因子权重)
- 奖励: 组合日收益

### rl_dynamic/trainer.py
- Walk-Forward PPO 训练
- 每个窗口训练一个 FactorWeightNet
- MPS GPU 加速

### rl_dynamic/predictor.py
- `.predict(factor_df) → DataFrame[code, score, rank]`
- 与 EnsemblePredictor 相同接口

## 账户配置

- account_id: 18
- run_id: 5
- 交易参数: TradingConfig (佣金0.09‱, 印花税0.5‱, 滑点0.1%)
- 初始资金: 1,000,000

## 回测验证

对比 v1.6 vs RL-Dynamic：
- 相同时间段 (2020-2026)
- 相同候选池 (500只)
- 对比 CAGR/Sharpe/MaxDD

## 文件清单

| 文件 | 说明 |
|------|------|
| rl_dynamic/__init__.py | 模块导出 |
| rl_dynamic/state_builder.py | 市场状态特征 |
| rl_dynamic/factor_pool.py | 因子池管理 |
| rl_dynamic/policy_net.py | RL策略网络 |
| rl_dynamic/env.py | 训练环境 |
| rl_dynamic/trainer.py | Walk-Forward训练 |
| rl_dynamic/predictor.py | 预测接口 |
| rl_dynamic/backtest_runner.py | 回测入口 |
| scripts/run_rl_backtest.py | CLI回测脚本 |
| tests/test_rl_dynamic.py | 测试 |
