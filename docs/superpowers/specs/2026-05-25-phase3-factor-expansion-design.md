# Phase 3: 因子扩展 + 状态识别 + 集成优化 — 设计文档

> 日期：2026-05-25  
> 状态：已确认

---

## 一、目标

在 Phase 2 基线（54.2% 准确率 / 25% 召回率 / 4 窗口 / 34 因子上，逐步提升至年化回测 > 40%。

## 二、新增模块

### 2.1 Alpha191 A 股因子（`factors/alpha191_*.py`）

28 个候选因子分 6 组实现，每组独立文件：

| 文件 | 因子数 | 依赖数据 |
|---|---|---|
| `factors/alpha191_turnover.py` | 5 | turnover, close |
| `factors/alpha191_intraday.py` | 4 | open, high, low, close |
| `factors/alpha191_flow.py` | 6 | high, low, close, volume, turnover |
| `factors/alpha191_vol.py` | 5 | close, amount |
| `factors/alpha191_gap.py` | 4 | open, close |
| `factors/alpha191_liquidity.py` | 4 | high, low, close, volume, amount |

每因子门禁：|RankIC| 均值 > 0.02 + t > 2.0 + 与已有因子 max |corr| < 0.7。

### 2.2 正交性筛选（`factors/screening.py`）

- `compute_factor_correlation_matrix(factors_df)` → 相关性矩阵
- `select_orthogonal_factors(factors_df, threshold=0.7)` → 贪心去冗余
- `compute_marginal_contribution(factor_name, baseline_accuracy)` → 边际贡献检验

### 2.3 市场状态识别（`models/regime.py`）

- 大盘年线位置：price vs MA(250) → 牛/熊市二分类
- 波动率状态：vol_20 分位数 → 高波/低波
- `detect_regime(index_df)` → 返回 {date: regime_label}
- XGBoost/LightGBM 按 regime 分模型训练

### 2.4 多模型集成（更新 `models/trainer.py`）

- `EnsemblePredictor`: 封装 XGBoost + LightGBM
- 概率平均融合：`score = (xgb_prob + lgb_prob) / 2`
- `walk_forward_train_ensemble()` 返回集成结果

### 2.5 阈值+超参调优（`models/tuning.py`）

- Optuna 优化：`n_estimators`, `max_depth`, `learning_rate`, `threshold`
- 目标函数：`maximize(recall × 0.4 + precision × 0.3 + accuracy × 0.3)`
- 最优阈值替代默认 0.5

## 三、实施步骤

| 序号 | 步骤 | 文件 | 验收标准 |
|---|---|---|---|
| 1 | 换手率 5 因子 + 门禁 | `alpha191_turnover.py` | 通过门禁数 ≥ 3 |
| 2 | 日内形态 4 因子 + 门禁 | `alpha191_intraday.py` | 通过门禁数 ≥ 2 |
| 3 | 资金流向 6 因子 + 门禁 | `alpha191_flow.py` | 通过门禁数 ≥ 3 |
| 4 | 波动率高阶 5 因子 | `alpha191_vol.py` | 通过门禁数 ≥ 3 |
| 5 | 隔夜效应 4 因子 | `alpha191_gap.py` | 通过门禁数 ≥ 2 |
| 6 | 流动性高阶 4 因子 | `alpha191_liquidity.py` | 通过门禁数 ≥ 2 |
| 7 | 激活 extra_data | 更新 `run_ml_backtest.py` | log_mcap/pb_pct/sh_change 不再全 NaN |
| 8 | 市场状态识别 | `models/regime.py` | 牛熊分模型训练 > 单模型 |
| 9 | 正交性筛选 | `factors/screening.py` | 去冗余后因子数 ≥ 20 |
| 10 | XGBoost + LightGBM 集成 | 更新 `models/trainer.py` | 集成 > 单模型准确率 |
| 11 | Optuna 阈值+超参调优 | `models/tuning.py` | 召回率 > 40%，准确率 ≥ 52% |

## 四、出口标准

- 全量测试通过
- E2E 准确率 ≥ 56% 且召回率 ≥ 40%
- Walk-forward 窗口 ≥ 4
- 因子总数 ≥ 50（经筛选）
