# RL-Dynamic v2.0 — 完整设计文档

## 策略逻辑

### 交易理念

市场状态不是离散的几个"牛/熊/震荡"标签能描述的。同样的因子在不同时间表现截然不同：
- 动量在牛市有效（趋势延续），在熊市失效（反弹即卖点）
- 反转在恐慌后有效（超跌反弹），在阴跌中失效（基本面恶化）

RL-Dynamic 的核心思想：**让强化学习学会"什么时候用什么因子、给多少权重"**。

### 与舞策略的关系

| | 舞 v1.6 | RL-Dynamic |
|------|------|------|
| 因子 | IC 筛选固定 11 个 | 全量 86+ 个，权重动态 |
| 市场适应 | 5 状态固定参数 | 连续状态空间 |
| 决策逻辑 | "过去 IC 高就用" | "当前状态适合什么就用什么" |

### 状态空间 (~50 维)

```
市场宽度 (8): 涨跌比、涨停数、跌停数、资金占比、收益均值/标准差、
              换手率、活跃度
板块动量 (5): 科创/主板大盘/主板小盘/红利 20日收益
波动率 (3):   指数 20/60 日波动 + 20日收益
恐贪指标 (5): 市场情绪综合分
因子 IC (N):  各因子近 20 日 RankIC
```

### 动作空间

N 维连续向量（softmax 归一化），每个元素 = 对应因子的权重。

### 奖励

`加权因子得分选出的 Top-10 股票的平均日收益`。RL 的目标是最大化选股收益，间接学会最优因子权重。

### 训练方式

Walk-Forward PPO（与舞策略一致的时间窗口）：
- 3 年训练 / 1 年验证，年步进
- 每个窗口独立训练一个 PPO 模型
- MPS GPU 加速（如可用），否则 CPU

### 预测接口

```python
def predict(factor_df, market_state) -> DataFrame[code, score, rank]:
    weights = ppo_model.predict(market_state)  # RL 输出因子权重
    scores = factor_matrix @ weights            # 加权打分
    return sort_by_score(scores)
```

与 `EnsemblePredictor.predict()` 接口兼容。

### 选股与风控

复用舞策略的 NDrop + 等权分配 + 风险控制：
- `select_topk_ndrop(K=15, N=2)`
- 个股止损 -8%，组合回撤 -20% 减仓 / -25% 清仓
- 涨停过滤（无法买入）

### 回测标准

所有回测参数与舞策略一致：
- 初始资金 100 万
- 佣金 0.09‱，印花税 0.5‱（卖出），滑点 0.1%
- 前复权(qfq)数据
- Walk-Forward 时间分割（gap_days=5）
- IC 筛选限首窗口（无前视）
- 宇宙选取限首年成交额（无前视）

### 过拟合防护

1. Walk-Forward 严格时间分割
2. PPO 训练仅用窗口内训练数据
3. 因子 IC 仅用窗口内数据
4. 因子权重由 RL 在线学习，不依赖全局统计

### 模拟盘集成

```
run_daily_paper.py
  ├── 舞 v1.6: walk_forward_train_by_regime + EnsemblePredictor
  ├── 舞 v1.5: walk_forward_train_by_regime + EnsemblePredictor
  └── RL-Dynamic: walk_forward_train_rl_weights + RLDynamicPredictor
```

RL 策略在 `run_strategy` 中走独立分支（`factor_mode == "rl"` 或 `type == "rl"`）。

### 验证清单

- [ ] `python scripts/run_rl_backtest.py` 产出有效 CAGR/Sharpe/MaxDD
- [ ] `python scripts/run_daily_paper.py` 三策略全部正常执行
- [ ] Web `/paper` 页面 RL-Dynamic 面板显示持仓/信号/权益曲线
- [ ] 回测 CAGR 与舞 v1.6 可比（预期 20-35%）
- [ ] 无前视偏差（IC 限首窗口、宇宙限首年、gap_days=5）
