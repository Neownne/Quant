# Quant — A 股量化交易系统

> 最后更新：2026-06-06 (v2.0 大小票平滑分配)  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)（私有仓库）

---

## 核心工作流

```
因子计算 → 模型训练 → 历史回测(防过拟合) → 模拟盘验证 → 归因反馈闭环
```

---

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│           Web 监控层 (FastAPI + HTMX + Alpine.js)         │
│  行情看板 │ 回测对比 │ 模拟盘 │ 数据状态 │ 因子监控        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                  PostgreSQL (31张表)                      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│              后台研究引擎 (脚本/定时任务)                    │
│  数据同步 → 质量校验 → 因子计算 → 训练 → 回测              │
│                                                          │
│  大票策略 (舞 v1.85): 成交额 Top-500, ML 集成, 日频        │
│  小票策略 (alpha v2.0): 成交额 1000-3000, 反转+低波, 周度  │
└─────────────────────────────────────────────────────────┘
```

---

## 回测结果

> 统一参数：100 万本金、佣金万 0.9、印花税万 5（卖出单向）、小票滑点 0.3%、大票滑点 0.1%。
> 回测区间 2020-01 ~ 2026-05（6年）+ 2015-01 ~ 2026-06（10年跨周期），严格 Walk-Forward 验证。

### 2020-2026（6年）

| 策略 | Sharpe | 年化 | MaxDD | 调仓 |
|------|:---:|:---:|:---:|:---:|
| **大小票平滑分配 v1.0** | **1.50** | **47.7%** | **31.3%** | 周度 |
| 小市值 alpha v2.0 | 0.93 | 25.3% | 25.5% | 周度 |
| 舞 v1.85 | 0.72 | 15.0% | 26.5% | 日频 |

### 2015-2026（10年跨周期）

| 策略 | Sharpe | 年化 | MaxDD |
|------|:---:|:---:|:---:|
| 大小票平滑分配 v1.0 | 0.87 | 23.1% | 36.8% |

### 策略逻辑

**大小票平滑分配 v1.0**（最优）：
- CSI1000 偏离 MA60 程度 → 小票/大票比例平滑调整（10%~90%）
- 极端弱势（跌破 MA60 超 2%）→ 逐步提现，最低保留 50% 仓位
- 小票侧：11 因子 ML 模型选股（2000 只候选池）
- 大票侧：纯动量排名，买跌最多的 Top-200 龙头（弱势期反转效应）

**小市值 alpha v2.0**（独奏）：
- 11 个原创因子（反转+低波+价值+质量+换手异动）
- CSI1000 < MA60 → 减半仓
- 排雷：ST/退市/商誉暴雷/庄股/次新
- 财务数据 45 天公告滞后，截面排名按 trade_date 隔离
- 零前视偏差

---

## 目录结构

```
quant/
├── .env / .env.example / .gitignore
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py           # 集中配置
│
├── data/                     # 数据层
│   ├── db.py                 # DDL + 连接池 + upsert
│   ├── fetcher.py            # 数据获取 (AKShare/腾讯/新浪)
│   ├── sync.py               # 批量同步 (多进程增量)
│   ├── quality.py            # 数据质量校验
│   └── availability.py       # 交易日历
│
├── factors/                  # 因子层 (83 个因子)
│   ├── engine.py             # FactorEngine + 截面/行业中性化
│   ├── alpha101.py           # 30 个 Alpha101 因子
│   ├── alpha191_*.py         # Alpha191 因子 (6 类 28 个)
│   ├── custom.py             # 7 个自定义因子
│   ├── fundamental.py        # 11 个基本面因子
│   ├── intraday_minute.py    # 7 个日内因子
│   ├── monitor.py            # IC/ICIR 监控
│   └── screening.py          # 正交筛选 (IC 门禁 + 贪心)
│
├── models/                   # ML 预测层
│   ├── dataset.py            # 数据集构造 + walk-forward 切分
│   ├── trainer.py            # XGBoost/LightGBM + EnsemblePredictor
│   ├── regime.py             # 5 状态市场检测
│   └── tuning.py             # 阈值搜索 + Optuna 调优
│
├── portfolio/                # 组合优化层
│   ├── selector.py           # 选股 (Top-N + NDrop + ST/停牌/涨跌停过滤)
│   ├── allocator.py          # 仓位分配 (等权/波动率倒数 + 上限)
│   ├── risk.py               # 风控 (止损/回撤减仓/清仓)
│   └── paper_engine.py       # 模拟盘引擎
│
├── small_cap/                # 小市值 alpha 策略 (v2.0)
│   └── backtest.py           # 完整回测 (~530行, 自包含)
│
├── tests/                    # 测试 (61 通过)
│
├── web/                      # Web 界面 (FastAPI + HTMX + ECharts)
│   ├── main.py
│   ├── routes/ (api/dashboard/backtest/paper/data_status/factors)
│   ├── static/
│   └── templates/
│
├── scripts/                  # 工具脚本
│   ├── run_ml_backtest.py    # 大票 ML 回测
│   ├── run_daily_paper.py    # 每日模拟盘
│   └── ...
│
└── docs/                     # 文档 & 研究
```

---

## 策略逻辑

### 大票策略 (舞 v1.85)

```
候选池: 成交额 Top-500 → 83因子(alpha101/191/custom/fundamental/intraday)
→ IC门禁 + 正交筛选 → Walk-Forward XGBoost+LGBM 集成
→ 5状态 regime 自适应 → NDrop 调仓(K=15,N=2) → 日频调仓
```

### 小票策略 (小市值 alpha v2.0)

```
候选池: 成交额 1000-3000 → 11因子(反转+低波+价值+质量+换手)
→ Walk-Forward XGBoost+LGBM 集成 → 排雷 → 趋势过滤 → NDrop 调仓 → 周度调仓

趋势过滤: CSI1000 < MA60 → 减半仓
排雷: ST/退市/商誉暴雷/庄股/次新
```

---

## 数据库

| 类别 | 表 | 说明 |
|------|---|---|
| 行情 | stock_basic, stock_daily, index_daily, stock_tick, stock_minute | 日线 OHLCV + 分钟线 |
| 基本面 | stock_daily_extra, stock_shareholder, stock_financial, stock_industry, stock_pledge | 估值/财务/行业/质押 |
| 模拟盘 | paper_account, paper_orders, paper_positions, paper_daily_pnl | 账户/订单/持仓/净值 |
| 回测 | backtest_results, strategy_configs, strategy_versions | 回测结果 + 策略管理 |

---

## 启动

```bash
cd /Users/chenwan/Documents/quant
source .venv/bin/activate

# Web 界面
uvicorn web.main:app --host 0.0.0.0 --port 8899

# 大票回测
python scripts/run_ml_backtest.py --strategy 舞

# 小票回测
python -m small_cap.backtest

# 每日模拟盘
python scripts/run_daily_paper.py
```

---

## 变更记录

### v2.0 — 小市值 alpha + 大小票平滑分配 (2026-06-04 ~ 2026-06-06)

**方向重构**：
- 放弃大票因子军备竞赛，转向机构覆盖盲区（成交额 1000-3000 名）
- 11 个原创因子：反转(3) + 低波(2) + 价值 PB/PE(2) + 质量(3) + 换手异动(1)
- 排雷增强：ST/退市、商誉暴雷(>50%)、庄股、次新(<120天)

**全面审计与修复**：
- 修复 7 处未来函数（IC 滞后对齐、Tracker 索引、因子发现）
- 修复 4 处实盘一致性问题（T+1 开盘入场、日收益对齐、止损机制）
- 财务数据 45 天公告滞后，PB/PE 从真实 bps/eps 计算

**弱势期大票反转发现**：
- 小票弱势期（CSI1000 < MA60），Top-200 大票反转策略 Sharpe 3.4（独立验证）
- 年化验证：2020-2025 每年 Sharpe > 2.0，p-value < 0.0001

**大小票平滑分配 v1.0**：
- CSI1000 偏离 MA60 程度 → 平滑调整小票/大票比例（10%~90%）
- 极端弱势提现保护（最低 50% 仓位）
- 6 年结果：Sharpe 1.50, 年化 47.7%, MaxDD 31.3%
- 10 年结果：Sharpe 0.87, 年化 23.1%, MaxDD 36.8%（含 2015 股灾/2016 熔断/2018 贸易战）

**成本模型**：小票滑点 0.3%、大票滑点 0.1%（真实流动性折价）

### v1.85 — 自适应 NDrop (2026-06-05)

- 自适应 N：基于分数离散度动态调整替换数 (1~4)
- Tier 1 修复：forward_days 5→1, IC 阈值 0.02→0.03, 体制阈值 ±3%→±5%, 正交阈值 0.7→0.5
- 前视修复：滞后 IC 计算、Tracker IC 索引修正

### v1.8 — Universe 优化 (2026-06-04)

- Universe 500→1000
- 模拟盘 account_id=15, run_id=2

### v1.7 — +50% 止盈 (2026-06-03)

- 止盈线 50%
- 模拟盘统一

### v1.5 ~ v1.6 — 因子扩展 + 前复权 (2026-05-31 ~ 2026-06-03)

- v1.6: 全量因子 (qfq 前复权)
- v1.5: 板块打分 + 前复权

### v1.4 — 实盘误差消除 (2026-05-30 ~ 2026-05-31)

- 跌停顺延、补全 extra_data、模拟盘/回测打通
- 策略定名「舞」

### v1.3 — 市场状态自适应 (2026-05-30)

- 5 状态市场检测 (强牛/弱牛/快熊/慢熊/震荡)
- 按状态自适应调仓频率/仓位/止损

### v1.12 — 分钟因子 + 行业中性化 + 多周期预测 (2026-05-28 ~ 2026-05-29)

- 7 个 60min K 线日内因子、行业截面中性化 (19 个 SW1 行业)
- T+1/5/20 多周期集成预测
- 72 组参数网格搜索确定最优组合

### v1.11 — 动态反馈闭环 (2026-05-28)

- 因子淘汰/发现 + 日度信号追踪
- IC 衰减 → 权重衰减 → 淘汰/重训

### v1.10 — 组合优化 (2026-05-25 ~ 2026-05-27)

- 全市场选股、8 项排雷过滤、NDrop 增量调仓
- 组合风控 (-20%减仓/-25%清仓)、真实交易成本
- ML 策略差异化（因子预设 + 算法选择 + Optuna 调优）
- Web 架构从 Streamlit 迁移到 FastAPI + HTMX

### 前期 (2026-05-22 ~ 2026-05-25)

- 项目初始化、数据模块、OHLCV 同步（腾讯/新浪/交易所 API）
- Alpha101 + Alpha191 因子库建设、IC 监控
- ML 管线 (XGBoost/LGBM + Walk-Forward)
- 组合优化（选股/仓位/风控）、模拟盘引擎
- 批量回测系统、过拟合检测
