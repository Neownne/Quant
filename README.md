# Quant — A 股量化交易系统

> 最后更新：2026-05-26  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)（私有仓库）

---

## 核心工作流

```
因子研究 (ML Monitor)  →  策略构建 (Strategy Editor)  →  历史回测 (Backtest)
                                                              │
                                                    保存结果 / 对比历史
                                                              │
                                                     🚀 升级到模拟盘
                                                              │
                                                              ▼
                                                      模拟盘交易 (Paper Trade)
                                                      ├── ML 策略: PaperEngine
                                                      └── 静态策略: StaticPaperEngine
                                                              │
                                               ┌──────────────┴──────────────┐
                                               │  权益曲线 / 持仓分布 / P&L  │
                                               │  委托历史 / 已平仓交易 / 策略信息 │
                                               └─────────────────────────────┘

数据监控 (Data Monitor)  ←── 自动同步调度 + 质量检查（5 种模式每日 18:00）
```

---

## 架构概览

```
                    ┌──────────────────────────────────────────────────────────┐
                    │               Streamlit Web UI (app/)                     │
                    │  ┌──────┐ ┌──────┐ ┌───────┐ ┌──────┐ ┌─────┐ ┌──────┐ ┌──────┐│
                    │  │实时报价│ │K线图 │ │策略回测│ │模拟盘│ │股票池│ │ML监控│ │数据监控││
                    │  └──┬───┘ └──┬───┘ └───┬───┘ └──┬───┘ └──┬──┘ └──┬───┘ └──┬───┘│
                    │     └────────┼──────────┼────────┼────────┘       │        │    │
                    │   data_loader / backtest_runner / account_manager /        │    │
                    │   stock_pools / paper_engine / sync.check_data_quality     │    │
                    └──────────────┼──────────────────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │            data/ 数据层                  │
              │  ┌──────────┐  ┌──────┐  ┌───────────┐  │
              │  │fetcher.py│  │db.py │  │sync.py     │  │
              │  │ AKShare  │  │ PG   │  │数据同步     │  │
              │  └────┬─────┘  └──┬───┘  └─────┬─────┘  │
              │       └───────────┼─────────────┘        │
              └───────────────────┼──────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     PostgreSQL 数据库       │
                    │  15 张表（行情 + 基本面 + 回测 + 模拟盘）│
                    └───────────────────────────┘
```

---

## 目录结构

```
quant/
├── .env                      # 环境变量（不入 git）
├── .env.example              # 环境变量模板
├── .gitignore
├── requirements.txt          # Python 依赖
├── README.md                 # 本文件
│
├── config/
│   └── settings.py           # 集中配置（数据库、数据参数）
│
├── data/                     # 数据层
│   ├── db.py                 # 15 张表 DDL + 连接池 + upsert
│   ├── fetcher.py            # AKShare 数据获取（日线/分钟/实时/估值/股东/财务/质押）
│   ├── sync.py               # 历史数据批量同步（多进程增量，10 种模式 + 质量检查）
│   └── recorder.py           # 实盘数据录制（分钟K线 + 逐笔）
│
├── strategies/               # 策略层
│   ├── README.md             # 策略文档（逻辑、参数、风险提示）
│   ├── __init__.py           # STRATEGY_REGISTRY + 自定义策略加载
│   ├── sma_cross.py          # 双均线交叉策略
│   ├── macd_strategy.py      # MACD 金叉死叉策略
│   └── rsi_strategy.py       # RSI 超买超卖策略
│
├── factors/                    # 因子层（76 个因子）
│   ├── __init__.py             # ALL_FACTORS 注册表（76 个因子）
│   ├── engine.py               # FactorEngine + 截面中性化
│   ├── alpha101.py             # 30 个 Alpha101 核心因子
│   ├── alpha191_turnover.py    # Alpha191 换手率类 5 因子
│   ├── alpha191_intraday.py    # Alpha191 日内形态类 4 因子
│   ├── alpha191_flow.py        # Alpha191 资金流向类 6 因子
│   ├── alpha191_gap.py         # Alpha191 隔夜效应类 4 因子
│   ├── alpha191_vol.py         # Alpha191 波动率高阶类 5 因子
│   ├── alpha191_liquidity.py   # Alpha191 流动性高阶类 4 因子
│   ├── custom.py               # 7 个自定义 A 股因子
│   ├── fundamental.py          # 11 个基本面质量因子（九项排雷）
│   ├── monitor.py              # IC/ICIR/衰减曲线监控
│   └── screening.py            # 正交性筛选（Spearman + 贪心）
│
├── models/                     # ML 预测层
│   ├── __init__.py             # 模块导出
│   ├── dataset.py              # 因子数据集构造 + walk-forward 切分
│   ├── trainer.py              # XGBoost/LightGBM 训练器 + EnsemblePredictor
│   ├── predictor.py            # DailyPredictor 横截面打分排序
│   ├── regime.py               # 市场状态识别（牛/熊/震荡）
│   └── tuning.py               # 阈值搜索 + Optuna 超参调优
│
├── portfolio/                  # 组合优化层
│   ├── __init__.py             # 模块导出
│   ├── selector.py             # 选股（Top-N + ST/停牌/涨跌停/次新过滤）
│   ├── allocator.py            # 仓位分配（等权/波动率倒数 + 单只10%/行业30%上限）
│   ├── risk.py                 # 风控（个股-8%止损/ATR止损/组合回撤减仓/最大回撤清仓）
│   └── paper_engine.py         # PaperEngine 日频ML模拟盘引擎（信号→订单→DB写入）
│
├── tests/                      # 测试（61 通过 + 1 跳过）
│   ├── test_fetcher.py         # 数据获取测试
│   ├── test_factors.py         # 因子+引擎+IC 测试
│   ├── test_alpha191.py        # Alpha191 因子测试（13 个）
│   ├── test_models.py          # ML 模型测试（数据集+训练+集成+调优）
│   ├── test_regime.py          # 市场状态识别测试
│   ├── test_screening.py       # 正交筛选测试
│   └── test_portfolio.py       # 组合优化测试（选股+仓位+风控）
│
├── app/                      # Web 界面层
│   ├── main.py               # 入口：导航 + sys.path + 每日同步调度
│   ├── pages/
│   │   ├── 1_📈_Watchlist.py     # 自选股实时报价
│   │   ├── 2_📊_Charts.py        # K线图 + 技术指标 + 自选分组
│   │   ├── 3_🧪_Backtest.py      # 策略回测（单只 / 股票池批量）
│   │   ├── 4_📋_Paper_Trade.py   # 模拟盘账户/持仓/委托
│   │   ├── 5_📝_Strategy_Editor.py # 在线策略编辑器
│   │   ├── 6_📦_Stock_Pools.py   # 自定义股票池（代码编写筛选规则）
│   │   ├── 7_🔴_Recorder.py      # 数据录制（分钟K线 + 逐笔交易）
│   │   ├── 8_📊_ML_Monitor.py    # ML策略监控（IC看板/模型表现/当日信号/模拟盘净值/数据健康）
│   │   └── 9_📡_Data_Monitor.py  # 数据同步监控（质量概览 + 手动同步 + 覆盖度追踪）
│   └── utils/
│       ├── account_manager.py # 统一账户 CRUD（创建/查询/更新/策略绑定）
│       ├── data_loader.py    # DB查询 + 指标计算 + K线图构建 + 实时行情
│       ├── backtest_runner.py # backtrader 回测封装 + A股费用 + 大盘过滤器
│       └── stock_pools.py    # 股票池引擎（加载/编译/执行筛选规则）
│
├── scripts/                  # 批量工具
│   ├── batch_backtest.py         # 全量回测：所有股票 × 策略 × 参数 × 牛熊市
│   ├── grid_backtest.py          # 全市场回测 + 参数网格搜索 + 过拟合检测
│   ├── run_ml_backtest.py        # ML 选股端到端评估（因子→筛选→训练→汇总）
│   ├── run_simulation.py         # ML 每日模拟回测（选股→分配→止损→P&L）
│   └── verify_paper_trading.py   # 端到端验证（simulation vs PaperEngine 对比）
│
├── docs/                     # 文档 & 研究报告
│   ├── quant-strategy-research.md  # 量化策略研究报告（技术指标+学术前沿）
│   ├── strategy-interpretability.md # ML策略可解释性分析（因子+筛选+回测验证）
│   └── postgresql-guide.md         # PostgreSQL 使用指南
│
├── output/                   # 批量回测输出（不入 git）
│   ├── batch_results.csv     # 逐笔回测结果明细
│   └── batch_summary.md      # 排名汇总报告
│
└── notebooks/                # Jupyter Notebook（预留）
```

---

## 数据流

### 1. 历史数据同步

```
data/sync.py
  │
  ├──> data/fetcher.py  ─── AKShare API ─── 腾讯/新浪/百度/交易所
  │         │
  │         └── 返回 DataFrame（字段对齐数据库表结构）
  │
  └──> data/db.py       ─── PostgreSQL
            │
            └── upsert_df()：临时表 → ON CONFLICT DO UPDATE
```

**sync.py 同步模式**：
| 模式 | 说明 | 数据源 |
|---|---|---|
| `stock` | 股票基本信息（行业/上市日期） | 深交所 + 上交所 + 北交所 |
| `stock-daily` | 日线 OHLCV（后复权，2015 年起） | 腾讯财经 |
| `index` | 7 只指数日线 | 腾讯财经 |
| `daily-extra` | 估值指标（市值/PE/PB/股本，~1 年日频） | 百度财经 |
| `shareholder` | 股东户数（季度数据，2013 年起） | 东方财富 |
| `financial` | 财务摘要（营收/利润/ROE/毛利率等） | 同花顺 |
| `financial-supplement` | 财务补充（资产负债表/现金流/扣非净利润） | 同花顺 `_new_ths` 系列 |
| `pledge` | 全市场股权质押数据 | 东方财富 |
| `industry` | 行业分类（申万一级/二级） | 东方财富 |
| `all` | 全量同步 | — |

**特性**：多进程并发（ProcessPoolExecutor）、增量同步（跳过已覆盖股票）、单股 60s 超时、tqdm 进度条。

## ML 选股回测结果

> 详见 [docs/strategy-interpretability.md](docs/strategy-interpretability.md)

| 指标 | 800-Stock 全市场 | 说明 |
|------|:---:|------|
| Sharpe | **1.01** | 经无风险利率调整 |
| 年化收益 | **26.3%** | 2021-2026 区间 |
| 最大回撤 | -36.7% | 主要发生在 2022 年熊市 |
| 胜率 | 53.7% | 日频预测准确率 |
| Walk-forward 窗口 | 4 | 滚动训练+验证 |

**参数优化**（50-stock 网格搜索，12 组）：
- 最优参数：`top_n=15, ndrop=True, ndrop_n=2`
- NDrop 增量调仓在所有参数级别均优于每日全换仓
- top_n=15 是甜点区（10 太集中，20 太分散）

**过拟合检测**：样本外 (2024-2026) Sharpe=2.27 > 样本内 (2019-2022) Sharpe=1.09，衰减比 208.1%，**无过拟合**。

**因子筛选链**：76 个因子 → IC 门禁 (65→20, \|IC\|>0.02, \|t\|>2.0) → 正交筛选 (20→8, Spearman<0.7) → XGBoost+LightGBM 概率平均集成

**基本面排雷**（月频）：8 项排雷综合评分 (audit_score ≥ -3) + 三项硬排除（商誉>27%/质押>72%/负债>105%），每月约 330-390/800 只通过。

---

### 2. 股票池系统

```
app/pages/6_📦_Stock_Pools.py  (编辑器)
  │
  └──> app/utils/stock_pools.py  (引擎)
        │
        ├── 池文件保存在 ~/.quant_stock_pools/*.py
        ├── 每文件定义 filter_stocks(basic, extra, shareholder) -> list[str]
        ├── 实时筛选预览（编译测试 + 运行预览）
        └── 被回测页面集成：支持「单只股票」或「股票池批量」回测
```

**数据输入**：
- `basic` — 股票基本信息（代码、名称、行业、市场、上市日期、ST 标记）
- `extra` — 估值指标最新快照（市值、PB）
- `shareholder` — 股东户数最新报告期

**筛选范式示例**：小盘股（流通市值 10-200 亿）、高散户参与度（股东户数 > 20000）、排除特定行业、排除 ST/新股、低 PE 价值股。

### 3. Web 界面数据流

```
app/main.py (入口)
  │
  ├── 后台线程：每日 18:00 自动运行 5 种同步模式
  │   (index / stock-daily / daily-extra / shareholder / financial)
  │
  └── 页面导航
        │
        ├── Watchlist ──> data_loader.get_realtime_quotes()
        │                  └── AKShare stock_zh_a_spot_em()（5s 缓存）
        │                  非交易时段回退：
        │                  └── data_loader.get_latest_daily_batch()
        │                      └── stock_daily 表最近交易日数据
        │
        ├── Charts ──> data_loader.load_ohlcv()     ──> stock_daily
        │              data_loader.build_kline_chart() ──> Plotly Figure
        │              data_loader.calc_ma/ema/macd/rsi/bollinger()
        │
        ├── Backtest ──> backtest_runner.run_backtest()
        │                 │
        │                 ├── 单只模式：K 线图 + 权益曲线 + 交易明细
        │                 ├── 股票池模式：逐只回测 → 汇总统计 + 排名表 + 分布直方图
        │                 ├── 结果持久化 → backtest_results 表（历史对比/CSV 导出）
        │                 ├── 🚀 升级到模拟盘 → account_manager.promote_strategy_to_account()
        │                 ├── PgDataFeed：DataFrame → backtrader 数据源
        │                 ├── AShareCommission：A股真实费用
        │                 │   ├── 佣金 万0.9（买卖双向）
        │                 │   └── 印花税 万5（仅卖出）
        │                 └── MarketAwareSizer：动态仓位管理
        │                     ├── 牛市（指数 ≥ 200MA）→ 95% 仓位
        │                     └── 熊市（指数 < 200MA）→ 自动降至 40%
        │
        ├── Paper Trade ──> portfolio/paper_engine
        │                    │
        │                    ├── ML 策略 → PaperEngine.run_daily()
        │                    ├── 静态策略 → StaticPaperEngine.run_replay()
        │                    ├── 权益曲线 + 回撤阴影（Plotly）
        │                    ├── 行业分布饼图 + 个股权重柱状图
        │                    └── 策略信息卡片（类型/名称/参数/费率）
        │
        ├── Stock Pools ──> stock_pools 引擎
        │                    └── filter_stocks() 编译/测试/保存/预览
        │
        ├── ML Monitor ──> portfolio/paper_engine + factors/monitor
        │                  ├── IC 看板：因子 RankIC 柱状图 + 时序 + 衰减
        │                  ├── 模型表现：Walk-forward 指标表 + 特征重要性
        │                  ├── 当日信号：最新截面预测 Top-20 + 行业分布
        │                  ├── 模拟盘净值：权益曲线 + 回撤图 + 策略信息
        │                  └── 数据健康：各表最新日期/覆盖度/缺失检测
        │
        ├── Data Monitor ──> sync.check_data_quality()
        │                    ├── 数据质量概览（10 张表状态）
        │                    ├── 最近交易日覆盖度进度条
        │                    └── 手动同步触发（7 种模式独立触发）
        │
        └── Strategy Editor ──> ~/.quant_strategies/（保存 .py 文件）
                                │
                                └── strategies/__init__.py 动态加载
```

---

## 数据库表结构（15 张表）

### 行情数据（5 张）

| 表 | 主键 | 说明 | 数据源 |
|---|---|---|---|
| `stock_basic` | code | 5,522 只股票基本信息（含行业/上市日期） | 深交所 + 上交所 + 北交所官网 |
| `stock_daily` | (code, trade_date) | 日线 OHLCV + 换手率（后复权，2015-01 起） | 腾讯财经 |
| `index_daily` | (code, trade_date) | 7 只指数日线（上证/深证/创业板/科创/沪深300/中证500/中证1000） | 腾讯财经 |
| `stock_tick` | (code, trade_time) | 逐笔成交（实时录制） | 腾讯财经逐笔 |
| `stock_minute` | (code, trade_time, period) | 分钟K线（实时录制） | 新浪财经分钟K线 |

### 基本面数据（5 张）

| 表 | 主键 | 说明 | 数据源 |
|---|---|---|---|
| `stock_daily_extra` | (code, trade_date) | 估值指标（总市值、流通市值、PE、PB、总股本、流通股本） | 百度财经 |
| `stock_shareholder` | (code, end_date) | 股东户数（季度，含户均持股市值/持股数量） | 东方财富 |
| `stock_financial` | (code, report_date) | 财务数据（营收/利润/ROE/毛利率/净利率/EPS/BPS + 资产负债表+现金流+扣非净利润，共18列） | 同花顺（AKShare `_new_ths` 系列） |
| `stock_industry` | code | 行业分类（申万一级/二级行业） | 东方财富 |
| `stock_pledge` | (code, trade_date) | 股权质押数据（质押比例/股数/市值/笔数） | 东方财富 |

### 模拟盘 & 回测（5 张）

| 表 | 说明 |
|---|---|
| `paper_account` | 模拟账户（名称、初始资金、现金、策略类型/名称/参数、费率、大盘过滤） |
| `paper_orders` | 委托记录（代码、方向、价格、数量、状态） |
| `paper_positions` | 当前持仓（代码、数量、均价） |
| `paper_daily_pnl` | 每日净值（现金、持仓市值、总资产、日收益、回撤） |
| `backtest_results` | 回测结果持久化（策略参数、资产模式、股票池、指标汇总、完整结果 JSON） |

> ETF/基金相关 4 张表（etf_basic / etf_daily / fund_basic / fund_nav）DDL 保留但未启用。fetcher.py 和 sync.py 中对应函数代码完整保留但调用链已注释。

---

## 策略系统

### 架构

```
strategies/__init__.py
  │
  ├── STRATEGY_REGISTRY（内置策略）
  │   ├── "双均线交叉"  → SMACross
  │   ├── "MACD金叉死叉" → MACDStrategy
  │   └── "RSI超买超卖"  → RSIStrategy
  │
  └── get_all_strategies() → 内置 + ~/.quant_strategies/ 自定义策略
        │
        └── 回测页面 STRATEGY_REGISTRY 动态策略列表
```

### 策略详情

详见 [strategies/README.md](strategies/README.md)

| 策略 | 信号 | 默认参数 | 适用场景 |
|---|---|---|---|
| 双均线交叉 | 快线上穿买入，下穿卖出 | (5, 20) | 趋势市 |
| MACD 金叉死叉 | DIF 上穿 DEA 买入，下穿卖出 | (12, 26, 9) | 趋势转折 |
| RSI 超买超卖 | RSI<30 买入，RSI>70 卖出 | (14, 30, 70) | 震荡市 |

### 自定义策略（Web 编辑器）

- 在线编写 backtrader 策略代码，保存为 `~/.quant_strategies/*.py`
- 自动注册到回测页面的策略下拉列表
- 编写指南内嵌在编辑器中（含指标、止损模板示例）

---

## 股票池系统

### 自定义股票池（Web 编辑器）

- 在线编写 Python 筛选代码，保存为 `~/.quant_stock_pools/*.py`
- 每文件定义 `filter_stocks(basic, extra, shareholder) -> list[str]`
- 数据输入：股票基本信息 + 估值指标 + 股东户数
- 实时的数据概览（股票总数、估值覆盖率、股东数据覆盖率）
- 筛选预览（编译 → 运行 → 显示前 20 匹配股票）

### 回测集成

- 回测页面新增「单只股票 / 股票池」模式切换
- 股票池模式：逐只回测 → 汇总统计（均值/中位数/夏普/回撤/胜率）
- 排名表 + 收益率分布直方图 + CSV 导出

---

## 批量回测系统

### scripts/batch_backtest.py

全量回测：所有股票 × 策略参数网格 × 牛熊市周期。

```
用法：
    python scripts/batch_backtest.py                         # 全量运行（8 并发）
    python scripts/batch_backtest.py --workers 16            # 指定并发数
    python scripts/batch_backtest.py --limit 50 --resume     # 只测前 50 只，从断点继续
```

**特性**：
- 细粒度并行：每个 策略×参数×周期 组合作为独立任务提交 ProcessPoolExecutor
- Worker 级 OHLCV 缓存：同一进程内同只股票只查一次 DB
- batch_mode 加速：跳过 TimeReturn/CSV writer 等非必要输出
- 大盘年线过滤器：牛市 95% 仓位 / 熊市 40% 仓位
- 断点续跑 + 增量保存

### 回测结果

| 策略 | 平均收益率% | 平均胜率% | 平均回撤% |
|---|---|---|---|
| RSI超买超卖 | 11.6 | 47.2 | 19.1 |
| MACD金叉死叉 | 4.8 | 33.1 | 34.1 |
| 双均线交叉 | 2.6 | 28.3 | 34.1 |

- RSI(7, 20, 80) 在全周期上有大量 100% 胜率案例（交易次数 ≥ 10）
- 熊市（慢熊2021-24）平均收益 -7.3%，反弹期（2024-26）平均 15.4%

---

## 量化策略研究

详见 [docs/quant-strategy-research.md](docs/quant-strategy-research.md)，涵盖：

1. **技术指标数学原理**：SMA/EMA/MACD/RSI/Stochastic/CCI/Bollinger/ATR/OBV
2. **经典策略体系**：趋势跟踪 / 均值回归 / 配对交易 / 统计套利 / 动量
3. **Alpha101 & Alpha191 因子体系**：5 大类因子 + A 股适配
4. **2020-2025 因子挖掘新范式**：
   - LLM + RL 公式化因子自动发现（中科院 2025）
   - ABCM 神经网络 Alpha-Beta 协同挖掘（东方证券 2024）
   - 分析师预期正交因子（华泰证券 2024）
   - Risk-Attention / VAE / KAN-Autoencoder 因子模型
5. **ML 前沿应用**：XGBoost/LightGBM/Transformer/GAN
6. **可行性路线图**：因子库建设 → 单模型预测 → 正交 Alpha 扩展 → 集成优化

---

## ML 环境

已安装（Python 3.14）：

| 包 | 版本 | 用途 |
|---|---|---|
| scikit-learn | 1.8.0 | 特征工程、模型评估 |
| xgboost | 3.2.0 | 梯度提升（因子→收益预测） |
| lightgbm | 4.6.0 | 梯度提升（大规模特征） |
| statsmodels | 0.14.6 | 统计检验、回归分析 |
| torch | 2.12.0 | 深度学习（Transformer/LSTM） |

---

## 文件关联关系

### config/settings.py → 全局配置中心
- `DBConfig`：被 `data/db.py` 的 `get_engine()` 使用
- `DataConfig`：被 `data/sync.py` 和 `data/fetcher.py` 使用

### data/fetcher.py → 数据获取
- 被 `data/sync.py` 调用（批量历史同步）
- 被 `data/recorder.py` 调用（实盘录制）
- 被 `app/utils/data_loader.py` 间接调用（实时行情）
- `@retry_on_network_error` 装饰器提供指数退避自动重试
- `fetch_stock_lg_indicator()` — 估值指标（市值/PE/PB/股本，数据源百度财经）
- `fetch_shareholder_count()` — 股东户数（季度，数据源东方财富）
- `fetch_financial_data()` — 财务摘要（营收/利润/ROE/毛利率/净利率/EPS/BPS/现金流，数据源同花顺）
- `fetch_financial_supplement()` — 财务补充（资产负债表/现金流/扣非净利润，数据源同花顺 `_new_ths` 系列）
- `fetch_pledge_data()` — 股权质押（全市场快照，数据源东方财富）
- `fetch_industry_classification()` — 行业分类（申万一级/二级，数据源东方财富）

### data/db.py → 数据库层
- `init_db()`：建表（幂等，15 张表），被 `app/main.py` 各页面调用
- `upsert_df()`：写入，被 `data/sync.py` 和 `data/recorder.py` 调用

### data/sync.py → 数据同步
- 10 种同步模式：stock / stock-daily / index / daily-extra / shareholder / financial / financial-supplement / pledge / industry / all
- `check_data_quality(engine)` — 数据质量报告（覆盖度/过期天数/记录数），被数据监控页面调用
- 默认起始日期 20150101（10 年历史）
- 多进程并发（ProcessPoolExecutor），模块级 worker 函数

### app/utils/data_loader.py → Web 数据服务
- `@st.cache_data` 缓存策略：实时行情 5s、日线 60s、股票列表 3600s
- `build_kline_chart()`：生成 Plotly 多 pane K 线图

### app/utils/backtest_runner.py → 回测引擎
- `AShareCommission`：A 股真实交易费用
- `MarketAwareSizer`：大盘年线过滤器
- `load_index_data()`：加载上证指数日线
- `TradeRecorder`：FIFO 逐笔交易记录器
- `SignalRecorder`：回测逐笔交易信号采集，支持策略升级到模拟盘
- `batch_mode=True`：跳过权益曲线/CSV 输出加速批量回测

### app/utils/account_manager.py → 统一账户管理
- `create_account()` — 创建模拟账户（含策略参数+费率配置）
- `get_account()` / `update_account_config()` — 查询/更新账户配置
- `list_accounts()` — 列出所有账户
- `promote_strategy_to_account()` — 将回测策略升级为模拟账户

### app/utils/stock_pools.py → 股票池引擎
- 池文件 `~/.quant_stock_pools/*.py`，每文件定义 `filter_stocks()`
- `compile_pool()` / `execute_pool()` / `get_pool_data()`
- 被回测页面集成，支持批量回测

### scripts/batch_backtest.py → 批量回测
- 细粒度并行 + worker 级数据缓存 + batch_mode 加速
- 输出 `output/batch_results.csv` + `output/batch_summary.md`

### strategies/__init__.py → 策略注册
- 被 `app/pages/3_🧪_Backtest.py` 调用
- 动态扫描 `~/.quant_strategies/` 加载自定义策略

---

## 启动方式

```bash
cd /Users/chenwan/Documents/quant
source .venv/bin/activate

# 数据同步（首次使用）
python -m data.sync --mode stock           # 股票基本信息
python -m data.sync --mode stock-daily     # 日线行情
python -m data.sync --mode daily-extra     # 估值指标
python -m data.sync --mode shareholder     # 股东户数

# 启动 Web 界面
streamlit run app/main.py                  # 浏览器打开 http://localhost:8501
```

---

## 变更记录

| 日期 | 变更内容 |
|---|---|
| 2026-05-26 | **架构重构：统一量化工作流**：统一账户配置系统 `account_manager.py`（策略类型/名称/参数/费率绑定到 paper_account）；回测结果持久化 `backtest_results` 表（历史对比/查看详情/CSV 导出）；策略→模拟盘无缝衔接（回测页"🚀 升级到模拟盘"→账户选择→跳转）；模拟盘页面全面升级（权益曲线+回撤阴影/行业分布饼图/个股权重柱状图/策略信息卡片）；StaticPaperEngine 支持静态策略模拟盘（backtrader 信号回放）；ML Monitor 新增"数据健康"Tab（5 页签）；新建 `app/pages/9_📡_Data_Monitor.py` 数据监控页面（质量概览+手动同步+覆盖度追踪）；数据同步调度扩展为 5 种模式（index/stock-daily/daily-extra/shareholder/financial）；数据库 14→15 张表；所有 11 个 Python 文件 AST 解析通过 |
| 2026-05-26 | **全市场回测+参数优化+过拟合检测**：`scripts/grid_backtest.py` 新增参数网格搜索（12组合）+ 过拟合检测（样本内2019-2022 vs 样本外2024-2026）。800只股票（沪深300+中证500成分股）全市场回测完成：Sharpe=1.01, 年化收益26.3%, 最大回撤-36.7%, 4个Walk-forward窗口。最优参数：top_n=15, NDrop(ndrop_n=2)。因子筛选链：76因子→IC门禁(65→20, |IC|>0.02, |t|>2.0)→正交筛选(20→8, Spearman<0.7)。过拟合诊断：无（样本外Sharpe 2.27 > 样本内 1.09, 衰减比208.1%）。`docs/strategy-interpretability.md` 策略可解释性分析文档（因子经济学含义+IC门禁+正交筛选+双周期设计+NDrop逻辑+特征重要性+失效场景+改进方向）。行业数据补充（沪市/北交所）因东方财富API不可用暂时跳过 |
| 2026-05-25 | **基本面因子扩展+财务数据补全**：`factors/fundamental.py` 新增 fin_debt_ratio（资产负债率）、fin_goodwill_ratio（商誉风险）、fin_pledge_risk（质押风险）3 个因子，fin_cashflow_gap 增强为优先使用经营现金流总额对比，fin_audit_score 扩展至 8 项排雷检查（新增扣非<0/负债>70%/商誉>30%）。`data/fetcher.py` 新增 fetch_financial_supplement（资产负债表+现金流+利润表）、fetch_pledge_data（质押快照）、fetch_industry_classification（申万行业）。`data/db.py` 扩展 stock_financial（+7 列 ALTER TABLE）+ 新增 stock_pledge 表。因子总数 65→76，9 项排雷可用 6/9。数据库 13→14 张表。61 测试通过 |
| 2026-05-25 | **Phase 4 组合优化+风控+模拟盘**：选股增强（停牌/涨跌停过滤）；仓位上限（单只10%+行业30%迭代裁剪+再分配）；风控增强（ATR止损/组合-20%减仓50%/最大-25%清仓）；新建 `portfolio/paper_engine.py`（PaperEngine 日频ML模拟盘引擎，过滤→预测→分配→风控→订单→DB写入）；新增 `paper_daily_pnl` 表（第13张表）；新建 `app/pages/8_📊_ML_Monitor.py`（IC看板/模型表现/当日信号/模拟盘净值四Tab）；`scripts/run_simulation.py` 重构为使用 portfolio/risk 函数；`scripts/verify_paper_trading.py` 端到端验证（simulation vs PaperEngine 选股重合度100%收益相关性1.0）。59测试通过 |
| 2026-05-25 | **Phase 3 因子扩展+集成优化**：新增 28 个 Alpha191 A 股因子（换手率5/日内形态4/资金流向6/波动率5/隔夜效应4/流动性4 共 6 类）；激活 extra_data（市值+PB+股东户数）激活 3 个估值因子；市场状态识别 `models/regime.py`（牛/熊/震荡三态）；正交筛选 `factors/screening.py`（Spearman 秩相关 < 0.7 门禁）；XGBoost+LightGBM 概率平均集成 `EnsemblePredictor`；`models/tuning.py` 阈值搜索+Optuna 调优。因子总数 37→65，经正交筛选 ≥ 50 活跃。E2E 集成回测 4 窗口 acc=54.2%, prec=52.1%, rec=26.9%。全量 59 测试通过 + 1 跳过 |
| 2026-05-25 | **Phase 2 ML 选股管线**：新增 `models/` 模块（dataset 构造+walk-forward 切分、XGBoost/LightGBM 训练器、DailyPredictor 截面排序）；新增 `portfolio/` 模块（Top-N 选股+ST/次新过滤、等权/波动率倒数分配、个股止损+回撤风控）；新增 `scripts/run_ml_backtest.py` 端到端验证脚本；新增 16 个测试（models 9 + portfolio 7），全量 36 测试通过；Phase 1+2 合计 37 个因子 + IC 监控 + ML 训练预测 + 组合优化 |
| 2026-05-25 | **股票池系统**：新增自定义股票池编辑器（`app/pages/6_📦_Stock_Pools.py`），支持 Python 代码定义筛选规则，集成回测批量模式；**回测升级**：支持单只/股票池双模式，池模式含汇总统计+排名表+分布直方图+CSV 导出；**基本面数据**：新增 `stock_daily_extra`（市值/PB）和 `stock_shareholder`（股东户数）2 张表及对应 fetcher/sync；**历史数据扩展**：stock_daily 起始日期 2020→2015，约 1,000 万行；**fetcher 修复**：`stock_a_indicator_lg` 不存在→改用 `stock_zh_valuation_baidu` |
| 2026-05-25 | **清理 ETF/基金**：删除 etf_basic/etf_daily/fund_basic/fund_nav 4 张表；data_loader 路由点 + 页面标签精简为纯股票；fetcher/sync 中对应代码注释保留可恢复；数据库从 12 表→8 表。**ML 环境就绪**：sklearn 1.8.0 / xgboost 3.2.0 / lightgbm 4.6.0 / statsmodels 0.14.6 / torch 2.12.0。**量化策略研究报告**：新增 2020-2025 因子挖掘新范式（LLM+RL/ABCM/分析师因子/Risk-Attention/VAE-KAN），更新参考文献至 19 篇。**回测性能优化**：细粒度并行 + worker 级数据缓存 + batch_mode 加速，默认 8 并发。**基金净值同步 bug 修复**：accumulated_nav 列 None→np.nan |
| 2026-05-24 | **A 股实盘交易规则**：`AShareCommission`（佣金万0.9 买卖双向 + 印花税万5 卖出单向）；**大盘年线过滤器**：`MarketAwareSizer` 牛市 95% / 熊市 40% 动态仓位；**批量回测系统**：`scripts/batch_backtest.py` 全股票多策略多参数牛熊市对比，输出排名报告；基金净值同步（部分数据源缺失跳过） |
| 2026-05-24 | ETF/基金全页面支持：data_loader 按资产类型自动路由；Git 初始化 + GitHub 私有仓库推送 |
| 2026-05-24 | Web 策略编辑器；策略文档；自选股非交易时段回退；K线图分组切换；回测默认参数更新 |
| 2026-05-24 | Streamlit Web 界面：实时报价、K线图+技术指标、策略回测、模拟盘、每日自动同步；模拟盘 3 张表；backtrader 回测引擎 |
| 2026-05-23 | 弃用东方财富，迁移至腾讯/新浪/交易所；后复权统一；sync v2（tqdm + ProcessPoolExecutor + 超时）；stock_tick/stock_minute 表 + recorder.py |
| 2026-05-22 | 初始化项目结构、数据模块、配置文件、ETF/基金支持、网络重试机制 |
