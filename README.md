# Quant — A 股量化交易系统

> 最后更新：2026-05-30 (v1.3)  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)（私有仓库）

---

## 核心工作流

```
因子计算(定时任务) → 模型训练 → 历史回测(防过拟合检查) → 模拟盘验证 → 归因反馈闭环
```

---

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│           Web 监控层 (FastAPI + HTMX + Alpine.js)         │
│  行情看板 │ 回测对比 │ 模拟盘 │ 数据状态 │ 因子监控        │
│  routes/api.py (主API) + dashboard/backtest/paper 路由   │
│  templates/ (Jinja2 + ECharts 图表)                      │
└──────────────────────┬──────────────────────────────────┘
                       │ 只读查询
┌──────────────────────┴──────────────────────────────────┐
│                  PostgreSQL (31张表)                      │
└──────────────────────┬──────────────────────────────────┘
                       │ 读写
┌──────────────────────┴──────────────────────────────────┐
│              后台研究引擎 (脚本/定时任务)                    │
│  数据同步→质量校验→因子计算→训练→回测→归因→调参            │
│  v1.12: 分钟因子(7) + 行业中性化 + 多周期预测(T+1/5/20)    │
└─────────────────────────────────────────────────────────┘
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
│   ├── settings.py           # 集中配置（数据库、数据参数）
│   └── com.quant.sync.plist  # macOS 定时同步任务配置
│
├── data/                     # 数据层
│   ├── db.py                 # 29 张表 DDL + 连接池 + upsert
│   ├── fetcher.py            # 数据获取（日线:AKShare/腾讯, 分钟:新浪直连API）
│   ├── sync.py               # 历史数据批量同步（多进程增量，10 种模式 + 质量检查）
│   ├── recorder.py           # 实盘数据录制（分钟K线 + 逐笔）
│   ├── quality.py            # 数据质量校验（覆盖度/缺失/异常检测）
│   ├── lineage.py            # 数据血缘追踪（来源→转换→消费者）
│   └── availability.py       # 交易日历 + 数据可用性检查
│
├── strategies/               # 策略层
│   ├── README.md             # 策略文档（逻辑、参数、风险提示）
│   ├── __init__.py           # STRATEGY_REGISTRY + 自定义策略加载
│   ├── sma_cross.py          # [已废弃] 双均线交叉策略
│   ├── macd_strategy.py      # [已废弃] MACD 金叉死叉策略
│   ├── rsi_strategy.py       # [已废弃] RSI 超买超卖策略
│   └── grid_shock.py         # 震荡网格(高抛低吸)策略
│
├── factors/                    # 因子层（83 个因子）
│   ├── __init__.py             # ALL_FACTORS 注册表（83 个因子）
│   ├── engine.py               # FactorEngine + 截面中性化 + 行业中性化
│   ├── alpha101.py             # 30 个 Alpha101 核心因子
│   ├── alpha191_turnover.py    # Alpha191 换手率类 5 因子
│   ├── alpha191_intraday.py    # Alpha191 日内形态类 4 因子
│   ├── alpha191_flow.py        # Alpha191 资金流向类 6 因子
│   ├── alpha191_gap.py         # Alpha191 隔夜效应类 4 因子
│   ├── alpha191_vol.py         # Alpha191 波动率高阶类 5 因子
│   ├── alpha191_liquidity.py   # Alpha191 流动性高阶类 4 因子
│   ├── custom.py               # 7 个自定义 A 股因子
│   ├── fundamental.py          # 11 个基本面质量因子（九项排雷）
│   ├── intraday_minute.py      # 7 个分钟频率日内因子（60min K线聚合）
│   ├── monitor.py              # IC/ICIR/衰减曲线监控
│   └── screening.py            # 正交性筛选（Spearman + 贪心）
│
├── models/                     # ML 预测层
│   ├── __init__.py             # 模块导出
│   ├── dataset.py              # 因子数据集构造 + walk-forward 切分 + 多周期标签
│   ├── trainer.py              # XGBoost/LightGBM 训练器 + EnsemblePredictor + MultiHorizonEnsemble
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
├── web/                      # Web 界面层 (FastAPI + HTMX + Alpine.js + ECharts)
│   ├── main.py               # FastAPI 入口 + 路由注册 + 静态文件
│   ├── routes/
│   │   ├── api.py            # 主 API 路由（行情/K线/回测详情/数据状态/模拟盘）
│   │   ├── dashboard.py      # 行情看板路由
│   │   ├── backtest.py       # 回测对比路由
│   │   ├── paper.py          # 模拟盘路由
│   │   ├── data_status.py    # 数据状态监控路由
│   │   └── factors.py        # 因子监控路由
│   ├── static/
│   │   └── app.js            # 前端 JS（ECharts 图表管理）
│   └── templates/
│       ├── base.html         # 基础布局（导航栏 + HTMX + Alpine.js + ECharts CDN）
│       ├── dashboard.html    # 行情看板（自选列表 + K线图）
│       ├── backtest.html     # 回测对比（列表 + 详情面板 + 双线权益曲线）
│       ├── paper.html        # 模拟盘管理
│       ├── data.html         # 数据状态监控
│       └── factors.html      # 因子就绪状态 + 血缘追踪
│
├── scripts/                  # 批量工具（16个脚本）
│   ├── run_all_backtests.sh      # 一键运行全部 6 个策略（1静态+5ML，含动态反馈闭环）
│   ├── run_static_backtest.py    # 静态策略回测（backtrader Cerebro+等权聚合）
│   ├── run_ml_backtest.py        # ML 选股端到端评估（因子预设+算法选择+Optuna调优）
│   ├── run_simulation.py         # ML 每日模拟回测（选股→分配→止损→P&L）
│   ├── batch_backtest.py         # 全量回测：所有股票 × 策略 × 参数 × 牛熊市
│   ├── grid_backtest.py          # 全市场回测 + 参数网格搜索 + 过拟合检测
│   ├── sync_minute_data.py       # 60 分钟 K 线批量同步（Sina 直连，超时+续传）
│   ├── compare_freq.py           # 日频 vs 分钟频 ML 回测对比
│   ├── validate_minute_data.py   # 分钟线日收益 vs 日线收益一致性校验
│   ├── verify_paper_trading.py   # 端到端验证（simulation vs PaperEngine 对比）
│   ├── overfit_check.py          # 过拟合检测（样本内 vs 样本外 Sharpe 对比）
│   ├── attribution.py            # 收益归因分析（因子/行业/风格归因）
│   ├── auto_adjust.py            # 自动调参（基于归因结果调整模型参数）
│   ├── health_check.py           # 系统健康检查（DB/数据/因子/模型全链路）
│   ├── command_worker.py         # 后台命令执行 worker
│   └── run_all_backtests.py      # [已废弃] 旧版批量回测（依赖已移除的app.utils）
│
├── docs/                     # 文档 & 研究报告
│   ├── quant-strategy-research.md  # 量化策略研究报告（技术指标+学术前沿）
│   ├── strategy-interpretability.md # ML策略可解释性分析（因子+筛选+回测验证）
│   ├── postgresql-guide.md         # PostgreSQL 使用指南
│   ├── project-learning-guide.md   # 项目学习指南
│   └── factor-table.md             # 因子分类表
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

### 2. 数据质量校验

```
data/quality.py  ─── 质量门禁
  │
  ├── 覆盖度检查（每只股票最后交易日是否在 N 天内）
  ├── 缺失值检测（OHLCV 字段空值率）
  ├── 异常值检测（涨跌幅 > 11% 或 OHLC 逻辑矛盾）
  ├── 交易日对齐（与交易日历对比，标记缺失交易日）
  └── 质量评分 → 质量报告 → 阻断不达标的因子计算

data/lineage.py  ─── 数据血缘追踪
  │
  ├── 记录每条数据的来源（API/文件/手动）
  ├── 记录转换步骤（复权/合并/聚合）
  └── 追溯消费方（因子/模型/回测）

data/availability.py  ─── 数据可用性
  │
  ├── 交易日历（沪深交易所）
  └── 实时数据可用性查询（某股票在某日期是否有数据）
```

## 回测结果（2026-05-28, v1.12）

> v1.12 管线：分钟K线(60min) → 日内特征聚合(7个) → 行业截面中性化 → 多周期标签(T+1/5/20) → 集成预测。全市场选股 → ML打分 → 排雷过滤 → NDrop调仓 → 等权持有 + 组合风控。
> 统一参数：100 万本金、佣金万 0.9（买卖双向）、印花税万 5（卖出单向）、滑点 0.1%。
> ```bash
> bash scripts/run_all_backtests.sh          # 一键运行全部 6 个策略
> ```

### v1.12 新增功能（2026-05-28）

**v1.12 vs Baseline 对比**（500只候选池, 52因子, 2024-03~2026-04）：

| 指标 | Baseline | v1.12 | 改善 |
|------|:---:|:---:|:---:|
| 最终活跃因子 | 13 | 19 | +6 |
| **总收益率** | 100.81% | **140.25%** | +39.44pp |
| **年化收益率** | 86.87% | **119.47%** | +32.60pp |
| **Sharpe** | 2.12 | **3.35** | +1.23 |
| **最大回撤** | 27.37% | **12.52%** | -14.85pp |

**三项新功能**：
- **分钟频率因子（7个）**：从 `stock_minute` 60min K线聚合日频特征，通过 `factors/intraday_minute.py` → `extra_data` 注入因子引擎
- **行业截面中性化**：`factors/engine.py::neutralize_by_industry()`，每个因子减去同行业同日截面均值（19个SW1行业，~2,900只覆盖）
- **多周期预测**：`models/trainer.py::MultiHorizonEnsemble`，T+1/T+5/T+20 三对模型加权打分（默认 0.5/0.3/0.2）

**命令行**：
```bash
# 全功能回测（新默认：无需额外参数）
python scripts/run_ml_backtest.py
# 关闭某个功能
python scripts/run_ml_backtest.py --no-multi-horizon --forward-days 5
# 今日预测
python scripts/predict_today.py --top-n 15
```

### v1.12 参数优化：日频网格搜索（2026-05-29）

> 由于分钟数据覆盖率不足（1,472/4,911），对 v1.12 日频管线做 6 维网格搜索，寻找无分钟因子的最优参数组合。
> 固定：`+momentum+reversal+volatility+liquidity+fundamental` 因子集（~66个）、动态多因子闭环、300只候选池。

**搜索空间**（72 组排列组合）：

| 参数 | 候选值 |
|------|--------|
| 多周期集成 | ON (T+1/5/20) / OFF |
| 市场状态分治 | ON / OFF |
| 行业截面中性化 | ON / OFF |
| 持仓数 (top-n) | 10 / 15 / 20 |
| 调仓频率 | 每日 / 周度(5日) |

**Top 10 结果**（回测区间 2022-01~2026-05, CAGR年化, Score=CAGR-2×|DD|+0.5×Sharpe）：

| Rank | 多周期 | Regime | 中性化 | Top-N | 调仓 | CAGR | Sharpe | MaxDD | Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | — | ON | — | 10 | 日 | 169.5% | 2.99 | 23.0% | 138.4 |
| 2 | — | ON | — | 15 | 日 | 134.3% | 2.74 | 23.8% | 100.4 |
| **3** | **ON** | — | **ON** | **15** | **周** | **110.3%** | **2.99** | **16.0%** | **93.3** |
| 4 | — | ON | — | 20 | 日 | 117.9% | 2.65 | 22.1% | 87.0 |
| 5 | ON | — | — | 20 | 周 | 99.3% | 3.16 | 14.8% | 85.6 |
| 6 | — | — | — | 20 | 周 | 103.8% | 2.61 | 19.2% | 78.5 |
| 7 | — | — | ON | 10 | 周 | 103.2% | 2.34 | 18.4% | 78.0 |
| 8 | ON | — | — | 15 | 日 | 81.7% | 2.94 | 10.1% | 76.3 |
| 9 | — | ON | ON | 20 | 日 | 101.7% | 2.44 | 18.9% | 76.0 |
| 10 | ON | — | ON | 20 | 周 | 96.0% | 2.73 | 18.5% | 72.6 |

**关键发现**：
- **Multi-Horizon 更稳**：MH 组合回撤普遍 15-20%，远优于单周期 23-30%
- **Regime 对 MH 无效**：`--multi-horizon` 代码路径优先于 `--regime`，MH 模式下分状态训练被跳过
- **Industry-neutralize 双刃剑**：MH 下开中性化→CAGR↑/DD↑，关→CAGR↓/DD↓
- **top-n=15 最优**：10 太集中，20 摊薄超额收益
- **T+5+日频** 年化虚高：外样本仅 ~1.4 年，CAGR 放大效应；MH 相对可靠
- **第 1 名虽然 Score 最高但第 3 名才是最优实际选择**：综合收益、回撤、稳健性后，**第 3 名（MH+中性化+15只+周度）**被选为新默认 CLI 参数

**脚本**：`scripts/grid_search.py`，结果文件 `output/grid_results.csv`。

### v1.10 改进
- **全市场选股**：修复 `ORDER BY code LIMIT 200` 仅选深市股偏差，默认5238只非ST
- **排雷过滤**：8项质量检查（调整后净利润/负债率/商誉/质押/现金流/ROE/净利率），允许≤3项违规
- **NDrop 增量调仓**：每次最多替换2只最差持仓，换手率~4%（vs 旧版~16%）
- **组合风控**：-20%减半仓 / -25%清仓(10天冷静期) / 指数15日跌超12%空仓

### ML 策略（5个，周度调仓 + A股真实成本 + 200只候选池）

> 统一参数：100 万本金、佣金万 0.9（买卖双向）、印花税万 5（卖出单向）、滑点 0.1%。

| 策略 | 因子池 | 模型 | 标签 | 年化收益 | Sharpe | 最大回撤 | 日均换手 | 年化成本 |
|---|---|---|---|---|---|---|---|---|
| ML-默认集成 | 动量+反转 | XGB+LGBM集成 | ret_1d | 39.25% | 1.27 | -36.03% | 1.7% | 1.6% |
| ML-动量精选 | 动量/趋势(17个) | XGBoost | ret_5d | — | — | — | — | — |
| ML-反转精选 | 反转(20个) | LightGBM | ret_1d | 40.00% | 1.23 | -41.09% | — | — |
| ML-全量因子测试 | 全部76因子 | XGB+LGBM集成 | ret_1d | 35.26% | 1.17 | -36.94% | — | — |
| ML-动态多因子 | 全部76因子 | XGB+LGBM集成 | ret_1d | 35.56% | 1.22 | -36.32% | — | — |

> **ML-动量精选**：17 个动量因子对 5 日标签(ret_5d)的 IC 均低于 0.02 门禁（|IC| 0.003~0.017），IC 门禁后无可用因子，0 个有效 walk-forward 窗口。
> 
> 回测区间：2020-01-01 ~ 2026-05-01，200 只候选池，周度调仓。

### 静态策略（1个，backtrader Cerebro + 等权聚合）

| 策略 | 年化收益 | Sharpe | 最大回撤 | 参数 |
|---|---|---|---|---|
| 震荡网格(高抛低吸) | 4.23% | 0.58 | -16.36% | size=500, buy_step=0.02, ma_period=30 |

**防过拟合检查**：`scripts/overfit_check.py`，检查项：样本外一致性、交易次数、市场状态覆盖、参数敏感性

---

## 分钟数据（已集成到主选股管线）

> v1.12 已将分钟因子通过 `extra_data` 注入主 ML 管线，不再需要单独的频率对比回测。

**当前状态**（2026-05-27）：已同步 **1,309 / 4,909** 只股票（26.7%），区间 2024-03 ~ 2026-05-27。新浪 API 有 ~75 次/IP 封堵，需分批冷却。

**同步命令**：
```bash
python scripts/sync_minute_data.py --limit 50                    # 测试
python scripts/sync_minute_data.py --skip 1309                   # 续传
python scripts/sync_minute_data.py --batch-size 70 --cooldown 2100  # 每70只冷却35分钟
```

**数据**：新浪财经 `money.finance.sina.com.cn` 直连 API（akshare 的 Sina 端点返回 HTTP 456 已废弃）。

**因子聚合**：分钟 K 线因子按 (code, trade_date) groupby last，标签仍为日频 ret_1d，bar_per_day=4。

### 前期对比结果（1,004 只股票，2026-05-26）

| 指标 | 日频(daily) | 分钟频(60min) | 差异 |
|------|:---:|:---:|:---:|
| 年化收益率 | 67.00% | 431.28% | +364pp |
| 总收益率 | 68.36% | 445.55% | +377pp |
| 最大回撤 | -11.9% | -10.2% | +1.7pp |
| 夏普比率 | 2.76 | 5.78 | +3.02 |
| 胜率 | 51.77% | 58.11% | +6.34pp |
| 交易次数 | 1030 | 1051 | +21 |
| 可用因子数 | 21 | 15 | -6 |
| 耗时 | 35s | 36s | +1s |

> **注意**：分钟频复权方式为前复权(qfq)，日频为后复权(hfq)。日收益理论上不受复权方式影响，但绝对价格差异可能影响仓位计算粒度。431% 年化偏高，待进一步排查是否有前瞻偏差或复权影响。

**同步命令**：
```bash
python scripts/sync_minute_data.py --limit 50    # 测试
python scripts/sync_minute_data.py --skip 1381    # 续传（跳过前 N 只）
```

**数据校验**：
```bash
python scripts/validate_minute_data.py    # 对比分钟聚合日收益 vs stock_daily
```

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

### 3. Web 界面数据流 (FastAPI + HTMX + ECharts)

```
web/main.py (FastAPI 入口)
  │
  ├── 路由注册: routes/api.py (主API) + dashboard/backtest/paper/data_status/factors
  │
  └── 页面导航
        │
        ├── 行情看板 (/dashboard)
        │   ├── GET /api/quotes/{group} → stock_daily JOIN stock_basic (代码+名称+现价)
        │   └── GET /api/kline/{code} → ECharts K线图 + 成交量柱 (dataZoom缩放, 近1年默认)
        │
        ├── 回测对比 (/backtest)
        │   ├── GET /api/backtest-list → 8策略列表 (年化收益/最大回撤/Sharpe/质量)
        │   └── GET /api/backtest-detail/{id} → 双线权益曲线(策略+上证指数) + 完整指标表 + 因子构成
        │
        ├── 模拟盘 (/paper)
        │   └── GET /api/paper-runs → 模拟盘运行列表 + 持仓/信号详情
        │
        ├── 数据状态 (/data)
        │   ├── GET /api/data-status → 31张表最新日期+行数
        │   └── POST /api/sync/trigger → 触发后台数据同步
        │
        └── 因子监控 (/factors)
            ├── GET /api/factor-overview → 因子就绪状态 + 交易日/因子数/最早就绪
            └── 因子血缘: 因子名 → 上游字段 → 上次校验时间
```

**前后端交互模式**：HTMX 属性触发 AJAX 请求 (`hx-get/hx-post/hx-trigger`)，服务端返回 HTML 片段直接替换 DOM。ECharts 图表通过 API 响应中的内联 `<script>` 标签初始化，避免事件监听器的时机问题。

---

## 数据库表结构（31 张表）

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

### 模拟盘 & 回测 & 策略配置（6 张）

| 表 | 说明 |
|---|---|
| `paper_account` | 模拟账户（名称、初始资金、现金、策略类型/名称/参数、费率、大盘过滤） |
| `paper_orders` | 委托记录（代码、方向、价格、数量、状态） |
| `paper_positions` | 当前持仓（代码、数量、均价） |
| `paper_daily_pnl` | 每日净值（现金、持仓市值、总资产、日收益、回撤） |
| `backtest_results` | 回测结果持久化（策略参数、资产模式、股票池、指标汇总、完整结果 JSON） |
| `ml_strategy_config` | ML 策略配置持久化（因子/训练/组合/风控参数，支持 UI 编辑和版本管理） |

> ETF/基金相关 4 张表（etf_basic / etf_daily / fund_basic / fund_nav）DDL 保留但未启用。fetcher.py 和 sync.py 中对应函数代码完整保留但调用链已注释。

---

## 策略系统

### 架构

```
strategies/__init__.py
  │
  ├── STRATEGY_REGISTRY（内置策略）
  │   └── "震荡网格(高抛低吸)" → GridShockStrategy
  │
  └── get_all_strategies() → 内置 + ~/.quant_strategies/ 自定义策略
        │
        └── 回测页面 STRATEGY_REGISTRY 动态策略列表
```

### 策略详情

详见 [strategies/README.md](strategies/README.md)

| 策略 | 信号 | 默认参数 | 适用场景 |
|---|---|---|---|
| 震荡网格(高抛低吸) | 价格触及网格下沿买入、上沿卖出 | (20, 0.05) | 震荡市 |

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

> 批量回测结果请查看 `output/batch_summary.md`。

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

# 启动 Web 界面 (FastAPI + HTMX)
uvicorn web.main:app --host 0.0.0.0 --port 8000 --reload
# 浏览器打开 http://localhost:8000

# 运行全部策略回测
bash scripts/run_all_backtests.sh

# 单独运行某个 ML 策略
python scripts/run_ml_backtest.py --strategy "ML-动量精选" \
    --factor-preset momentum --forward-days 5 --model xgboost \
    --no-ensemble --optuna --optuna-trials 20

# 单独运行某个静态策略
python scripts/run_static_backtest.py --strategy "震荡网格(高抛低吸)" --top-n 30
```

---

## v2.0 架构说明

v2.0 完成了从 Streamlit 单体应用到 FastAPI + HTMX 分层架构的迁移。Web 端使用服务端渲染 HTML 片段 + ECharts 内联脚本初始化图表。

**核心变更**：

| 维度 | v1.0 (Streamlit) | v2.0 (FastAPI + HTMX) |
|------|------------------|----------------------|
| Web 框架 | Streamlit (`app/`) | FastAPI (`web/`) |
| 前端 | Streamlit 自动渲染 | HTMX + Alpine.js + ECharts |
| 图表 | Plotly | ECharts (内联 script 初始化) |
| 目录 | `app/pages/`, `app/utils/` | `web/routes/`, `web/templates/`, `web/static/` |
| 数据库表 | 16 张 | 31 张（新增策略管理、信号归因、因子权重等） |
| 回测运行器 | `backtest_runner.py` | `run_static_backtest.py` + `run_ml_backtest.py` |
| ML 策略差异化 | 全部使用相同因子池 | 因子预设 + 算法选择 + 标签周期 + Optuna |

**工作流**：
```
因子计算(定时) → 模型训练 → 历史回测(防过拟合检查) → 模拟盘验证 → 归因反馈闭环
```

---

## 变更记录

| 日期 | 变更内容 |
|---|---|---|

| 2026-05-30 | **v1.3 舞策略上线**：5状态市场检测(强牛/弱牛/快熊/慢熊/震荡) + 按状态自适应调仓频率/仓位/止损。新增3个纯价格动量因子。模拟盘系统V2(paper_runs/signals/positions)打通,`run_daily_paper.py`每日驱动。回测CAGR 44.6%/Sharpe 1.34/MaxDD 25.8%。Web UI: 权益曲线+持仓明细+市场状态+版本号。 |
| 2026-05-29 | **v1.12 参数优化：日频网格搜索**：分钟数据覆盖率不足，对日频管线做 6 维 72 组参数网格搜索（多周期/Regime/行业中性化/持仓数/调仓频率）。结论：MH+行业中性化+15只+周度调仓为最优均衡组合（CAGR 110.3%、Sharpe 2.99、MaxDD 16.0%）。`scripts/run_ml_backtest.py` 默认参数更新：`--multi-horizon`/`--industry-neutralize`/`--dynamic` 默认开启，`--top-n` 默认 15，增加 `--no-*` 关闭选项。新增 `scripts/grid_search.py` 通用网格搜索工具。 |
| 2026-05-28 | **v1.12 分钟因子+行业中性化+多周期预测**：新增 `factors/intraday_minute.py`（7个60min K线日内因子，因子总数 76→83）；`factors/engine.py` 新增 `neutralize_by_industry()` 行业截面中性化（19个SW1行业）；`models/dataset.py` 支持多周期标签 `forward_days=[1,5,20]` 和行业中性化开关；`models/trainer.py` 新增 `MultiHorizonEnsemble` 加权复合打分和 `walk_forward_train_multihorizon`；`scripts/run_ml_backtest.py` 新增 `--multi-horizon`/`--industry-neutralize`/`--horizon-weights` CLI 参数、分钟/行业数据加载；`scripts/predict_today.py` 同步更新。回测验证：500只候选池/52因子，年化收益 86.87%→119.47%（+32.6pp），Sharpe 2.12→3.35（+1.23），最大回撤 27.37%→12.52%（-14.85pp）。修复 `MultiHorizonEnsemble.predict()` 索引对齐 bug（composite 用 code 字符串索引而非整数位置）。 |
| 2026-05-28 | **v1.11 动态反馈闭环**：新增 ML-动态多因子策略，两级联动——窗口级因子淘汰/发现（IC衰减→权重衰减→淘汰<0.3）和日度级信号追踪（滚动IC→连续衰减告警→触发重训）。新增 `strategy_health`/`strategy_commands` 表。全市场选股+组合风控，Sharpe 1.30。 |
| 2026-05-27 | **交易成本修正 + 策略精简**：ML 回测新增 A 股真实交易成本（佣金万 0.9 + 印花税万 5 + 滑点 0.1%，基于每日换手率扣除）。静态回测确认 100 万本金 + 万 0.9 佣金 + 万 5 印花税 + 0.1% 滑点。删除 RSI/MACD/双均线三种零收益静态策略，策略总数从 9 降至 6（1 静态 + 5 ML）。README 全面更新。 |
| 2026-05-27 | **动态反馈闭环 + 日度信号追踪**：新增 ML-动态多因子策略（`--dynamic` 标志），两级联动。窗口级 `BacktestFeedbackLoop`：因子重要性归因 → Sharpe 趋势/t 检验衰退 → 衰减因子×0.8 → 权重<0.3 淘汰。日度级 `DailySignalTracker`：每日 Rank IC → 滚动 20 日 IC → 连续 5 日衰减警告 → 连续 10 日触发重训。数据写入 `strategy_health` + `strategy_commands`。6 策略全 valid，年化 7.69%~28.70%。 |
| 2026-05-27 | **Web 增强 + ML 策略差异化 + 分钟数据同步优化**：行情看板显示股票名称（JOIN stock_basic）；K线图新增 dataZoom 缩放（默认展示近1年，可拖动扩展）；回测列表新增年化收益/最大回撤/Sharpe 列。ML 策略全面差异化：因子预设系统（momentum/reversal/volatility/liquidity/fundamental 五组+联合预设）、`--forward-days` 标签周期切换、单模型/集成可选、Optuna 贝叶斯超参优化，4个 ML 策略年化 19.10%~31.82%。新增 `run_static_backtest.py`（backtrader Cerebro + 等权聚合）和 `run_all_backtests.sh`。`sync_minute_data.py` 新增 `--batch-size`/`--cooldown` 分批冷却机制适配新浪 ~75 次封 IP 限制，`--skip` 改为在 missing-only 过滤后计数，新增 `missing_only` 参数只同步无分钟数据的股票。分钟数据已覆盖 1,309/4,909 只。修复单模型 equity curve、regime_count 默认值、dataset 多周期收益列等 bug。全部 6 策略 quality=valid。 |
| 2026-05-26 | **分钟频 ML 回测 + API 换源**：`data/fetcher.py` 分钟数据从 akshare(`quotes.sina.cn`)切换到新浪直连 API(`money.finance.sina.com.cn`)，旧 JSONP 端点被 Sina 封锁(HTTP 456)。1004 只股票分钟数据（000/001/002/300/600 五个板块），日频 vs 60 分钟频对比回测完成（分钟频胜率+6%、夏普翻倍）。新增 `scripts/sync_minute_data.py`(批量同步)、`scripts/compare_freq.py`(频率对比)、`scripts/validate_minute_data.py`(数据校验)。`ml_backtest.py` 支持 `freq="daily"|"60min"`，因子引擎支持 `bar_per_day`。新增 `GridShockStrategy`(震荡网格)。数据库 15→16 张表(新增 `ml_strategy_config`)。`ml_config_manager.py` 实现 ML 策略配置 CRUD+内置预设。回测页新增日期区间筛选+权益曲线 x 轴限制 |
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
