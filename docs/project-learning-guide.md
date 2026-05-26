# Quant 项目学习指南

> 帮助你快速消化理解整个项目的架构、数据流和关键设计决策。

---

## 一、项目是什么？

一个 **A 股量化交易系统**，核心是一条 ML 选股管线：每天用机器学习模型对全市场股票打分排序，选出最可能上涨的 Top-N 只股票，模拟实盘买卖。

技术栈：Python + PostgreSQL + Streamlit + AKShare + backtrader + XGBoost/LightGBM

---

## 二、一张图看懂整个系统

```
┌──────────────────────────────────────────────────────────────┐
│                     数据来源（AKShare）                         │
│  腾讯财经(行情) / 同花顺(财务) / 东方财富(质押/行业) / 百度(估值)    │
└──────────────────────────┬───────────────────────────────────┘
                           │ data/fetcher.py (获取)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   PostgreSQL 数据库（14 张表）                  │
│  行情: stock_daily / index_daily / stock_tick / stock_minute  │
│  基本面: stock_basic / stock_daily_extra / stock_financial    │
│          stock_shareholder / stock_pledge / stock_industry     │
│  模拟盘: paper_account / paper_orders / paper_positions        │
│          paper_daily_pnl                                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                 ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │ factors/     │ │ models/      │ │ portfolio/   │
   │ 76个因子     │ │ ML训练+预测   │ │ 选股+风控     │
   │ IC筛选+正交  │ │ XGB+LightGBM │ │ NDrop调仓     │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
          │                │                 │
          └────────────────┼─────────────────┘
                           │
                           ▼
              ┌───────────────────────┐
              │  Streamlit Web UI     │
              │  8 个功能页面          │
              └───────────────────────┘
```

---

## 三、代码阅读顺序（推荐）

按理解难度和依赖关系，从底层到上层：

### 第一站：基础配置（5分钟）

| 文件 | 看什么 |
|------|-------|
| [config/settings.py](../config/settings.py) | 数据库连接、指数代码列表 |

### 第二站：数据层（15分钟）

| 文件 | 看什么 |
|------|-------|
| [data/db.py](../data/db.py) | 14 张表的 DDL，理解表结构就是理解数据模型 |
| [data/fetcher.py](../data/fetcher.py) | `fetch_stock_daily()` 是最核心的接口，看它怎么从 AKShare 拿数据 |
| [data/sync.py](../data/sync.py) | `sync_stock_daily()` 看增量同步逻辑，`main()` 看 9 种同步模式 |

**关键理解**：数据格式是统一的后复权 OHLCV，代码格式是纯数字（如 `000001`），不带 SH/SZ 前缀。

### 第三站：因子层（20分钟）

| 文件 | 看什么 |
|------|-------|
| [factors/__init__.py](../factors/__init__.py) | `ALL_FACTORS` 字典——76 个因子的完整清单和参数 |
| [factors/engine.py](../factors/engine.py) | 因子计算引擎 + 截面中性化（市值/行业） |
| [factors/screening.py](../factors/screening.py) | `filter_factors_by_ic()` IC 门禁 + `select_orthogonal_factors()` 正交筛选 |
| [factors/fundamental.py](../factors/fundamental.py) | 11 个基本面质量因子（九项排雷体系） |

**关键理解**：因子 = 对股票某个维度的量化描述。IC（Information Coefficient）= 因子值与未来收益的相关性。筛选链：76 因子 → IC 门禁 → ~20 因子 → 正交去冗余 → ~8 因子 → 模型。

### 第四站：ML 模型层（20分钟）

| 文件 | 看什么 |
|------|-------|
| [models/dataset.py](../models/dataset.py) | `build_factor_dataset()` 构造训练数据 + walk-forward 切分 |
| [models/trainer.py](../models/trainer.py) | `walk_forward_train_ensemble()` 核心训练流程 + `EnsemblePredictor` 概率平均 |
| [models/dual_period.py](../models/dual_period.py) | 双周期模型：月频基本面排雷 + 日频量价 ML |
| [models/regime.py](../models/regime.py) | 市场状态识别（牛/熊/震荡） |

**关键理解**：Walk-forward = 滚动训练，每个时间窗口用 3 年历史训练 + 1 年验证，避免前视偏差。Ensemble = XGBoost + LightGBM 两个模型概率取平均，比单模型更稳健。

### 第五站：组合优化层（15分钟）

| 文件 | 看什么 |
|------|-------|
| [portfolio/selector.py](../portfolio/selector.py) | `select_top_n()` / `select_topk_ndrop()` 选股 + ST/停牌/涨跌停过滤 |
| [portfolio/allocator.py](../portfolio/allocator.py) | 等权分配 + 仓位上限（单只 10%、行业 30%） |
| [portfolio/risk.py](../portfolio/risk.py) | 止损（-8% + ATR）+ 组合回撤控制 + 指数崩盘过滤器 |
| [portfolio/paper_engine.py](../portfolio/paper_engine.py) | PaperEngine 日频模拟盘引擎——从信号到订单到 DB 写入的完整管线 |

**关键理解**：NDrop = 非Drop，每日持仓对比 Top-K 排序，只替换排名掉出前 K 的股票，每次最多替换 N 只。K=15、N=2 是最优参数。

### 第六站：回测引擎（10分钟）

| 文件 | 看什么 |
|------|-------|
| [app/utils/backtest_runner.py](../app/utils/backtest_runner.py) | `run_backtest()` 封装 backtrader + A 股费用 + 大盘过滤器 + FIFO 交易记录 |

### 第七站：Web 页面（按需阅读）

| 页面 | 做什么 |
|------|-------|
| [Watchlist](../app/pages/1_📈_Watchlist.py) | 自选股实时报价（akshare 5s 缓存） |
| [Charts](../app/pages/2_📊_Charts.py) | K 线图 + 技术指标 |
| [Backtest](../app/pages/3_🧪_Backtest.py) | 静态策略回测 + 股票池批量 |
| [Paper Trade](../app/pages/4_📋_Paper_Trade.py) | 模拟盘管理 + 已平仓交易 P&L |
| [ML Monitor](../app/pages/8_📊_ML_Monitor.py) | 因子 IC 看板 + 当日信号 + 模拟盘净值 + 最近交易 |

### 第八站：批量脚本（按需运行）

| 脚本 | 做什么 |
|------|-------|
| `scripts/run_simulation.py` | ML 选股每日模拟回测（~15min），含逐笔交易记录 |
| `scripts/run_ml_backtest.py` | ML 端到端评估（因子→筛选→训练→汇总） |
| `scripts/grid_backtest.py` | 参数网格搜索 + 过拟合检测（~30min） |
| `scripts/batch_backtest.py` | 静态策略全量回测（所有股票 × 策略 × 参数） |
| `scripts/verify_paper_trading.py` | simulation vs PaperEngine 一致性验证 |

---

## 四、关键设计决策（Why）

| 决策 | 原因 |
|------|------|
| **日频量价 + 月频基本面分离** | 量价因子日频变化快需要每日更新，基本面季频变化慢月频足够，解耦后问题可独立定位 |
| **IC 门禁 \|IC\|>0.02** | 过滤掉与未来收益无关的噪音因子（通过率 ~31%），减少模型过拟合 |
| **Spearman 正交筛选 < 0.7** | 去除高度相关的冗余因子，8 个独立信号源比 65 个相关因子更稳健 |
| **XGBoost + LightGBM 集成** | 两个树模型结构不同（逐层 vs 逐叶），概率平均降低单模型方差 |
| **NDrop 增量调仓** | 好信号不会一天消失，保留大部分持仓减少换手成本，同时保持对信号的响应 |
| **后复权统一** | 分红拆股不产生虚假价格跳跃，回测结果更准确 |
| **PostgreSQL 而非 CSV** | 5000+ 股票 × 10 年日线 = 1000 万+行，数据库索引查询秒级响应 |
| **AKShare 而非 tushare** | 开源免费、支持多个数据源（腾讯/同花顺/东方财富/百度/交易所）、pip 安装即用 |

---

## 五、数据同步策略

| 数据 | 频率 | 命令 |
|------|------|------|
| 股票日线 | 每日 | `python -m data.sync --mode stock-daily` |
| 指数日线 | 每日 | `python -m data.sync --mode index` |
| 估值指标 | 每日 | `python -m data.sync --mode daily-extra` |
| 财务数据 | 季度 | `python -m data.sync --mode financial` |
| 质押数据 | 按需 | `python -m data.sync --mode pledge` |
| 行业分类 | 按需 | `python -m data.sync --mode industry` |

**自动同步**：app/main.py 已配置每日 18:00 自动运行 stock-daily 同步。

---

## 六、策略优化后台任务

如果想用最新实盘数据持续优化策略，在后台运行：

```bash
# 1. 每日数据同步（关键）
python -m data.sync --mode index --start 20260101          # 指数（~5秒）
python -m data.sync --mode stock-daily --start 20260101 --workers 8  # 日线（~10分钟）
python -m data.sync --mode daily-extra --start 20260101 --workers 8  # 估值（~10分钟）

# 2. ML 端到端回测（~15-30分钟）
python scripts/run_simulation.py --top-n 15 --ndrop --ndrop-n 2 \
    --save-results /tmp/simulation_$(date +%Y%m%d).json

# 3. 参数优化（~30-60分钟，周末跑）
python scripts/grid_backtest.py --codes-file scripts/test_50_codes.txt

# 4. 全市场回测（~2-3小时，周末跑）
python scripts/grid_backtest.py
```

**建议节奏**：
- **每日收盘后**：跑数据同步（stock-daily + index + daily-extra）
- **每日**：跑 `run_simulation.py` 看当日模拟结果
- **每周末**：跑 `grid_backtest.py` 做参数优化和过拟合检测
- **每月**：跑 `financial` 和 `pledge` 同步更新基本面数据
