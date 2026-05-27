# Quant 项目学习指南

> 帮助你快速消化理解整个项目的架构、数据流和关键设计决策。
> 最后更新：2026-05-28 (v1.10)

---

## 一、项目是什么？

一个 **A 股量化交易系统**，核心是一条 ML 选股管线：每天用机器学习模型对全市场股票打分排序，经过排雷过滤和 NDrop 增量调仓，选出最可能上涨的 Top-N 只股票，配合组合级风控（回撤减仓/清仓/指数大跌空仓），模拟实盘买卖。

当前版本：**v1.10**

技术栈：Python + PostgreSQL + FastAPI + HTMX + AKShare + backtrader + XGBoost/LightGBM

---

## 二、一张图看懂整个系统

```
┌──────────────────────────────────────────────────────────────┐
│                     数据来源（AKShare）                         │
│  腾讯财经(行情) / 同花顺(财务) / 东方财富(质押/行业) / 百度(估值) / 新浪(分钟)│
└──────────────────────────┬───────────────────────────────────┘
                           │ data/fetcher.py (获取)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   PostgreSQL 数据库（16 张表）                  │
│  行情: stock_daily / stock_minute / index_daily / stock_tick     │
│  基本面: stock_basic / stock_daily_extra / stock_financial       │
│          stock_shareholder / stock_pledge / stock_industry        │
│  模拟盘: paper_account / paper_orders / paper_positions           │
│          paper_daily_pnl / backtest_results                       │
│  策略配置: ml_strategy_config                                     │
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
              │  FastAPI + HTMX Web UI │
              │  5 个功能模块          │
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
| [data/db.py](../data/db.py) | 16 张表的 DDL，理解表结构就是理解数据模型 |
| [data/fetcher.py](../data/fetcher.py) | `fetch_stock_daily()`(日线/AKShare) + `fetch_minute_data()`(分钟/新浪直连) |
| [data/sync.py](../data/sync.py) | `sync_stock_daily()` 看增量同步逻辑，`main()` 看 9 种同步模式 |

**关键理解**：数据格式是统一的后复权 OHLCV（日线）、前复权（分钟线），代码格式是纯数字（如 `000001`），不带 SH/SZ 前缀。分钟线使用新浪财经 `money.finance.sina.com.cn` 直连 API（旧 akshare `quotes.sina.cn` JSONP 端点已被 Sina 封锁返回 HTTP 456）。

### 第三站：因子层（20分钟）

| 文件 | 看什么 |
|------|-------|
| [factors/__init__.py](../factors/__init__.py) | `ALL_FACTORS` 字典——76 个因子的完整清单和参数 |
| [factors/engine.py](../factors/engine.py) | 因子计算引擎 + 截面中性化（市值/行业） |
| [factors/screening.py](../factors/screening.py) | `filter_factors_by_ic()` IC 门禁 + `select_orthogonal_factors()` 正交筛选 |
| [factors/fundamental.py](../factors/fundamental.py) | 11 个基本面质量因子（九项排雷体系） |

**关键理解**：因子 = 对股票某个维度的量化描述。IC（Information Coefficient）= 因子值与未来收益的相关性。筛选链：76 因子 → IC 门禁 → ~20 因子 → 正交去冗余 → ~8-16 因子 → 模型。

**v1.10 完整选股管线**：全市场非ST股票 → 因子计算 → ML 打分排序 → 排雷过滤（8项检查，允许≤3项违规）→ NDrop 增量调仓（K=15, N=2）→ 等权持有

**v1.10 风控管线**：每日检查 → 个股-8%硬止损 → 组合回撤-20%减半仓 → 组合回撤-25%清仓（10天冷静期+重置peak）→ 指数15日跌超12%空仓

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
| `scripts/run_ml_backtest.py` | ML 端到端回测（因子→筛选→训练→汇总） |
| `scripts/overfit_check.py` | 过拟合检测（样本内 vs 样本外 Sharpe 对比） |

### 第七站：Web 页面（按需阅读）

| 模块 | 做什么 |
|------|-------|
| 行情看板 | 自选股实时报价 + K线图 |
| 回测对比 | 回测结果历史对比 + 导出 |
| 模拟盘 | 模拟盘账户/持仓/委托管理 |
| 数据状态 | 数据覆盖度/质量/同步监控 |
| 因子监控 | 因子 IC 看板 + 模型表现 + 当日信号 |

### 第八站：后台研究管线（新增 v2.0）

v2.0 引入完整的后台研究管线，以脚本+定时任务的方式运行：

```
数据同步(sync.py)
    │
    ▼
质量校验(quality.py) ── 覆盖度/缺失值/异常值/交易日对齐
    │
    ▼
因子计算(factors/engine.py) ── 76个因子 + 截面中性化
    │
    ▼
模型训练(models/trainer.py) ── Walk-forward + XGBoost+LightGBM 集成
    │
    ▼
历史回测(run_ml_backtest.py) ── 过拟合检测(overfit_check.py)
    │
    ▼
模拟盘验证(paper_engine.py) ── 日频信号→订单→DB写入
    │
    ▼
归因分析(attribution.py) ── 因子/行业/风格收益归因
    │
    ▼
自动调参(auto_adjust.py) ── 基于归因结果调整模型参数
```

| 脚本 | 做什么 |
|------|-------|
| `scripts/overfit_check.py` | 样本内 vs 样本外 Sharpe 衰减比检测 |
| `scripts/attribution.py` | 收益归因分析（因子/行业/风格维度） |
| `scripts/auto_adjust.py` | 基于归因反馈自动调参 |
| `scripts/health_check.py` | 系统全链路健康检查 |
| `scripts/command_worker.py` | 后台命令执行 worker |

### 第九站：批量脚本（按需运行）

| 脚本 | 做什么 |
|------|-------|
| `scripts/run_simulation.py` | ML 选股每日模拟回测（~15min），含逐笔交易记录 |
| `scripts/run_ml_backtest.py` | ML 端到端评估（因子→筛选→训练→汇总） |
| `scripts/grid_backtest.py` | 参数网格搜索 + 过拟合检测（~30min） |
| `scripts/batch_backtest.py` | 静态策略全量回测（所有股票 × 策略 × 参数） |
| `scripts/verify_paper_trading.py` | simulation vs PaperEngine 一致性验证 |
| `scripts/sync_minute_data.py` | 60 分钟 K 线批量同步（超时+续传，~200 只/次） |
| `scripts/compare_freq.py` | 日频 vs 分钟频 ML 回测对比 |
| `scripts/validate_minute_data.py` | 分钟聚合日收益 vs stock_daily 一致性校验 |

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
| 股票分钟线 | 按需 | `python scripts/sync_minute_data.py --limit 200` |
| 指数日线 | 每日 | `python -m data.sync --mode index` |
| 估值指标 | 每日 | `python -m data.sync --mode daily-extra` |
| 财务数据 | 季度 | `python -m data.sync --mode financial` |
| 质押数据 | 按需 | `python -m data.sync --mode pledge` |
| 行业分类 | 按需 | `python -m data.sync --mode industry` |

**自动同步**：通过 macOS launchd plist (`config/com.quant.sync.plist`) 配置定时任务，每日收盘后自动运行 stock-daily 同步。也可通过 `scripts/command_worker.py` 手动触发。
**分钟线同步**：新浪 API 限制约 75 次连续请求后封 IP 30 分钟，建议分批 200 只/次，间隔 40 分钟。

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

# 3. 过拟合检测
python scripts/overfit_check.py

# 4. 收益归因分析
python scripts/attribution.py

# 5. 自动调参（基于归因结果）
python scripts/auto_adjust.py

# 6. 系统健康检查
python scripts/health_check.py

# 7. 参数优化（~30-60分钟，周末跑）
python scripts/grid_backtest.py --codes-file scripts/test_50_codes.txt

# 8. 全市场回测（~2-3小时，周末跑）
python scripts/grid_backtest.py
```

**建议节奏**：
- **每日收盘后**：跑数据同步（stock-daily + index + daily-extra）
- **每日**：跑 `run_simulation.py` 看当日模拟结果 + `health_check.py` 系统巡检
- **每周**：跑 `overfit_check.py` + `attribution.py` 归因分析
- **每周末**：跑 `grid_backtest.py` 做参数优化和过拟合检测，根据归因结果运行 `auto_adjust.py`
- **每月**：跑 `financial` 和 `pledge` 同步更新基本面数据

---

## 七、版本历史

### v1.10 (2026-05-28) — 全市场选股 + 组合风控

**重大修复**：股票池 `ORDER BY code LIMIT 200` 导致只选深市(0xxxxx)股票，VARCHAR字母序将沪市(6xxxxx)排在后面永不被选中。
修复为全市场非ST选股（默认5238只），可选 `--universe-size N` 按成交额取前N只。

**新增功能**：
- 排雷过滤：加载 stock_financial + stock_pledge，对8项质量检查允许≤3项违规
- NDrop 增量调仓：`select_topk_ndrop()` 每次最多替换2只最差持仓，换手率从~16%降至~4%
- 组合回撤风控：-20%减半仓 / -25%清仓(10天冷静期+peak_nav重置) / 指数15日跌超12%空仓
- 指数数据无条件加载（不再依赖 `--regime` 开关）
- 止损/风控事件记录到 backtest_results.metrics_json

**策略版本**：全部6个策略统一升级至 v1.10

### v1.00 (2026-05) — 基础ML选股管线

初始版本：因子计算 → IC门禁 → 正交筛选 → Walk-forward训练 → Top-N选股 → 等权持有。
