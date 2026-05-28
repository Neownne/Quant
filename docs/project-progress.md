# 项目进度文档

> 最后更新：2026-05-28

## 项目概述

量化多因子选股系统，A 股日频 + 分钟频数据处理、因子计算、ML 模型训练、回测、Web 展示、模拟交易。

## 技术栈

- **数据存储**：PostgreSQL（stock_daily, stock_minute, stock_industry 等 30+ 张表）
- **数据源**：Sina API（日频/分钟频）、AKShare（财务/股东）
- **因子引擎**：83 个已注册因子（momentum, reversal, volatility, liquidity, fundamental, intraday）
- **ML 框架**：XGBoost + LightGBM 集成 + MultiHorizonEnsemble（多周期加权），Walk-Forward 滚动训练
- **回测**：日频事件驱动回测，含交易成本（佣金+印花税+滑点）
- **Web**：FastAPI + ECharts 权益曲线 + Redis 缓存
- **模拟交易**：paper_* 表记录信号、持仓、订单、净值

## 策略演进

| 版本 | 日期 | 主要变化 | Sharpe | MaxDD | 收益 |
|------|------|----------|--------|-------|------|
| v1.5 | - | 初始 ML 策略，静态因子 | 1.27 | -36.0% | 186.1% |
| v1.7 | - | 因子筛选优化 | 1.23 | -41.1% | 191.0% |
| v1.8 | - | 加入排雷过滤 | 1.17 | -36.9% | 160.9% |
| v1.9 | - | NDrop 调仓 | 1.22 | -36.3% | 162.7% |
| v1.10 | 2026-05-28 | 全市场选股 + 组合风控 | - | - | - |
| v1.11 | 2026-05-28 | 动态反馈闭环 | 1.30 | -29.0% | 175.6% |
| **v1.12** | 2026-05-28 | 分钟因子+行业中性化+多周期 | **3.35** | **-12.52%** | **140.25%** |

## v1.11 核心架构

### 选股管线
```
全市场股票(5000+)
  → 因子计算(76个因子, 选22个活跃)
  → IC筛选(|IC|≥0.02) + 正交化(去重<0.7)
  → XGBoost+LightGBM 集成打分
  → 排雷过滤(8项检查, ≤3项违规通过)
  → ST过滤
  → NDrop增量调仓(K=30, N=2)
  → 等权持仓
```

### 风控管线
```
每日检查持仓
  → 个股-8%硬止损
  → 组合回撤-20% → 减半仓
  → 组合回撤-25% → 清仓(10天冷静期)
  → 指数15日跌超12% → 空仓
```

### 动态反馈闭环
- **因子淘汰**：|IC| < 0.05 且 排名后 50% → 淘汰（权重<0.3）
- **因子发现**：每 40 个调仓日扫描新因子，|IC| ≥ 0.02 → 加入
- **模型重训**：发现新因子或 IC 衰减超阈值时触发
- **阈值调优**：每个 Walk-Forward 窗口结束优化分类阈值

### Walk-Forward 回测
- 4 个窗口，每窗口训练 3 年，验证 1 年
- 固定窗口（非扩展窗口）：每窗口独立使用最近 3 年数据
- 交易仅发生在验证期（2023-01-03 ~ 2026-04-29）

## 关键文件

### 核心脚本
| 文件 | 功能 |
|------|------|
| `scripts/run_ml_backtest.py` | ML 策略回测主脚本（v1.11，~1200行） |
| `scripts/predict_today.py` | 今日持仓预测（独立运行） |
| `scripts/run_all_backtests.py` | 批量回测入口 |
| `scripts/run_all_backtests.sh` | 批量回测 Shell 脚本 |
| `scripts/sync_minute_data.py` | 分钟数据同步 |
| `scripts/verify_paper_trading.py` | 模拟交易验证 |

### 数据层
| 文件 | 功能 |
|------|------|
| `data/db.py` | 数据库引擎 + ORM 模型 |
| `data/sources/` | 各数据源同步脚本 |
| `data/validation.py` | 数据质量校验 |

### 因子
| 文件 | 功能 |
|------|------|
| `factors/__init__.py` | 因子注册表（83个因子） |
| `factors/momentum.py` | 动量/反转类因子 |
| `factors/volatility.py` | 波动率类因子 |
| `factors/liquidity.py` | 流动性类因子 |
| `factors/fundamental.py` | 财务因子（含排雷检查） |
| `factors/intraday_minute.py` | 分钟频率日内因子（7个，60min K线聚合） |
| `factors/screening.py` | 因子筛选（IC + 正交化） |

### 模型
| 文件 | 功能 |
|------|------|
| `models/dataset.py` | 因子数据集构建 + 标签生成 + 多周期 + 行业中性化 |
| `models/trainer.py` | XGBoost/LightGBM 训练 + Ensemble + MultiHorizonEnsemble |
| `models/tuning.py` | 阈值优化 |

### 组合
| 文件 | 功能 |
|------|------|
| `portfolio/selector.py` | NDrop 选股 + Top-N |
| `portfolio/risk.py` | 止损/风控检查 |
| `portfolio/backtest.py` | 回测引擎 |

### Web
| 文件 | 功能 |
|------|------|
| `web/app.py` | FastAPI 应用入口 |
| `web/routes/api.py` | API 路由（回测详情、权益曲线、标注） |

### 配置
| 文件 | 功能 |
|------|------|
| `config/settings.py` | TradingConfig 统一参数 |
| `config/watchlist.json` | Web 自选股列表 |

## 数据库表

### 核心数据表
| 表名 | 行数 | 说明 |
|------|------|------|
| `stock_daily` | - | 日频 OHLCV（全市场） |
| `stock_minute` | 256万 | 60min K线，1309只股票，2024-03起 |
| `stock_daily_extra` | - | 日频扩展数据（市值、PE、PB） |
| `stock_financial` | - | 财务报表（季报） |
| `stock_pledge` | - | 股东质押数据 |
| `stock_industry` | 2893 | 申万行业分类（19个SW1） |
| `stock_shareholder` | 42万 | 股东户数/持股数据 |
| `stock_basic` | - | 股票基本信息（名称、ST标记） |
| `index_daily` | 18147 | 指数日线（000001上证等） |

### 策略相关表
| 表名 | 说明 |
|------|------|
| `backtest_results` | 回测结果（JSON: 指标、权益曲线、日收益、标注事件） |
| `strategy_versions` | 策略版本记录 |
| `strategy_configs` | 策略参数配置 |
| `strategy_health` | 策略健康状态 |
| `factor_weights_history` | 因子权重历史 |
| `factor_availability` | 因子数据可用性 |
| `data_quality_log` | 数据质量日志 |

### 模拟交易表
| 表名 | 说明 |
|------|------|
| `paper_signals` | 交易信号 |
| `paper_orders` | 订单记录 |
| `paper_positions` | 持仓记录 |
| `paper_account` | 账户信息 |
| `paper_daily_pnl` | 每日盈亏 |
| `paper_runs` | 模拟运行记录 |

## 已完成功能

- [x] 全市场 ML 选股（83因子 → IC筛选 → 正交化 → 双模型集成 → NDrop）
- [x] 排雷过滤（8项财务质量检查）
- [x] ST 股票过滤
- [x] 组合级风控（-20%减仓、-25%清仓、指数大跌空仓）
- [x] Walk-Forward 回测（固定窗口，交易仅在验证期）
- [x] 动态反馈闭环（因子淘汰/发现 + 模型重训）
- [x] Web 权益曲线 + 事件标注（窗口/因子/重训/风控）
- [x] 今日持仓预测（predict_today）
- [x] 分钟数据同步（Sina API）
- [x] 模拟交易引擎
- [x] **分钟频率因子**（7个，60min K线聚合为日频特征）
- [x] **行业截面中性化**（neutralize_by_industry，19个SW1行业）
- [x] **多周期预测**（T+1/T+5/T+20加权，MultiHorizonEnsemble）

详见 [策略改进方向文档](strategy-improvement-ideas.md)

## 已知问题

1. **选股偏向小盘低流动性**：amihud_5 等因子主导排序，无流动性门槛（待引入流动性筛选）
2. **回测假设收盘价成交**：对低流动性股票不现实
3. **stock_tick 表为空**：无 tick 级数据，日内模式只能从 60min K 线提取
4. **stock_daily_extra.market_cap 大量 NULL**：市值因子不可用
5. **财务因子未在 predict_today 启用**：presumptive 为空，需额外数据源
6. **全量因子回测极慢**：83因子×5000+股票，单次因子计算需40+分钟（100% CPU）

## v1.12 已解决问题
- [x] 盈亏比 1.01 → 选股质量显著提升（Sharpe 2.12→3.35）
- [x] 收益高度集中 → 多周期预测分散信号来源
- [x] 止损过多 → 从96次降至94次（-2次），最大回撤从27.37%降至12.52%
- [x] 无分钟频率信号 → 7个60min K线日内因子加入选股管线
