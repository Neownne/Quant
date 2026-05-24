# Quant — A 股量化交易项目

> 最后更新：2026-05-24  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)（私有仓库）

---

## 架构概览

```
                    ┌─────────────────────────────────────────┐
                    │          Streamlit Web UI (app/)         │
                    │  ┌─────────┐ ┌──────┐ ┌──────────────┐  │
                    │  │实时报价  │ │K线图 │ │策略回测/编辑器│  │
                    │  └────┬────┘ └──┬───┘ └──────┬───────┘  │
                    │       └─────────┼─────────────┘          │
                    │    data_loader.py / backtest_runner.py   │
                    └─────────────────┼────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │           data/ 数据层                        │
              │  ┌──────────┐  ┌──────┐  ┌────────────────┐  │
              │  │fetcher.py│  │db.py │  │sync.py/recorder│  │
              │  │ AKShare  │  │ PG   │  │ 数据同步/录制   │  │
              │  └────┬─────┘  └──┬───┘  └───────┬────────┘  │
              │       └───────────┼───────────────┘           │
              └───────────────────┼───────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     PostgreSQL 数据库       │
                    │  12 张表（行情 + 模拟盘）    │
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
├── PROJECT.md                # 本文件
│
├── config/
│   └── settings.py           # 集中配置（数据库、数据参数）
│
├── data/                     # 数据层
│   ├── db.py                 # 12 张表 DDL + 连接池 + upsert
│   ├── fetcher.py            # AKShare 数据获取（全部数据源）
│   ├── sync.py               # 历史数据批量同步（多进程增量）
│   └── recorder.py           # 实盘数据录制（分钟K线 + 逐笔）
│
├── strategies/               # 策略层
│   ├── README.md             # 策略文档（逻辑、参数、风险提示）
│   ├── __init__.py           # STRATEGY_REGISTRY + 自定义策略加载
│   ├── sma_cross.py          # 双均线交叉策略
│   ├── macd_strategy.py      # MACD 金叉死叉策略
│   └── rsi_strategy.py       # RSI 超买超卖策略
│
├── app/                      # Web 界面层
│   ├── main.py               # 入口：导航 + sys.path + 每日同步调度
│   ├── pages/
│   │   ├── 1_📈_Watchlist.py     # 自选股实时报价
│   │   ├── 2_📊_Charts.py        # K线图 + 技术指标 + 自选分组
│   │   ├── 3_🧪_Backtest.py      # 策略回测引擎
│   │   ├── 4_📋_Paper_Trade.py   # 模拟盘账户/持仓/委托
│   │   └── 5_📝_Strategy_Editor.py # 在线策略编辑器
│   └── utils/
│       ├── data_loader.py    # DB查询 + 指标计算 + K线图构建 + 实时行情
│       └── backtest_runner.py # backtrader 回测封装 + 自定义分析器
│
└── notebooks/                # Jupyter Notebook（预留）
```

---

## 数据流

### 1. 历史数据同步

```
data/sync.py
  │
  ├──> data/fetcher.py  ─── AKShare API ─── 腾讯/新浪/交易所
  │         │
  │         └── 返回 DataFrame（字段对齐数据库表结构）
  │
  └──> data/db.py       ─── PostgreSQL
            │
            └── upsert_df()：临时表 → ON CONFLICT DO UPDATE
```

**sync.py 特性**：
- 多进程并发（ProcessPoolExecutor，因为 AKShare V8 引擎不支持多线程）
- 增量同步：计算最近交易日，跳过已覆盖到最新日期的股票
- 单股 60s 超时自动跳过
- tqdm 进度条

### 2. 实时数据（交易时段）

```
data/recorder.py
  │
  ├── 分钟K线模式：每 60s 轮询，并发拉取最新分钟数据 → stock_minute 表
  └── 逐笔成交模式：收盘后一次性抓取 → stock_tick 表
```

### 3. Web 界面数据流

```
app/main.py (入口)
  │
  ├── 后台线程：每日 18:00 自动运行 data/sync.py
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
        │                 ├── PgDataFeed：DataFrame → backtrader 数据源
        │                 ├── TradeRecorder：自定义分析器记录逐笔交易
        │                 ├── PercentSizer(95%)：仓位管理
        │                 └── 分析器：SharpeRatio, DrawDown, TradeAnalyzer, Returns
        │
        └── Strategy Editor ──> ~/.quant_strategies/（保存 .py 文件）
                                │
                                └── strategies/__init__.py 动态加载
```

---

## 数据库表结构（12 张表）

### 行情数据（9 张）

| 表 | 主键 | 记录数 | 数据源 |
|---|---|---|---|
| `stock_basic` | code | 5,522 | 深交所 + 上交所 + 北交所官网 |
| `stock_daily` | (code, trade_date) | 7,140,788 | 腾讯财经（后复权） |
| `index_daily` | (code, trade_date) | 10,815 | 腾讯财经 |
| `etf_basic` | code | 1,528 | 新浪财经 |
| `etf_daily` | (code, trade_date) | 1,166,415 | 新浪财经 |
| `fund_basic` | code | 16,958 | 天天基金 |
| `fund_nav` | (code, nav_date) | 同步中 | 天天基金 |
| `stock_tick` | (code, trade_time) | 0 | 腾讯财经逐笔 |
| `stock_minute` | (code, trade_time, period) | 0 | 新浪财经分钟K线 |

### 模拟盘（3 张）

| 表 | 说明 |
|---|---|
| `paper_account` | 模拟账户（名称、初始资金、现金） |
| `paper_orders` | 委托记录（代码、方向、价格、数量、状态） |
| `paper_positions` | 当前持仓（代码、数量、均价） |

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

## 文件关联关系

### config/settings.py → 全局配置中心
- `DBConfig`：被 `data/db.py` 的 `get_engine()` 使用，控制数据库连接
- `DataConfig`：被 `data/sync.py` 和 `data/fetcher.py` 使用，控制请求间隔和指数列表

### data/fetcher.py → 数据获取
- 被 `data/sync.py` 调用（批量历史同步）
- 被 `data/recorder.py` 调用（实盘录制）
- 被 `app/utils/data_loader.py` 间接调用（通过 `akshare` 直调实时行情）
- 所有函数统一返回 pandas DataFrame，字段名对齐数据库表结构
- `@retry_on_network_error` 装饰器提供指数退避自动重试

### data/db.py → 数据库层
- 被所有需要读写数据库的模块调用
- `init_db()`：建表（幂等），被 `app/main.py` 各页面调用
- `upsert_df()`：写入，被 `data/sync.py` 和 `data/recorder.py` 调用
- `get_existing_dates()`：增量查询，被 `data/sync.py` 调用

### app/utils/data_loader.py → Web 数据服务
- 被所有页面调用，封装 DB 查询和指标计算
- `@st.cache_data` 缓存策略：实时行情 5s、日线 60s、股票列表 3600s
- `build_kline_chart()`：生成 Plotly 多 pane K 线图

### app/utils/backtest_runner.py → 回测引擎
- 被 Backtest 页面调用
- 封装 backtrader 的 Cerebro、数据源、分析器、仓位管理
- `TradeRecorder`：自定义分析器，通过 `notify_order` 记录逐笔交易详情

### strategies/__init__.py → 策略注册
- 被 `app/pages/3_🧪_Backtest.py` 调用（获取策略列表）
- 动态扫描 `~/.quant_strategies/` 加载自定义策略

---

## 启动方式

```bash
cd /Users/chenwan/Documents/quant
source .venv/bin/activate

# 数据同步（首次使用）
python -m data.sync                        # 全量同步
python -m data.sync --mode stock-daily     # 只更新日线

# 启动 Web 界面
streamlit run app/main.py                  # 浏览器打开 http://localhost:8501
```

---

## 变更记录

| 日期 | 变更内容 |
|---|---|
| 2026-05-24 | ETF/基金全页面支持：data_loader 按资产类型自动路由（stock_daily/etf_daily/fund_nav），基金净值曲线图，类型标签显示；Git 初始化 + GitHub 私有仓库推送；.gitignore 排除 .env 和 .claude/ |
| 2026-05-24 | Web 策略编辑器（在线编写/保存/编译测试，自动注册到回测页面）；策略文档（strategies/README.md，含逻辑/参数/风险提示）；自选股非交易时段回退最近交易日数据；K线图代码直输+自选分组快捷切换；回测默认参数更新（20万本金、佣金万0.85） |
| 2026-05-24 | Streamlit Web 界面：实时报价、K线图+技术指标（MA/MACD/RSI/Bollinger）、策略回测、模拟盘、每日自动同步调度；新增 paper_account/paper_orders/paper_positions 3张模拟盘表；backtrader 回测引擎封装 |
| 2026-05-23 | 全面弃用东方财富，数据源迁移至腾讯/新浪/交易所；后复权统一；sync v2（tqdm + ProcessPoolExecutor + 跳过已完成 + 60s 超时）；db.py 临时表加 UUID；新增 stock_tick/stock_minute 表 + recorder.py |
| 2026-05-22 | 初始化项目结构、数据模块、配置文件、新增 ETF 和基金支持、网络重试机制 |
