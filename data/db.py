"""
数据库连接管理 & 表结构维护。
所有 SQL 操作都通过 SQLAlchemy，避免手动拼 SQL 字符串。
"""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from loguru import logger

from config.settings import DBConfig

# DDL —— 股票基本信息
DDL_STOCK_BASIC = """
CREATE TABLE IF NOT EXISTS stock_basic (
    code        VARCHAR(10) PRIMARY KEY,
    name        VARCHAR(50),
    industry    VARCHAR(50),
    area        VARCHAR(50),
    market      VARCHAR(10),
    list_date   DATE,
    is_st       BOOLEAN DEFAULT FALSE
);
"""

# DDL —— 日线行情（联合主键：code + trade_date）
DDL_STOCK_DAILY = """
CREATE TABLE IF NOT EXISTS stock_daily (
    code        VARCHAR(10),
    trade_date  DATE,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    amount      DOUBLE PRECISION,
    turnover    DOUBLE PRECISION,
    PRIMARY KEY (code, trade_date)
);
"""

# DDL —— 指数日线
DDL_INDEX_DAILY = """
CREATE TABLE IF NOT EXISTS index_daily (
    code        VARCHAR(10),
    trade_date  DATE,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    amount      DOUBLE PRECISION,
    PRIMARY KEY (code, trade_date)
);
"""

# DDL —— ETF 基本信息
DDL_ETF_BASIC = """
CREATE TABLE IF NOT EXISTS etf_basic (
    code        VARCHAR(10) PRIMARY KEY,
    name        VARCHAR(100),
    category    VARCHAR(50),
    market      VARCHAR(10)
);
"""

# DDL —— ETF 日线（结构同 stock_daily）
DDL_ETF_DAILY = """
CREATE TABLE IF NOT EXISTS etf_daily (
    code        VARCHAR(10),
    trade_date  DATE,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    amount      DOUBLE PRECISION,
    PRIMARY KEY (code, trade_date)
);
"""

# DDL —— ETF 三因子监测结果
DDL_ETF_MONITOR_DAILY = """
CREATE TABLE IF NOT EXISTS etf_monitor_daily (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    code        VARCHAR(10) NOT NULL,
    name        VARCHAR(50),
    close       DOUBLE PRECISION,
    chg_pct     DOUBLE PRECISION,
    volume_ma20 DOUBLE PRECISION,
    vol_ratio   DOUBLE PRECISION,
    vol_prob    DOUBLE PRECISION,
    dir_prob    DOUBLE PRECISION,
    share_prob  DOUBLE PRECISION,
    shares_delta_pct DOUBLE PRECISION,
    composite_prob DOUBLE PRECISION,
    signal_level VARCHAR(10),
    idx_chg     DOUBLE PRECISION,
    UNIQUE (date, code)
);
"""

# DDL —— 基金基本信息
DDL_FUND_BASIC = """
CREATE TABLE IF NOT EXISTS fund_basic (
    code        VARCHAR(10) PRIMARY KEY,
    name        VARCHAR(100)
);
"""

# DDL —— 基金净值（不同于 K 线，基金只有净值没有 OHLCV）
DDL_FUND_NAV = """
CREATE TABLE IF NOT EXISTS fund_nav (
    code            VARCHAR(10),
    nav_date        DATE,
    unit_nav        DOUBLE PRECISION,
    accumulated_nav DOUBLE PRECISION,
    daily_return    DOUBLE PRECISION,
    PRIMARY KEY (code, nav_date)
);
"""

# DDL —— 逐笔成交
DDL_STOCK_TICK = """
CREATE TABLE IF NOT EXISTS stock_tick (
    code         VARCHAR(10),
    trade_time   TIMESTAMP,
    price        DOUBLE PRECISION,
    price_change DOUBLE PRECISION,
    volume       BIGINT,
    amount       DOUBLE PRECISION,
    direction    VARCHAR(10),
    PRIMARY KEY (code, trade_time)
);
"""

# DDL —— 分钟K线
DDL_STOCK_MINUTE = """
CREATE TABLE IF NOT EXISTS stock_minute (
    code       VARCHAR(10),
    trade_time TIMESTAMP,
    period     VARCHAR(5),
    open       DOUBLE PRECISION,
    high       DOUBLE PRECISION,
    low        DOUBLE PRECISION,
    close      DOUBLE PRECISION,
    volume     BIGINT,
    amount     DOUBLE PRECISION,
    PRIMARY KEY (code, trade_time, period)
);
"""

# DDL —— 估值指标（市值、PE、PB）
DDL_STOCK_DAILY_EXTRA = """
CREATE TABLE IF NOT EXISTS stock_daily_extra (
    code             VARCHAR(10),
    trade_date       DATE,
    market_cap       DOUBLE PRECISION,   -- 总市值（亿元）
    float_market_cap DOUBLE PRECISION,   -- 流通市值（亿元）
    pe               DOUBLE PRECISION,   -- 市盈率
    pb               DOUBLE PRECISION,   -- 市净率
    total_share      DOUBLE PRECISION,   -- 总股本（亿股）
    float_share      DOUBLE PRECISION,   -- 流通股本（亿股）
    PRIMARY KEY (code, trade_date)
);
"""

# DDL —— 股东户数（散户代理变量）
DDL_STOCK_MCAP_PROXY = """
CREATE TABLE IF NOT EXISTS stock_mcap_proxy (
    code VARCHAR(10) PRIMARY KEY,
    implied_share DOUBLE PRECISION NOT NULL,
    base_mcap DOUBLE PRECISION,
    base_close DOUBLE PRECISION,
    base_date DATE,
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_STOCK_SHAREHOLDER = """
CREATE TABLE IF NOT EXISTS stock_shareholder (
    code               VARCHAR(10),
    end_date           DATE,              -- 报告期（季度末）
    shareholder_count  BIGINT,            -- 股东户数
    avg_holding_value  DOUBLE PRECISION,  -- 户均持股市值（万元）
    avg_holding_amount DOUBLE PRECISION,  -- 户均持股数量（股）
    total_market_cap   DOUBLE PRECISION,  -- 报告期总市值
    PRIMARY KEY (code, end_date)
);
"""

# DDL —— 财务数据（同花顺财务摘要）
DDL_STOCK_FINANCIAL = """
CREATE TABLE IF NOT EXISTS stock_financial (
    code               VARCHAR(10),
    report_date        DATE,
    revenue            DOUBLE PRECISION,
    net_profit         DOUBLE PRECISION,
    gross_margin       DOUBLE PRECISION,
    net_margin         DOUBLE PRECISION,
    roe                DOUBLE PRECISION,
    total_assets       DOUBLE PRECISION,
    total_liability    DOUBLE PRECISION,
    bps                DOUBLE PRECISION,
    eps                DOUBLE PRECISION,
    cash_flow          DOUBLE PRECISION,
    PRIMARY KEY (code, report_date)
);
"""

# Phase 4+: 补全资产负债表/现金流/质押等字段
DDL_STOCK_FINANCIAL_V2 = """
ALTER TABLE stock_financial ADD COLUMN IF NOT EXISTS goodwill          DOUBLE PRECISION;
ALTER TABLE stock_financial ADD COLUMN IF NOT EXISTS short_term_loans  DOUBLE PRECISION;
ALTER TABLE stock_financial ADD COLUMN IF NOT EXISTS cash_equivalents  DOUBLE PRECISION;
ALTER TABLE stock_financial ADD COLUMN IF NOT EXISTS operating_cash_flow DOUBLE PRECISION;
ALTER TABLE stock_financial ADD COLUMN IF NOT EXISTS adjusted_profit   DOUBLE PRECISION;
ALTER TABLE stock_financial ADD COLUMN IF NOT EXISTS parent_net_profit DOUBLE PRECISION;
ALTER TABLE stock_financial ADD COLUMN IF NOT EXISTS holder_equity     DOUBLE PRECISION;
"""

DDL_STOCK_PLEDGE = """
CREATE TABLE IF NOT EXISTS stock_pledge (
    code              VARCHAR(10),
    trade_date        DATE,
    pledge_ratio      DOUBLE PRECISION,
    pledge_shares     DOUBLE PRECISION,
    pledge_market_cap DOUBLE PRECISION,
    pledge_count      DOUBLE PRECISION,
    PRIMARY KEY (code, trade_date)
);
"""

# DDL —— 行业分类
DDL_STOCK_INDUSTRY = """
CREATE TABLE IF NOT EXISTS stock_industry (
    code               VARCHAR(10) PRIMARY KEY,
    industry_sw1       VARCHAR(50),
    industry_sw2       VARCHAR(50),
    market             VARCHAR(10)
);
"""

# DDL —— 模拟盘账户
DDL_PAPER_ACCOUNT = """
CREATE TABLE IF NOT EXISTS paper_account (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(50),
    initial_capital DOUBLE PRECISION,
    cash            DOUBLE PRECISION,
    created_at      TIMESTAMP DEFAULT NOW()
);
"""

# DDL —— 模拟盘委托
DDL_PAPER_ORDERS = """
CREATE TABLE IF NOT EXISTS paper_orders (
    id          SERIAL PRIMARY KEY,
    account_id  INT,
    code        VARCHAR(10),
    direction   VARCHAR(10),
    price       DOUBLE PRECISION,
    volume      INT,
    amount      DOUBLE PRECISION,
    order_time  TIMESTAMP DEFAULT NOW(),
    status      VARCHAR(20),
    note        VARCHAR(30)
);
"""

# DDL —— 模拟盘持仓（V2：脱钩 paper_account，挂钩 paper_runs）
DDL_PAPER_POSITIONS = """
CREATE TABLE IF NOT EXISTS paper_positions (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES paper_runs(id),
    signal_id INT REFERENCES paper_signals(id),
    stock_code VARCHAR(10) NOT NULL,
    entry_date DATE NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_date DATE,
    exit_price DOUBLE PRECISION,
    quantity INT NOT NULL DEFAULT 100,
    pnl DOUBLE PRECISION DEFAULT 0,
    pnl_pct DOUBLE PRECISION DEFAULT 0
);
"""

DDL_PAPER_POSITIONS_IDX = "CREATE INDEX IF NOT EXISTS idx_pp_run ON paper_positions (run_id);"

# DDL —— 模拟盘每日净值
DDL_PAPER_DAILY_PNL = """
CREATE TABLE IF NOT EXISTS paper_daily_pnl (
    id              SERIAL PRIMARY KEY,
    account_id      INT NOT NULL,
    trade_date      DATE NOT NULL,
    cash            DOUBLE PRECISION,
    position_value  DOUBLE PRECISION,
    total_value     DOUBLE PRECISION,
    daily_return    DOUBLE PRECISION,
    drawdown        DOUBLE PRECISION,
    UNIQUE (account_id, trade_date)
);
"""

# DDL —— 模拟账户扩展（策略+费率配置）
DDL_PAPER_ACCOUNT_V2 = """
ALTER TABLE paper_account ADD COLUMN IF NOT EXISTS strategy_type VARCHAR(20) DEFAULT 'ml';
ALTER TABLE paper_account ADD COLUMN IF NOT EXISTS strategy_name VARCHAR(100);
ALTER TABLE paper_account ADD COLUMN IF NOT EXISTS strategy_params JSONB DEFAULT '{}';
ALTER TABLE paper_account ADD COLUMN IF NOT EXISTS commission_rate DOUBLE PRECISION DEFAULT 0.00009;
ALTER TABLE paper_account ADD COLUMN IF NOT EXISTS stamp_duty_rate DOUBLE PRECISION DEFAULT 0.0005;
ALTER TABLE paper_account ADD COLUMN IF NOT EXISTS slippage DOUBLE PRECISION DEFAULT 0.01;
ALTER TABLE paper_account ADD COLUMN IF NOT EXISTS use_market_filter BOOLEAN DEFAULT TRUE;
"""

# DDL —— 回测结果持久化（V2：脱钩 paper_account，挂钩 strategy_versions）
DDL_BACKTEST_RESULTS = """
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    version_id INT NOT NULL REFERENCES strategy_versions(id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    quality VARCHAR(10) NOT NULL DEFAULT 'valid'
        CHECK (quality IN ('valid', 'suspect', 'invalid')),
    quality_flags TEXT[] DEFAULT '{}',
    metrics_json JSONB NOT NULL DEFAULT '{}',
    equity_curve_json JSONB NOT NULL DEFAULT '{}',
    daily_returns_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_BACKTEST_RESULTS_IDX = "CREATE INDEX IF NOT EXISTS idx_br_version ON backtest_results (version_id);"

# DDL —— 策略实验室实验记录
DDL_LAB_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS lab_experiments (
    id SERIAL PRIMARY KEY,
    batch_label VARCHAR(100),
    variant_name VARCHAR(100) NOT NULL,
    variant_params_json JSONB DEFAULT '{}',
    source VARCHAR(50) DEFAULT 'manual',
    source_url TEXT DEFAULT '',
    backtest_result_id INT REFERENCES backtest_results(id),
    composite_score DOUBLE PRECISION,
    rank INT,
    verdict VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_ML_STRATEGY_CONFIG = """
CREATE TABLE IF NOT EXISTS ml_strategy_config (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) UNIQUE NOT NULL,
    description     TEXT DEFAULT '',
    factor_names    JSONB DEFAULT '[]',
    ic_threshold    DOUBLE PRECISION DEFAULT 0.02,
    t_threshold     DOUBLE PRECISION DEFAULT 2.0,
    orthogonal_threshold DOUBLE PRECISION DEFAULT 0.7,
    label_mode      VARCHAR(20) DEFAULT 'binary',
    forward_days    INTEGER DEFAULT 1,
    train_years     INTEGER DEFAULT 3,
    val_years       INTEGER DEFAULT 1,
    model_type      VARCHAR(20) DEFAULT 'ensemble',
    top_n           INTEGER DEFAULT 15,
    rebalance_mode  VARCHAR(20) DEFAULT 'ndrop',
    ndrop_n         INTEGER DEFAULT 2,
    max_single      DOUBLE PRECISION DEFAULT 0.10,
    max_industry    DOUBLE PRECISION DEFAULT 0.30,
    stop_loss_pct           DOUBLE PRECISION DEFAULT 0.08,
    atr_multiplier          DOUBLE PRECISION DEFAULT 1.5,
    atr_period              INTEGER DEFAULT 20,
    portfolio_dd_threshold  DOUBLE PRECISION DEFAULT 0.20,
    portfolio_dd_reduce_to  DOUBLE PRECISION DEFAULT 0.50,
    max_dd_limit            DOUBLE PRECISION DEFAULT 0.25,
    stock_pool      VARCHAR(100) DEFAULT '',
    stock_count     INTEGER DEFAULT 500,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);
"""

# ========== 策略管理 ==========

DDL_STRATEGY_CONFIGS = """
CREATE TABLE IF NOT EXISTS strategy_configs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL CHECK (type IN ('ml', 'static')),
    description TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_STRATEGY_VERSIONS = """
CREATE TABLE IF NOT EXISTS strategy_versions (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    version VARCHAR(20) NOT NULL,
    algorithm_type VARCHAR(50) NOT NULL,
    feature_list_version VARCHAR(20) NOT NULL,
    model_file_path TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (strategy_id, version)
);
"""

DDL_FACTOR_WEIGHTS_HISTORY = """
CREATE TABLE IF NOT EXISTS factor_weights_history (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    factor_name VARCHAR(100) NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    effective_date DATE NOT NULL,
    source VARCHAR(10) NOT NULL CHECK (source IN ('auto', 'manual')),
    reason TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_FACTOR_WEIGHTS_HISTORY_IDX = "CREATE INDEX IF NOT EXISTS idx_fwh_strategy_date ON factor_weights_history (strategy_id, effective_date);"

# ========== 因子元数据 ==========

DDL_FACTOR_LINEAGE = """
CREATE TABLE IF NOT EXISTS factor_lineage (
    id SERIAL PRIMARY KEY,
    factor_name VARCHAR(100) NOT NULL UNIQUE,
    source_fields TEXT[] NOT NULL,
    computation_formula_hash VARCHAR(64) NOT NULL,
    upstream_factors TEXT[] DEFAULT '{}',
    last_validated_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_FACTOR_AVAILABILITY = """
CREATE TABLE IF NOT EXISTS factor_availability (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    factor_name VARCHAR(100) NOT NULL,
    data_ready_at TIMESTAMP NOT NULL,
    data_source VARCHAR(50) DEFAULT '',
    latency_ms INT DEFAULT 0,
    UNIQUE (trade_date, factor_name)
);
"""

DDL_FACTOR_AVAILABILITY_IDX = "CREATE INDEX IF NOT EXISTS idx_fa_date ON factor_availability (trade_date);"

# ========== 模拟盘运行记录 ==========

DDL_PAPER_RUNS = """
CREATE TABLE IF NOT EXISTS paper_runs (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    version_id INT NOT NULL REFERENCES strategy_versions(id),
    start_date DATE NOT NULL,
    end_date DATE,
    initial_capital DOUBLE PRECISION NOT NULL,
    status VARCHAR(10) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'paused', 'stopped')),
    created_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_PAPER_SIGNALS = """
CREATE TABLE IF NOT EXISTS paper_signals (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES paper_runs(id),
    signal_date DATE NOT NULL,
    stock_code VARCHAR(10) NOT NULL,
    predicted_score DOUBLE PRECISION,
    rank INT,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_PAPER_SIGNALS_IDX = "CREATE INDEX IF NOT EXISTS idx_ps_run_date ON paper_signals (run_id, signal_date);"

DDL_SIGNAL_FACTORS = """
CREATE TABLE IF NOT EXISTS signal_factors (
    id SERIAL PRIMARY KEY,
    signal_id INT NOT NULL REFERENCES paper_signals(id) ON DELETE CASCADE,
    factor_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    UNIQUE (signal_id, factor_name)
);
"""

DDL_SIGNAL_FACTORS_IDX_SIGNAL = "CREATE INDEX IF NOT EXISTS idx_sf_signal ON signal_factors (signal_id);"
DDL_SIGNAL_FACTORS_IDX_FACTOR = "CREATE INDEX IF NOT EXISTS idx_sf_factor ON signal_factors (factor_name);"

# ========== 归因分析 ==========

DDL_SIGNAL_ATTRIBUTION = """
CREATE TABLE IF NOT EXISTS signal_attribution (
    id SERIAL PRIMARY KEY,
    signal_id INT NOT NULL REFERENCES paper_signals(id),
    eval_date DATE NOT NULL,
    days_held INT NOT NULL DEFAULT 1,
    pnl DOUBLE PRECISION DEFAULT 0,
    pnl_pct DOUBLE PRECISION DEFAULT 0,
    factor_contrib_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_SIGNAL_ATTRIBUTION_IDX = "CREATE INDEX IF NOT EXISTS idx_sa_signal ON signal_attribution (signal_id);"

# ========== 权重调整记录 ==========

DDL_WEIGHT_ADJUSTMENTS = """
CREATE TABLE IF NOT EXISTS weight_adjustments (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    factor_name VARCHAR(100) NOT NULL,
    old_weight DOUBLE PRECISION NOT NULL,
    new_weight DOUBLE PRECISION NOT NULL,
    confidence_level DOUBLE PRECISION DEFAULT 0.95,
    source VARCHAR(10) NOT NULL CHECK (source IN ('auto', 'manual')),
    reason TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW()
);
"""

# ========== 策略健康 ==========

DDL_STRATEGY_HEALTH = """
CREATE TABLE IF NOT EXISTS strategy_health (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    overall_ic DOUBLE PRECISION,
    max_drawdown_7d DOUBLE PRECISION,
    regime_tag VARCHAR(10) DEFAULT 'unknown'
        CHECK (regime_tag IN ('bull', 'bear', 'range', 'unknown')),
    status VARCHAR(10) NOT NULL DEFAULT 'normal'
        CHECK (status IN ('normal', 'warning', 'critical')),
    action_required VARCHAR(20) DEFAULT 'none',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (strategy_id, date)
);
"""

# ========== 指令队列 ==========

DDL_STRATEGY_COMMANDS = """
CREATE TABLE IF NOT EXISTS strategy_commands (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    command_type VARCHAR(30) NOT NULL
        CHECK (command_type IN ('adjust_weight', 'pause', 'resume', 'rollback', 'retrain')),
    payload_json JSONB NOT NULL DEFAULT '{}',
    requested_by VARCHAR(50) DEFAULT 'user',
    requested_at TIMESTAMP DEFAULT NOW(),
    executed_at TIMESTAMP,
    execution_result TEXT DEFAULT '',
    rolled_back_by INT REFERENCES strategy_commands(id)
);
"""

# ========== 数据质量 ==========

DDL_DATA_QUALITY_LOG = """
CREATE TABLE IF NOT EXISTS data_quality_log (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    check_name VARCHAR(50) NOT NULL,
    expected_value TEXT,
    actual_value TEXT,
    passed BOOLEAN NOT NULL DEFAULT FALSE,
    detail TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_DATA_QUALITY_LOG_IDX = "CREATE INDEX IF NOT EXISTS idx_dql_date ON data_quality_log (trade_date);"


# ========== 策略管理 ==========

DDL_STRATEGY_CONFIGS = """
CREATE TABLE IF NOT EXISTS strategy_configs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL CHECK (type IN ('ml', 'static')),
    description TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_STRATEGY_VERSIONS = """
CREATE TABLE IF NOT EXISTS strategy_versions (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    version VARCHAR(20) NOT NULL,
    algorithm_type VARCHAR(50) NOT NULL,
    feature_list_version VARCHAR(20) NOT NULL,
    model_file_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (strategy_id, version)
);
"""

DDL_FACTOR_WEIGHTS_HISTORY = """
CREATE TABLE IF NOT EXISTS factor_weights_history (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    factor_name VARCHAR(100) NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    effective_date DATE NOT NULL,
    source VARCHAR(10) NOT NULL CHECK (source IN ('auto', 'manual')),
    reason TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_FACTOR_WEIGHTS_HISTORY_IDX = "CREATE INDEX IF NOT EXISTS idx_fwh_strategy_date ON factor_weights_history (strategy_id, effective_date);"

# ========== 因子元数据 ==========

DDL_FACTOR_LINEAGE = """
CREATE TABLE IF NOT EXISTS factor_lineage (
    id SERIAL PRIMARY KEY,
    factor_name VARCHAR(100) NOT NULL UNIQUE,
    source_fields TEXT[] NOT NULL,
    computation_formula_hash VARCHAR(64) NOT NULL,
    upstream_factors TEXT[] DEFAULT '{}',
    last_validated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_FACTOR_AVAILABILITY = """
CREATE TABLE IF NOT EXISTS factor_availability (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    factor_name VARCHAR(100) NOT NULL,
    data_ready_at TIMESTAMPTZ NOT NULL,
    data_source VARCHAR(50) DEFAULT '',
    latency_ms INT DEFAULT 0,
    UNIQUE (trade_date, factor_name)
);
"""

DDL_FACTOR_AVAILABILITY_IDX = "CREATE INDEX IF NOT EXISTS idx_fa_date ON factor_availability (trade_date);"

# ========== 回测结果（扩展） ==========

DDL_BACKTEST_RESULTS_V2 = """
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    version_id INT NOT NULL REFERENCES strategy_versions(id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    quality VARCHAR(10) NOT NULL DEFAULT 'valid'
        CHECK (quality IN ('valid', 'suspect', 'invalid')),
    quality_flags TEXT[] DEFAULT '{}',
    metrics_json JSONB NOT NULL DEFAULT '{}',
    equity_curve_json JSONB NOT NULL DEFAULT '{}',
    daily_returns_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_BACKTEST_RESULTS_V2_IDX = "CREATE INDEX IF NOT EXISTS idx_br_version ON backtest_results (version_id);"

# ========== 模拟盘 ==========

DDL_PAPER_RUNS = """
CREATE TABLE IF NOT EXISTS paper_runs (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    version_id INT NOT NULL REFERENCES strategy_versions(id),
    start_date DATE NOT NULL,
    end_date DATE,
    initial_capital DOUBLE PRECISION NOT NULL,
    status VARCHAR(10) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'paused', 'stopped')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_PAPER_SIGNALS = """
CREATE TABLE IF NOT EXISTS paper_signals (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES paper_runs(id),
    signal_date DATE NOT NULL,
    stock_code VARCHAR(10) NOT NULL,
    predicted_score DOUBLE PRECISION,
    rank INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_PAPER_SIGNALS_IDX = "CREATE INDEX IF NOT EXISTS idx_ps_run_date ON paper_signals (run_id, signal_date);"

DDL_SIGNAL_FACTORS = """
CREATE TABLE IF NOT EXISTS signal_factors (
    id SERIAL PRIMARY KEY,
    signal_id INT NOT NULL REFERENCES paper_signals(id) ON DELETE CASCADE,
    factor_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    UNIQUE (signal_id, factor_name)
);
"""

DDL_SIGNAL_FACTORS_IDX_SIGNAL = "CREATE INDEX IF NOT EXISTS idx_sf_signal ON signal_factors (signal_id);"
DDL_SIGNAL_FACTORS_IDX_FACTOR = "CREATE INDEX IF NOT EXISTS idx_sf_factor_date ON signal_factors (factor_name);"

DDL_PAPER_POSITIONS_V2 = """
CREATE TABLE IF NOT EXISTS paper_positions (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES paper_runs(id),
    signal_id INT REFERENCES paper_signals(id),
    stock_code VARCHAR(10) NOT NULL,
    entry_date DATE NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_date DATE,
    exit_price DOUBLE PRECISION,
    quantity INT NOT NULL DEFAULT 100,
    pnl DOUBLE PRECISION DEFAULT 0,
    pnl_pct DOUBLE PRECISION DEFAULT 0
);
"""

DDL_PAPER_POSITIONS_V2_IDX = "CREATE INDEX IF NOT EXISTS idx_pp_run ON paper_positions (run_id);"

# ========== 归因分析 ==========

DDL_SIGNAL_ATTRIBUTION = """
CREATE TABLE IF NOT EXISTS signal_attribution (
    id SERIAL PRIMARY KEY,
    signal_id INT NOT NULL REFERENCES paper_signals(id),
    eval_date DATE NOT NULL,
    days_held INT NOT NULL DEFAULT 1,
    pnl DOUBLE PRECISION DEFAULT 0,
    pnl_pct DOUBLE PRECISION DEFAULT 0,
    factor_contrib_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_SIGNAL_ATTRIBUTION_IDX = "CREATE INDEX IF NOT EXISTS idx_sa_signal ON signal_attribution (signal_id);"

# ========== 权重调整记录 ==========

DDL_WEIGHT_ADJUSTMENTS = """
CREATE TABLE IF NOT EXISTS weight_adjustments (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    factor_name VARCHAR(100) NOT NULL,
    old_weight DOUBLE PRECISION NOT NULL,
    new_weight DOUBLE PRECISION NOT NULL,
    confidence_level DOUBLE PRECISION DEFAULT 0.95,
    source VARCHAR(10) NOT NULL CHECK (source IN ('auto', 'manual')),
    reason TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

# ========== 策略健康 ==========

DDL_STRATEGY_HEALTH = """
CREATE TABLE IF NOT EXISTS strategy_health (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    overall_ic DOUBLE PRECISION,
    max_drawdown_7d DOUBLE PRECISION,
    regime_tag VARCHAR(10) DEFAULT 'unknown'
        CHECK (regime_tag IN ('bull', 'bear', 'range', 'unknown')),
    status VARCHAR(10) NOT NULL DEFAULT 'normal'
        CHECK (status IN ('normal', 'warning', 'critical')),
    action_required VARCHAR(20) DEFAULT 'none',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (strategy_id, date)
);
"""

# ========== 指令队列 ==========

DDL_STRATEGY_COMMANDS = """
CREATE TABLE IF NOT EXISTS strategy_commands (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    command_type VARCHAR(30) NOT NULL
        CHECK (command_type IN ('adjust_weight', 'pause', 'resume', 'rollback', 'retrain')),
    payload_json JSONB NOT NULL DEFAULT '{}',
    requested_by VARCHAR(50) DEFAULT 'user',
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    execution_result TEXT DEFAULT '',
    rolled_back_by INT REFERENCES strategy_commands(id)
);
"""

# ========== 数据质量 ==========

DDL_DATA_QUALITY_LOG = """
CREATE TABLE IF NOT EXISTS data_quality_log (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    check_name VARCHAR(50) NOT NULL,
    expected_value TEXT,
    actual_value TEXT,
    passed BOOLEAN NOT NULL DEFAULT FALSE,
    detail TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_DATA_QUALITY_LOG_IDX = "CREATE INDEX IF NOT EXISTS idx_dql_date ON data_quality_log (trade_date);"


def get_engine() -> Engine:
    """创建数据库引擎（每次调用返回同一个连接池）。"""
    return create_engine(
        DBConfig.url(),
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,          # 连接前检测可用性
        connect_args={
            "options": "-c timezone=Asia/Shanghai"
        },
    )


def init_db() -> None:
    """初始化所有表结构（幂等 —— 表不存在才创建）。"""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(DDL_STOCK_BASIC))
        conn.execute(text(DDL_STOCK_DAILY))
        conn.execute(text(DDL_INDEX_DAILY))
        # ETF 表已启用
        conn.execute(text(DDL_ETF_BASIC))
        conn.execute(text(DDL_ETF_DAILY))
        conn.execute(text(DDL_ETF_MONITOR_DAILY))
        # conn.execute(text(DDL_FUND_BASIC))
        # conn.execute(text(DDL_FUND_NAV))
        conn.execute(text(DDL_STOCK_TICK))
        conn.execute(text(DDL_STOCK_MINUTE))
        conn.execute(text(DDL_STOCK_DAILY_EXTRA))
        conn.execute(text(DDL_STOCK_MCAP_PROXY))
        conn.execute(text(DDL_STOCK_SHAREHOLDER))
        conn.execute(text(DDL_STOCK_FINANCIAL))
        conn.execute(text(DDL_STOCK_FINANCIAL_V2))
        conn.execute(text(DDL_STOCK_INDUSTRY))
        conn.execute(text(DDL_STOCK_PLEDGE))
        conn.execute(text(DDL_PAPER_ACCOUNT))
        conn.execute(text(DDL_PAPER_ORDERS))
        conn.execute(text(DDL_PAPER_DAILY_PNL))
        conn.execute(text(DDL_PAPER_ACCOUNT_V2))
        conn.execute(text(DDL_ML_STRATEGY_CONFIG))
        # ========== 策略管理 ==========
        conn.execute(text(DDL_STRATEGY_CONFIGS))
        conn.execute(text(DDL_STRATEGY_VERSIONS))
        conn.execute(text(DDL_FACTOR_WEIGHTS_HISTORY))
        conn.execute(text(DDL_FACTOR_WEIGHTS_HISTORY_IDX))
        # ========== 因子元数据 ==========
        conn.execute(text(DDL_FACTOR_LINEAGE))
        conn.execute(text(DDL_FACTOR_AVAILABILITY))
        conn.execute(text(DDL_FACTOR_AVAILABILITY_IDX))
        # ========== 回测结果（新版：挂钩 strategy_versions） ==========
        conn.execute(text(DDL_BACKTEST_RESULTS))
        conn.execute(text(DDL_BACKTEST_RESULTS_IDX))
        # ========== 策略实验室 ==========
        conn.execute(text(DDL_LAB_EXPERIMENTS))
        # ========== 模拟盘运行记录 ==========
        conn.execute(text(DDL_PAPER_RUNS))
        conn.execute(text(DDL_PAPER_SIGNALS))
        conn.execute(text(DDL_PAPER_SIGNALS_IDX))
        conn.execute(text(DDL_SIGNAL_FACTORS))
        conn.execute(text(DDL_SIGNAL_FACTORS_IDX_SIGNAL))
        conn.execute(text(DDL_SIGNAL_FACTORS_IDX_FACTOR))
        # ========== 模拟盘持仓（新版：挂钩 paper_runs） ==========
        conn.execute(text(DDL_PAPER_POSITIONS))
        conn.execute(text(DDL_PAPER_POSITIONS_IDX))
        # ========== 归因分析 ==========
        conn.execute(text(DDL_SIGNAL_ATTRIBUTION))
        conn.execute(text(DDL_SIGNAL_ATTRIBUTION_IDX))
        # ========== 权重调整记录 ==========
        conn.execute(text(DDL_WEIGHT_ADJUSTMENTS))
        # ========== 策略健康 ==========
        conn.execute(text(DDL_STRATEGY_HEALTH))
        # ========== 指令队列 ==========
        conn.execute(text(DDL_STRATEGY_COMMANDS))
        # ========== 数据质量 ==========
        conn.execute(text(DDL_DATA_QUALITY_LOG))
        conn.execute(text(DDL_DATA_QUALITY_LOG_IDX))
        # migration: add note column for trade reason tracking
        try:
            conn.execute(text(
                "ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS note VARCHAR(100)"
            ))
        except Exception:
            pass
    logger.info("数据库表初始化完成")
    engine.dispose()


def upsert_df(df: pd.DataFrame, table: str, engine: Engine | None = None) -> int:
    """
    将 DataFrame 按主键 upsert 到表中。
    主键冲突时更新（ON CONFLICT ... DO UPDATE），无冲突时插入。
    返回写入行数。
    """
    if df.empty:
        return 0

    _engine = engine or get_engine()
    own_engine = engine is None

    with _engine.begin() as conn:
        # 临时表加随机后缀，避免多线程并发冲突
        import uuid
        tmp = f"_tmp_{table}_{uuid.uuid4().hex[:8]}"
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
        df.to_sql(tmp, conn, if_exists="replace", index=False)

        # 获取列名用于动态生成 SQL
        cols = df.columns.tolist()
        col_names = ", ".join(f'"{c}"' for c in cols)
        excluded = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c != "code")

        # 根据表类型确定主键
        pk_map = {
            "stock_basic": "code",
            "etf_basic": "code",
            "fund_basic": "code",
        }
        if table in pk_map:
            pk = pk_map[table]
        elif "minute" in table:
            pk = "code, trade_time, period"
        elif "tick" in table:
            pk = "code, trade_time"
        elif "shareholder" in table:
            pk = "code, end_date"
        elif "financial" in table:
            pk = "code, report_date"
        elif "industry" in table:
            pk = "code"
        elif "fund_nav" in table:
            pk = "code, nav_date"
        else:
            pk = "code, trade_date"

        sql = f"""
            INSERT INTO {table} ({col_names})
            SELECT {col_names} FROM {tmp}
            ON CONFLICT ({pk}) DO UPDATE SET {excluded};
        """
        result = conn.execute(text(sql))
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))

    if own_engine:
        _engine.dispose()

    return result.rowcount


def get_existing_dates(table: str, code: str, engine: Engine | None = None) -> set:
    """获取某只股票已有的交易日集合，用于增量同步。"""
    _engine = engine or get_engine()
    own_engine = engine is None

    with _engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT trade_date FROM {table} WHERE code = :code"),
            {"code": code},
        ).fetchall()

    if own_engine:
        _engine.dispose()

    return {r[0] for r in rows}
