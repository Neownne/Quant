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
    status      VARCHAR(20)
);
"""

# DDL —— 模拟盘持仓
DDL_PAPER_POSITIONS = """
CREATE TABLE IF NOT EXISTS paper_positions (
    id         SERIAL PRIMARY KEY,
    account_id INT,
    code       VARCHAR(10),
    volume     INT,
    avg_cost   DOUBLE PRECISION,
    UNIQUE (account_id, code)
);
"""

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
        # -- ETF/基金表暂时禁用 --
        # conn.execute(text(DDL_ETF_BASIC))
        # conn.execute(text(DDL_ETF_DAILY))
        # conn.execute(text(DDL_FUND_BASIC))
        # conn.execute(text(DDL_FUND_NAV))
        conn.execute(text(DDL_STOCK_TICK))
        conn.execute(text(DDL_STOCK_MINUTE))
        conn.execute(text(DDL_STOCK_DAILY_EXTRA))
        conn.execute(text(DDL_STOCK_SHAREHOLDER))
        conn.execute(text(DDL_STOCK_FINANCIAL))
        conn.execute(text(DDL_STOCK_INDUSTRY))
        conn.execute(text(DDL_PAPER_ACCOUNT))
        conn.execute(text(DDL_PAPER_ORDERS))
        conn.execute(text(DDL_PAPER_POSITIONS))
        conn.execute(text(DDL_PAPER_DAILY_PNL))
    logger.info("数据库表初始化完成（13张表）")
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
