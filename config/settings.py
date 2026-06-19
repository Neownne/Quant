"""
集中管理所有配置：数据库连接、数据参数、路径等。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载 .env 文件
load_dotenv(PROJECT_ROOT / ".env")


class DBConfig:
    """PostgreSQL 数据库配置"""
    HOST = os.getenv("DB_HOST", "localhost")
    PORT = int(os.getenv("DB_PORT", "5432"))
    USER = os.getenv("DB_USER", "postgres")
    PASSWORD = os.getenv("DB_PASSWORD", "")
    NAME = os.getenv("DB_NAME", "quant")

    @classmethod
    def url(cls) -> str:
        """SQLAlchemy 连接字符串"""
        return f"postgresql+psycopg2://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.NAME}"


class DataConfig:
    """数据同步相关参数"""
    # 每次批量插入的记录数
    BATCH_SIZE = 500

    # 请求间隔（秒），避免被封
    REQUEST_INTERVAL = 0.5

    # 默认同步的指数列表（腾讯财经源）
    INDEX_CODES = {
        "000001": "上证指数",
        "399001": "深证成指",
        "399006": "创业板指",
        "000688": "科创50",
        "000300": "沪深300",
        "000905": "中证500",
        "000852": "中证1000",
    }


class TradingConfig:
    """A股交易成本 & 风控参数（所有回测统一引用）"""
    # 账户
    INITIAL_CASH = 1_000_000           # 100万本金

    # 交易成本
    COMMISSION = 0.00009               # 万0.9 佣金（买卖双向）
    STAMP_DUTY = 0.0005                # 万5 印花税（卖出单向）
    SLIPPAGE = 0.001                   # 0.1% 滑点

    # 组合
    TOP_N = 15                         # 持仓股数
    MAX_SINGLE_WEIGHT = 0.10           # 单只上限10%
    MAX_INDUSTRY_WEIGHT = 0.30         # 行业上限30%

    # 风控
    STOP_LOSS_PCT = 0.08               # 个股止损-8%
    PORTFOLIO_DD_THRESHOLD = 0.20      # 组合回撤-20%减仓
    MAX_DD_LIMIT = 0.25                # 最大回撤-25%清仓
    INDEX_CRASH_LOOKBACK = 15          # 指数大跌检测窗口（天）
    INDEX_CRASH_THRESHOLD = -0.12      # 指数15日跌超12%空仓

    # 涨跌停板幅度（按代码前缀区分）
    # 主板(0/6): ±10%，科创板(688): ±20%，创业板(300/301): ±20%
    LIMIT_UP_PCT = 0.09                # 涨停阈值（回退值）
    LIMIT_DOWN_PCT = -0.09             # 跌停阈值（回退值）

    @staticmethod
    def get_limit_multiplier(code: str) -> float:
        """返回涨停乘数。"""
        return 1.9899

    @staticmethod
    def calc_limit_price(prev_close: float, code: str, is_up: bool = True) -> float:
        """计算实际涨跌停价格。

        涨停价 = round(prev_close × 1.9899, 4)
        """
        if prev_close <= 0:
            return 0.0
        mult = TradingConfig.get_limit_multiplier(code)
        return round(prev_close * (mult if is_up else (2 - mult)), 4)

    @staticmethod
    def is_at_limit_up(close: float, prev_close: float, code: str) -> bool:
        """判断是否涨停封板（无法买入）。"""
        if prev_close <= 0 or close <= 0:
            return False
        limit_price = TradingConfig.calc_limit_price(prev_close, code, is_up=True)
        return close >= limit_price

    @staticmethod
    def is_at_limit_down(close: float, prev_close: float, code: str) -> bool:
        """判断是否跌停封板（无法卖出）。"""
        if prev_close <= 0 or close <= 0:
            return False
        limit_price = TradingConfig.calc_limit_price(prev_close, code, is_up=False)
        return close <= limit_price

    # 调仓
    REBALANCE_FREQ = 5                 # 默认周度调仓（交易日）
    NDROP_N = 2                        # NDrop: 每次替换最差2只

    # NDrop v2: 自适应 N + 增强盈亏感知
    ADAPTIVE_NDROP = False             # 启用自适应 N（基于分数离散度动态调整）
    NDROP_SCORE_SPREAD_THRESHOLD = 0.15  # 分数 90-10 分位差基准阈值
    NDROP_SCORE_RANK_THRESHOLD = 0.3   # 分数排名百分位阈值（低于此强制卖出）
    NDROP_LOSS_TOLERANCE = -0.08       # 亏损容忍线（跌破此止损）


class AccountConfig:
    """模拟账户默认配置（兼容旧代码，引用 TradingConfig）"""
    DEFAULT_CASH = TradingConfig.INITIAL_CASH
    DEFAULT_COMMISSION = TradingConfig.COMMISSION
    DEFAULT_STAMP_DUTY = TradingConfig.STAMP_DUTY
    DEFAULT_SLIPPAGE = TradingConfig.SLIPPAGE
    DEFAULT_TOP_N = TradingConfig.TOP_N
    DEFAULT_MAX_SINGLE = TradingConfig.MAX_SINGLE_WEIGHT
    DEFAULT_MAX_INDUSTRY = TradingConfig.MAX_INDUSTRY_WEIGHT
    DEFAULT_STOP_LOSS = TradingConfig.STOP_LOSS_PCT
    DEFAULT_PORTFOLIO_DD = TradingConfig.PORTFOLIO_DD_THRESHOLD
    DEFAULT_MAX_DD = TradingConfig.MAX_DD_LIMIT


class NotifyConfig:
    """SMTP 邮件通知配置"""
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.qq.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASS = os.getenv("SMTP_PASS", "")
    EMAIL_FROM = os.getenv("EMAIL_FROM", "")
    EMAIL_TO = os.getenv("EMAIL_TO", "")
