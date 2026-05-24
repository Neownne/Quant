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

    # 默认同步的指数列表（东方财富代码）
    INDEX_CODES = {
        "000001": "上证指数",
        "399001": "深证成指",
        "399006": "创业板指",
        "000688": "科创50",
        "000300": "沪深300",
        "000905": "中证500",
        "000852": "中证1000",
    }
