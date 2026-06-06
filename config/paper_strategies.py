"""模拟盘策略配置中心。

定义每日跑数的所有策略及其参数。
run_daily_paper.py 遍历此列表，逐一执行。
"""
from config.settings import TradingConfig

PAPER_STRATEGIES = [
    {
        "name": "舞",
        "version": "v1.85",
        "account_id": 15,
        "run_id": 2,
        "universe_size": 1000,  # v1.85: adaptive N 动态调仓, Sharpe ~1.63 (2023-2025)
        "forward_days": 1,  # v1.85: 1日预测对齐1日交易
        "train_years": 3,
        "top_n": TradingConfig.TOP_N,
        "factor_mode": "all",
        "adaptive_ndrop": True,  # v1.85: 自适应 N，基于分数离散度
    },
]
