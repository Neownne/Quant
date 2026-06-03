"""模拟盘策略配置中心。

定义每日跑数的所有策略及其参数。
run_daily_paper.py 遍历此列表，逐一执行。
"""
from config.settings import TradingConfig

PAPER_STRATEGIES = [
    {
        "name": "舞",
        "version": "v1.6",
        "account_id": 15,
        "run_id": 2,
        "universe_size": 500,
        "forward_days": 5,
        "train_years": 3,
        "top_n": TradingConfig.TOP_N,
        "factor_mode": "all",
    },
    {
        "name": "舞",
        "version": "v1.5",
        "account_id": 17,
        "run_id": 4,
        "universe_size": 500,
        "forward_days": 5,
        "train_years": 3,
        "top_n": TradingConfig.TOP_N,
        "factor_mode": "full",
    },
    {
        "name": "RL-Dynamic",
        "version": "v1.0",
        "account_id": 18,
        "run_id": 5,
        "type": "rl",
        "universe_size": 500,
        "forward_days": 5,
        "train_years": 3,
        "top_n": TradingConfig.TOP_N,
        "factor_mode": "all",
        "rl_timesteps": 3000,
    },
]
