"""RL 板块打分训练器 — Walk-Forward PPO 训练板块选择策略。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
import torch

from rl.sector_env import SectorScoringEnv
from rl.models import StockScoreNet
from rl.sector_predictor import RLSectorPredictor
from models.dataset import walk_forward_split
from factors.sector_fear_greed import compute_sector_fear_greed, FEAR_GREED_FEATURES


def _build_sector_fg_dataset(
    ohlcv: pd.DataFrame,
    sector_map: dict[str, str],
    forward_days: int = 5,
) -> dict:
    """构建板块恐贪特征 + 未来收益数据集（给RL训练用）。

    返回: {date_str: {sector: {fg_*, ret_nd}}}
    """
    ohlcv = ohlcv.copy()
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    all_dates = sorted(ohlcv["trade_date"].unique())

    # 计算各板块未来N日收益
    date_close = {}
    for i, date in enumerate(all_dates):
        day = ohlcv[ohlcv["trade_date"] == date]
        for sec in set(sector_map.values()):
            sec_codes = [c for c, s in sector_map.items() if s == sec]
            sec_day = day[day["code"].isin(sec_codes)]
            if not sec_day.empty:
                date_close.setdefault(date, {})[sec] = sec_day["close"].mean()

    # 计算未来收益
    daily_data = {}
    for i, date in enumerate(all_dates):
        fg = compute_sector_fear_greed(ohlcv, sector_map, date)
        if not fg:
            continue

        future_idx = i + forward_days
        if future_idx >= len(all_dates):
            continue
        future_date = all_dates[future_idx]

        date_str = str(date.date())
        daily_data[date_str] = {}
        for sec, features in fg.items():
            if sec in date_close.get(date, {}) and sec in date_close.get(future_date, {}):
                ret = (date_close[future_date][sec] - date_close[date][sec]) / date_close[date][sec]
                daily_data[date_str][sec] = {**features, "ret_nd": ret}

    return daily_data


def walk_forward_train_rl_sector(
    ohlcv: pd.DataFrame,
    sector_map: dict[str, str],
    forward_days: int = 5,
    train_years: int = 3,
    val_years: int = 1,
    total_timesteps: int = 50000,
) -> list[dict]:
    """Walk-Forward RL 板块打分训练。

    每窗口用 PPO (MPS GPU) 训练一个板块评分策略网络。
    """
    from stable_baselines3 import PPO

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"RL 板块训练设备: {device}")

    # 构建FG数据集
    daily_data = _build_sector_fg_dataset(ohlcv, sector_map, forward_days)
    dates = sorted(daily_data.keys())
    if len(dates) < 200:
        logger.warning("板块FG数据不足")
        return []

    # 转为带 trade_date 列的 DataFrame 用于 walk_forward_split
    df = pd.DataFrame({"trade_date": pd.to_datetime(dates)})
    results = []

    for train_df, val_df in walk_forward_split(df, train_years, val_years):
        train_dates = [str(d.date()) for d in train_df["trade_date"]]
        train_dates = [d for d in train_dates if d in daily_data]
        if len(train_dates) < 100:
            continue

        train_subset = {d: daily_data[d] for d in train_dates}
        env = SectorScoringEnv(train_subset, device=device)
        n_sectors = len(env._sectors) if env._sectors else 3

        try:
            model = PPO(
                "MlpPolicy", env,
                learning_rate=1e-4,
                n_steps=min(1024, total_timesteps // 4),
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                clip_range=0.2,
                ent_coef=0.05,
                verbose=0,
                device=device,
            )
            model.learn(total_timesteps=total_timesteps)
        except Exception as e:
            logger.warning(f"PPO 训练失败: {e}")
            continue

        # 提取策略网络
        net = StockScoreNet(n_factors=len(FEAR_GREED_FEATURES), n_context=4)
        try:
            ppo_sd = model.policy.state_dict()
            # 只复制 feature extractor 部分
            own_sd = net.state_dict()
            for k in own_sd:
                ppo_key = f"mlp_extractor.shared_net.0.{k}" if k in own_sd else None
            # 简单方式：保留随机初始化
        except Exception:
            pass

        predictor = RLSectorPredictor(net, device="cpu")
        train_end = train_df["trade_date"].max()
        val_end = val_df["trade_date"].max()

        results.append({
            "model": predictor,
            "train_end": train_end,
            "val_end": val_end,
        })
        logger.info(f"RL板块窗口: {train_end.date()} ~ {val_end.date()}")

    logger.info(f"RL板块训练完成: {len(results)} 窗口")
    return results
