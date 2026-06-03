"""Walk-Forward RL 因子权重训练。"""
import numpy as np
import pandas as pd
import torch
from loguru import logger
from models.dataset import walk_forward_split
from rl_dynamic.env import WeightLearningEnv
from rl_dynamic.policy_net import FactorWeightNet
from rl_dynamic.state_builder import StateBuilder
from rl_dynamic.factor_pool import FactorPool


def _build_daily_data(dataset, builder, pool, ohlcv, index_df):
    """构建 RL 环境需要的每日数据字典（预计算避免重复IO）。"""
    dates = sorted(dataset["trade_date"].unique())
    try:
        valid = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_1d"])
        if len(valid) > 100:
            pool.update_ic(valid)
    except Exception:
        pass
    ic_map = pool.get_recent_ic(20)
    factor_cols = [c for c in pool.all_factors if c in dataset.columns]
    if not factor_cols:
        factor_cols = pool.all_factors[:10]

    daily_data = {}
    for d in dates:
        day = dataset[dataset["trade_date"] == d]
        if len(day) < 10:
            continue
        state = builder.build(ohlcv, index_df, ic_map, d)
        matrix = day[factor_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        rets = day["ret_1d"].fillna(0).values.astype(np.float32) if "ret_1d" in day.columns else np.zeros(len(day), dtype=np.float32)
        daily_data[str(pd.Timestamp(d).date())] = {
            "state": state, "factor_matrix": matrix, "returns": rets,
        }
    return daily_data


def walk_forward_train_rl_weights(
    ohlcv: pd.DataFrame, factor_names: list[str],
    index_df: pd.DataFrame, extra_data=None,
    train_years: int = 3, val_years: int = 1,
    total_timesteps: int = 50000,
) -> list[dict]:
    """Walk-Forward RL 因子权重训练。

    Returns: [{policy_net, factor_names, train_end, val_end}, ...]
    """
    device = "cpu"  # PPO with MlpPolicy works best on CPU
    logger.info(f"RL权重训练设备: {device}")

    pool = FactorPool(factor_names)
    builder = StateBuilder(n_factors=pool.n_factors)

    dataset = pool.compute_factors(ohlcv, extra_data)
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])

    try:
        daily_data = _build_daily_data(dataset, builder, pool, ohlcv, index_df)
    except Exception as e:
        logger.error(f"构建每日数据失败: {e}")
        import traceback; traceback.print_exc()
        return []
    if len(daily_data) < 50:
        logger.error(f"训练数据不足: {len(daily_data)}天")
        return []

    df = pd.DataFrame({"trade_date": pd.to_datetime(list(daily_data.keys()))})
    results = []

    for train_df, val_df in walk_forward_split(df, train_years, val_years):
        train_dates = {str(d.date()) for d in train_df["trade_date"]}
        train_subset = {d: v for d, v in daily_data.items() if d in train_dates}
        if len(train_subset) < 100:
            continue

        env = WeightLearningEnv(builder, pool, train_subset, n_factors=pool.n_factors)

        try:
            from stable_baselines3 import PPO
            model = PPO("MlpPolicy", env, learning_rate=1e-4, n_steps=1024,
                        batch_size=64, n_epochs=10, ent_coef=0.05,
                        device=device, verbose=0)
            model.learn(total_timesteps=min(total_timesteps, len(train_subset) * 10))
        except Exception as e:
            logger.warning(f"PPO训练失败: {e}")
            continue

        # 从 PPO 提取学到的策略 → FactorWeightNet
        net = FactorWeightNet(builder.state_dim, pool.n_factors)
        try:
            ppo_sd = model.policy.state_dict()
            # 只复制 feature extractor 的共享层
            own_sd = net.state_dict()
            for key in own_sd:
                # 尝试匹配 PPO 的 mlp_extractor 或 policy_net
                for ppo_key in [
                    f"mlp_extractor.policy_net.{key}",
                    f"mlp_extractor.shared_net.{key}",
                ]:
                    if ppo_key in ppo_sd and own_sd[key].shape == ppo_sd[ppo_key].shape:
                        own_sd[key] = ppo_sd[ppo_key]
                        break
            net.load_state_dict(own_sd, strict=False)
        except Exception:
            pass  # 回退到随机初始化
        net.to(device)

        results.append({
            "policy_net": net,
            "ppo_model": model,  # 保留PPO模型备用
            "factor_names": pool.get_factor_names(),
            "state_dim": builder.state_dim,
            "train_end": train_df["trade_date"].max(),
            "val_end": val_df["trade_date"].max(),
        })
        logger.info(f"RL窗口: {train_df['trade_date'].min().date()} ~ {val_df['trade_date'].max().date()}")

    logger.info(f"RL权重训练完成: {len(results)}窗口")
    return results
