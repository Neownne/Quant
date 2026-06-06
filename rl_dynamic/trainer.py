"""Walk-Forward GRPO 因子权重训练。

v2: PPO → GRPO, 加入因子筛选(IC+正交), 训练窗口 5年/1年。
"""
import numpy as np
import pandas as pd
import torch
from loguru import logger
from models.dataset import walk_forward_split
from rl_dynamic.state_builder import StateBuilder
from rl_dynamic.factor_pool import FactorPool
from rl_dynamic.grpo_trainer import DirichletPolicyNet, train_grpo_weights, GRPOPredictor


def _build_daily_data(dataset, builder, pool, ohlcv, index_df):
    """构建 RL 环境需要的每日数据字典。

    按日期逐一计算截面 IC，用滚动平均值去前视偏差。
    pool.all_factors 决定使用哪些因子列。
    """
    dates = sorted(dataset["trade_date"].unique())
    factor_cols = [c for c in pool.all_factors if c in dataset.columns]
    if not factor_cols:
        factor_cols = pool.all_factors

    # 按日期追踪每个因子的滚动 IC
    rolling_ic = {f: [] for f in factor_cols}

    daily_data = {}
    for d in dates:
        day = dataset[dataset["trade_date"] == d]
        if len(day) < 10:
            continue

        # 当日截面 IC
        ret_col = "ret_1d" if "ret_1d" in day.columns else None
        for f in factor_cols:
            if f not in day.columns or ret_col is None:
                continue
            sub = day[[f, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) >= 10 and sub[f].nunique() > 1:
                ic = sub[f].corr(sub[ret_col], method="spearman")
                rolling_ic[f].append(float(ic) if not np.isnan(ic) else 0.0)
            else:
                rolling_ic[f].append(0.0)

        # 滚动平均 IC map (最近20日)
        ic_map = {}
        for i, f in enumerate(factor_cols):
            hist = rolling_ic[f][-20:]
            ic_map[i] = float(np.mean(hist)) if hist else 0.0

        state = builder.build(ohlcv, index_df, ic_map, d)
        matrix = day[factor_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        rets = day[ret_col].fillna(0).values.astype(np.float32) if ret_col else np.zeros(len(day), dtype=np.float32)
        daily_data[str(pd.Timestamp(d).date())] = {
            "state": state, "factor_matrix": matrix, "returns": rets,
        }
    return daily_data


def _screen_factors(dataset, factor_names, max_factors=15, ic_threshold=0.02):
    """因子筛选：IC 门禁 + 正交贪心。

    Args:
        dataset: build_factor_dataset 输出 (含 factor_names 列 + ret_1d)
        factor_names: 候选因子列表
        max_factors: 最多保留因子数
        ic_threshold: IC 绝对值阈值

    Returns:
        selected: 筛选后的因子名列表 (最少保证 3 个)
    """
    from factors.screening import filter_factors_by_ic, select_orthogonal_factors

    n_before = len(factor_names)

    # Step 1: IC gate
    passing = filter_factors_by_ic(
        dataset, factor_names, ret_col="ret_1d",
        ic_threshold=ic_threshold, t_threshold=2.0,
    )

    if len(passing) < 3:
        # 放宽阈值重试
        passing = filter_factors_by_ic(
            dataset, factor_names, ret_col="ret_1d",
            ic_threshold=0.01, t_threshold=1.5,
        )

    if len(passing) < 3:
        logger.warning(f"IC筛选后仅{len(passing)}个因子，使用原始top-{max_factors}")
        return factor_names[:max_factors]

    # Step 2: 正交筛选
    selected = select_orthogonal_factors(
        dataset, passing, threshold=0.7, ic_summary=None,
    )

    selected = selected[:max_factors]
    if len(selected) < 3:
        selected = passing[:max_factors]

    logger.info(f"因子筛选: {n_before} → IC{len(passing)} → 正交{len(selected)}")
    return selected


def walk_forward_train_rl_weights(
    ohlcv: pd.DataFrame, factor_names: list[str],
    index_df: pd.DataFrame, extra_data=None,
    train_years: int = 5, val_years: int = 1,
    total_timesteps: int = 50000,
    screen_factors: bool = True,
    max_factors: int = 15,
    concept_features=None,
) -> list[dict]:
    """Walk-Forward GRPO 因子权重训练。

    Args:
        ohlcv: OHLCV 日频数据
        factor_names: 候选因子名列表
        index_df: 指数日频数据
        extra_data: 额外数据 (基本面/估值等)
        train_years: 训练窗口年数 (默认5)
        val_years: 验证窗口年数 (默认1)
        total_timesteps: 总训练步数 (GRPO 中用作 epochs)
        screen_factors: 是否做因子筛选
        max_factors: 筛选后最多保留因子数
        concept_features: ConceptBoardFeatures 实例或 None

    Returns:
        [{policy, predictor, factor_names, builder, state_dim, n_factors, train_end, val_end}, ...]
    """
    device = "cpu"
    logger.info(f"GRPO权重训练设备: {device}, train_years={train_years}, val_years={val_years}")

    # ── 1. 计算全量因子数据集 ──
    pool = FactorPool(factor_names)
    dataset = pool.compute_factors(ohlcv, extra_data)
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])

    # ── 2. 因子筛选 ──
    if screen_factors and len(factor_names) > max_factors:
        selected = _screen_factors(dataset, factor_names, max_factors=max_factors)
    else:
        selected = factor_names[:max_factors]

    pool = FactorPool(selected)
    builder = StateBuilder(n_factors=pool.n_factors, concept_features=concept_features)

    # ── 3. 构建每日数据 ──
    try:
        daily_data = _build_daily_data(dataset, builder, pool, ohlcv, index_df)
    except Exception as e:
        logger.error(f"构建每日数据失败: {e}")
        import traceback; traceback.print_exc()
        return []
    if len(daily_data) < 50:
        logger.error(f"训练数据不足: {len(daily_data)}天")
        return []

    # ── 4. Walk-Forward 训练 ──
    df = pd.DataFrame({"trade_date": pd.to_datetime(list(daily_data.keys()))})
    results = []
    windows = list(walk_forward_split(df, train_years, val_years))

    for wi, (train_df, val_df) in enumerate(windows):
        train_dates = {str(d.date()) for d in train_df["trade_date"]}
        train_subset = {d: v for d, v in daily_data.items() if d in train_dates}
        if len(train_subset) < 200:
            logger.warning(f"窗口{wi+1}训练数据不足({len(train_subset)}天), 跳过")
            continue

        # 初始化策略网络
        policy = DirichletPolicyNet(
            state_dim=builder.state_dim, n_factors=pool.n_factors,
        ).to(device)

        # GRPO 训练
        epochs = max(5, min(total_timesteps, len(train_subset) // 10))
        try:
            policy = train_grpo_weights(
                policy, train_subset, selected,
                epochs=epochs, M=8, top_k=10,
                lr=5e-3, epsilon=0.2, entropy_coef=0.005,
                device=device,
            )
        except Exception as e:
            logger.warning(f"GRPO训练失败 (窗口{wi+1}): {e}")
            import traceback; traceback.print_exc()
            continue

        predictor = GRPOPredictor(policy, selected, builder, device=device)

        train_end = train_df["trade_date"].max()
        val_end = val_df["trade_date"].max()
        logger.info(f"GRPO窗口{wi+1}/{len(windows)}: "
                    f"{train_df['trade_date'].min().date()} ~ {val_end.date()} "
                    f"(训练{len(train_subset)}天, 预期验证{len(val_df)}天)")

        results.append({
            "policy": policy,
            "predictor": predictor,
            "builder": builder,
            "factor_names": selected,
            "state_dim": builder.state_dim,
            "n_factors": pool.n_factors,
            "train_end": train_end,
            "val_end": val_end,
        })

    logger.info(f"GRPO权重训练完成: {len(results)}/{len(windows)}窗口")
    return results
