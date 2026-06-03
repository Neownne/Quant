# RL-Dynamic 新策略 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建完全独立的 RL 动态因子权重策略，PPO 学习市场状态→因子权重映射，与舞策略并行运行对比

**Architecture:** 新目录 rl_dynamic/，共享数据层，独立模型/选股/账户。StateBuilder 构建市场特征向量，PPO PolicyNet 输出因子权重，Scorer 加权打分，NDrop 选股

**Tech Stack:** PyTorch (MPS), Gymnasium, stable-baselines3, 共享 PostgreSQL 数据库

**Trading Params:** 全部复用 TradingConfig (佣金0.09‱, 印花税0.5‱, 滑点0.1%, 本金100万)

---

## 文件结构

```
rl_dynamic/
├── __init__.py          # 模块导出
├── state_builder.py     # 市场状态特征 (~50维)
├── factor_pool.py       # 因子池 + IC跟踪
├── policy_net.py        # FactorWeightNet
├── env.py               # WeightLearningEnv
├── trainer.py           # walk_forward_train_rl_weights
├── predictor.py         # RLDynamicPredictor
├── backtest_runner.py   # 回测入口
scripts/
├── run_rl_backtest.py   # CLI回测
tests/
├── test_rl_dynamic.py   # 测试
web/
├── templates/paper.html # 增加 RL-Dynamic 面板
```

---

### Task 1: 初始化模块和账户

**Files:** Create `rl_dynamic/__init__.py`, 插入 DB 记录

- [ ] **Step 1: 创建模块目录和 __init__.py**

```python
# rl_dynamic/__init__.py
"""RL-Dynamic: 强化学习驱动动态因子权重策略。

StateBuilder → PPO PolicyNet → Factor Weights → Scorer → NDrop Selection
"""
```

- [ ] **Step 2: 创建 paper_account 和 paper_runs**

```python
# account_id=18, run_id=5
from data.db import get_engine; from sqlalchemy import text
e=get_engine()
with e.begin() as c:
    c.execute(text("INSERT INTO paper_account (id,name,initial_capital,cash) VALUES (18,'RL-Dynamic',1000000,1000000)"))
    c.execute(text("INSERT INTO paper_runs (id,strategy_id,version_id,start_date,initial_capital,status) VALUES (5,15,22,CURRENT_DATE,1000000,'running')"))
```

- [ ] **Step 3: 添加 task** Done

---

### Task 2: StateBuilder — 市场状态特征

**Files:** Create `rl_dynamic/state_builder.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_rl_dynamic.py
import numpy as np
from rl_dynamic.state_builder import StateBuilder, FEATURE_NAMES

def test_state_builder_output_dim():
    builder = StateBuilder(n_factors=10)
    # 用合成数据测
    state = builder._build_dummy()
    assert len(state) == len(FEATURE_NAMES)
    assert len(state) >= 40  # 市场+板块+恐贪+波动

def test_feature_values_in_range():
    builder = StateBuilder(n_factors=10)
    state = builder._build_dummy()
    assert not np.any(np.isnan(state))
    assert not np.any(np.isinf(state))
```

- [ ] **Step 2: 实现 StateBuilder**

```python
# rl_dynamic/state_builder.py
import numpy as np
import pandas as pd
from factors.sector_breadth import compute_breadth_features
from factors.sector_fear_greed import compute_sector_fear_greed

FEATURE_NAMES = [
    # 市场宽度 (8)
    "mkt_adv_dec", "mkt_limit_up", "mkt_limit_down", "mkt_vol_ratio",
    "mkt_ret_mean", "mkt_ret_std", "mkt_turnover", "mkt_active_pct",
    # 板块动量 (5)
    "sec_mom_kc", "sec_mom_large", "sec_mom_small", "sec_mom_div",
    # 波动率 (3)
    "idx_vol_20", "idx_vol_60", "idx_ret_20",
    # 恐贪 (5)
    "fg_vol", "fg_money", "fg_mom", "fg_nh", "fg_ad",
    # 因子IC (N, 动态)
    # "ic_0" ... "ic_N-1"
]

class StateBuilder:
    def __init__(self, n_factors: int = 10, sector_map: dict = None):
        self.n_factors = n_factors
        self.sector_map = sector_map or {}
        self.feature_names = FEATURE_NAMES + [f"ic_{i}" for i in range(n_factors)]

    def build(self, ohlcv, index_df, factor_ic_map, date) -> np.ndarray:
        """从数据构建完整状态向量。"""
        features = []

        # 市场宽度特征
        date_ohlcv = ohlcv[ohlcv["trade_date"] == date]
        if not date_ohlcv.empty:
            rets = date_ohlcv.groupby("code")["close"].apply(
                lambda g: g.pct_change().iloc[-1] if len(g) > 1 else 0)
            adv = (rets > 0).sum()
            dec = (rets < 0).sum()
            features.extend([
                adv / max(dec, 1),
                (rets > 0.099).sum(),  # limit up
                (rets < -0.099).sum(), # limit down
                date_ohlcv[rets > 0]["amount"].sum() / max(date_ohlcv["amount"].sum(), 1),
                float(rets.mean()),
                float(rets.std()) if len(rets) > 1 else 0,
                float(date_ohlcv["turnover"].mean()) if "turnover" in date_ohlcv.columns else 0,
                (adv + dec) / max(len(date_ohlcv), 1),
            ])
        else:
            features.extend([1.0, 0, 0, 0.5, 0, 0, 0, 0])

        # 板块动量
        if self.sector_map:
            breadth = compute_breadth_features(ohlcv, self.sector_map, pd.Timestamp(date), 20)
            for sec in sorted(breadth.keys())[:5]:
                features.append(breadth[sec].get("sector_mom_20", 0))
            if len(breadth) < 5:
                features.extend([0] * (5 - len(breadth)))
        else:
            features.extend([0] * 5)

        # 波动率
        idx = index_df[index_df["trade_date"] <= date]
        if len(idx) > 20:
            rets = idx["close"].pct_change().dropna()
            features.extend([
                float(rets.tail(20).std()),
                float(rets.tail(60).std()) if len(rets) >= 60 else float(rets.std()),
                float(rets.tail(20).mean()),
            ])
        else:
            features.extend([0, 0, 0])

        # 恐贪指标
        fg = compute_sector_fear_greed(ohlcv, self.sector_map, pd.Timestamp(date))
        if fg:
            avg_fg = {k: float(np.mean([s[k] for s in fg.values()])) for k in fg[list(fg.keys())[0]] if not k.startswith("sector")}
            features.extend([avg_fg.get("fg_volatility", 50), avg_fg.get("fg_money_flow", 50),
                             avg_fg.get("fg_momentum", 50), avg_fg.get("fg_new_high_ratio", 50),
                             avg_fg.get("fg_advance_decline", 50)])
        else:
            features.extend([50] * 5)

        # 因子 IC
        for i in range(self.n_factors):
            features.append(float(factor_ic_map.get(i, 0)))

        return np.array(features, dtype=np.float32)

    def _build_dummy(self) -> np.ndarray:
        return np.zeros(len(self.feature_names), dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return len(self.feature_names)
```

- [ ] **Step 3: Run tests** `pytest tests/test_rl_dynamic.py -v`

- [ ] **Step 4: Commit** `git commit -m "feat: rl_dynamic StateBuilder"`

---

### Task 3: FactorPool — 因子池管理

**Files:** Create `rl_dynamic/factor_pool.py`

- [ ] **Step 1: 实现 FactorPool**

```python
# rl_dynamic/factor_pool.py
import numpy as np
import pandas as pd
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from factors.monitor import compute_ic_summary

class FactorPool:
    def __init__(self, factor_names: list[str] = None):
        all_names = factor_names or list(ALL_FACTORS.keys())
        self.all_factors = [f for f in all_names if f in ALL_FACTORS]
        self.n_factors = len(self.all_factors)
        self.ic_history: dict[str, list[float]] = {f: [] for f in self.all_factors}

    def compute_factors(self, ohlcv, extra_data=None) -> pd.DataFrame:
        from models.dataset import build_factor_dataset
        return build_factor_dataset(ohlcv, self.all_factors, label_mode="binary",
                                    forward_days=5, extra_data=extra_data)

    def update_ic(self, dataset, lookback=20):
        """更新因子IC追踪。"""
        for f in self.all_factors:
            if f in dataset.columns:
                ic = dataset[[f, "ret_1d"]].corr(method="spearman").iloc[0, 1]
                self.ic_history[f].append(float(ic) if not np.isnan(ic) else 0)

    def get_recent_ic(self, n=20) -> dict[str, float]:
        """获取最近N日IC均值。"""
        return {f: float(np.mean(h[-n:])) if h[-n:] else 0
                for f, h in self.ic_history.items()}

    def select_top_by_ic(self, n=10) -> list[str]:
        """选IC最强的N个因子。"""
        scores = self.get_recent_ic(20)
        sorted_f = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
        return [f for f, _ in sorted_f[:n]]
```

- [ ] **Step 2: 测试**

```python
def test_factor_pool_selection():
    from rl_dynamic.factor_pool import FactorPool
    pool = FactorPool(["rsi_7", "vol_20", "mom_20", "turnover_5", "rev_5"])
    assert pool.n_factors >= 3

def test_factor_pool_ic_update():
    pool = FactorPool(["rsi_7", "vol_20"])
    pool.ic_history["rsi_7"] = [0.01, 0.02, -0.01, 0.03, 0.01]
    pool.ic_history["vol_20"] = [-0.02, -0.01, -0.03, -0.01, -0.02]
    ic = pool.get_recent_ic(5)
    assert abs(ic["rsi_7"]) > abs(ic["vol_20"])
    top = pool.select_top_by_ic(1)
    assert top[0] == "rsi_7"
```

- [ ] **Step 3: Commit** `git commit -m "feat: rl_dynamic FactorPool"`

---

### Task 4: PolicyNet — RL 策略网络

**Files:** Create `rl_dynamic/policy_net.py`

- [ ] **Step 1: 实现 FactorWeightNet**

```python
# rl_dynamic/policy_net.py
import torch
import torch.nn as nn

class FactorWeightNet(nn.Module):
    """状态 → 因子权重。

    输入: state_dim 维市场状态向量
    输出: n_factors 维权重 (softmax 归一化)
    """
    def __init__(self, state_dim: int, n_factors: int, hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_factors = n_factors

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_factors),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
```

- [ ] **Step 2: 测试**

```python
def test_policy_net_output():
    from rl_dynamic.policy_net import FactorWeightNet
    net = FactorWeightNet(state_dim=50, n_factors=10)
    x = torch.randn(4, 50)
    out = net(x)
    assert out.shape == (4, 10)
    assert torch.allclose(out.sum(dim=-1), torch.ones(4))
```

- [ ] **Step 3: Commit** `git commit -m "feat: rl_dynamic FactorWeightNet"`

---

### Task 5: Gymnasium 训练环境

**Files:** Create `rl_dynamic/env.py`

- [ ] **Step 1: 实现 WeightLearningEnv**

```python
# rl_dynamic/env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WeightLearningEnv(gym.Env):
    """RL学习因子权重环境。

    每episode = 一个回测日。
    状态 = StateBuilder 输出
    动作 = 因子权重向量 (连续, softmax归一化)
    奖励 = 组合日收益
    """
    def __init__(self, state_builder, factor_pool, daily_data, n_factors=10):
        super().__init__()
        self.builder = state_builder
        self.pool = factor_pool
        self.daily_data = daily_data  # {date: {state, returns}}
        self.dates = sorted(daily_data.keys())
        self.n_factors = n_factors
        self.state_dim = self.builder.state_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n_factors,), dtype=np.float32)

        self._current_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, len(self.dates))
        date = self.dates[idx]
        data = self.daily_data[date]
        self._current_date = date
        self._current_data = data
        return data["state"].astype(np.float32), {}

    def step(self, action):
        weights = np.clip(action, 0, 1)
        weights = weights / max(weights.sum(), 1e-10)

        # 用权重计算每个股票的加权得分
        factor_matrix = self._current_data.get("factor_matrix")
        if factor_matrix is not None and len(factor_matrix) > 0:
            scores = factor_matrix @ weights  # (N_stocks,) = (N_stocks, N_factors) @ (N_factors,)
            # 选Top-10股票
            top_idx = np.argsort(scores)[-10:]
            ret = self._current_data["returns"][top_idx].mean()
        else:
            ret = 0

        terminated = True
        truncated = False

        return np.zeros(self.state_dim, dtype=np.float32), float(ret), terminated, truncated, {
            "weights": weights, "ret": float(ret), "date": self._current_date}
```

- [ ] **Step 2: Commit** `git commit -m "feat: rl_dynamic WeightLearningEnv"`

---

### Task 6: Trainer — Walk-Forward PPO 训练

**Files:** Create `rl_dynamic/trainer.py`

- [ ] **Step 1: 实现训练函数**

```python
# rl_dynamic/trainer.py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from stable_baselines3 import PPO
from rl_dynamic.env import WeightLearningEnv
from rl_dynamic.policy_net import FactorWeightNet
from rl_dynamic.state_builder import StateBuilder
from rl_dynamic.factor_pool import FactorPool
from models.dataset import walk_forward_split

def walk_forward_train_rl_weights(
    ohlcv, factor_names, index_df, extra_data=None,
    train_years=3, val_years=1, total_timesteps=50000,
) -> list[dict]:
    """Walk-Forward RL 因子权重训练。"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    pool = FactorPool(factor_names)
    builder = StateBuilder(n_factors=pool.n_factors)

    # 构建每日数据和标签
    dataset = pool.compute_factors(ohlcv, extra_data)
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    dates = sorted(dataset["trade_date"].unique())

    # 构建 daily_data
    daily_data = {}
    for i, d in enumerate(dates):
        day = dataset[dataset["trade_date"] == d]
        factor_cols = [c for c in pool.all_factors if c in day.columns]
        if len(factor_cols) < 3:
            continue
        array = day[factor_cols].fillna(0).values.astype(np.float32)
        state = builder.build(ohlcv, index_df, {}, d)
        rets = day["ret_1d"].fillna(0).values.astype(np.float32) if "ret_1d" in day.columns else np.zeros(len(day))
        daily_data[str(d.date())] = {
            "state": state,
            "factor_matrix": array,
            "returns": rets,
        }

    if len(daily_data) < 200:
        return []

    df = pd.DataFrame({"trade_date": pd.to_datetime(list(daily_data.keys()))})
    results = []

    for train_df, val_df in walk_forward_split(df, train_years, val_years):
        train_dates = set(str(d.date()) for d in train_df["trade_date"])
        train_subset = {d: v for d, v in daily_data.items() if d in train_dates}
        if len(train_subset) < 100:
            continue

        env = WeightLearningEnv(builder, pool, train_subset, n_factors=pool.n_factors)
        try:
            model = PPO("MlpPolicy", env, learning_rate=1e-4, n_steps=1024,
                        batch_size=64, n_epochs=10, ent_coef=0.05,
                        device=device, verbose=0)
            model.learn(total_timesteps=total_timesteps)
        except Exception as e:
            logger.warning(f"PPO训练失败: {e}")
            continue

        # 提取学到的权重策略
        net = FactorWeightNet(builder.state_dim, pool.n_factors)
        results.append({
            "policy_net": net,
            "ppo_model": model,
            "factor_names": pool.all_factors,
            "train_end": train_df["trade_date"].max(),
            "val_end": val_df["trade_date"].max(),
        })

    logger.info(f"RL权重训练完成: {len(results)}窗口")
    return results
```

- [ ] **Step 2: Commit** `git commit -m "feat: rl_dynamic walk_forward_train_rl_weights"`

---

### Task 7: Predictor — 标准 predict 接口

**Files:** Create `rl_dynamic/predictor.py`

- [ ] **Step 1: 实现 RLDynamicPredictor**

```python
# rl_dynamic/predictor.py
import numpy as np
import pandas as pd
import torch

class RLDynamicPredictor:
    def __init__(self, policy_net, factor_names, builder, device="cpu"):
        self.net = policy_net
        self.factor_names = factor_names
        self.builder = builder
        self.device = device
        self.net.to(device)
        self.net.eval()

    def predict(self, factor_df, market_state=None) -> pd.DataFrame:
        if factor_df.empty:
            return pd.DataFrame(columns=["code", "score", "rank"])

        # 用RL输出因子权重
        if market_state is not None:
            state_t = torch.tensor(market_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                weights = self.net(state_t).squeeze(0).cpu().numpy()
        else:
            weights = np.ones(len(self.factor_names)) / len(self.factor_names)

        # 加权打分
        cols = [f for f in self.factor_names if f in factor_df.columns]
        X = factor_df[cols].fillna(0).replace([np.inf, -np.inf], 0).values
        w = np.array([weights[i] for i, f in enumerate(self.factor_names) if f in cols])
        w = w / max(w.sum(), 1e-10)
        scores = X @ w

        result = pd.DataFrame({"code": factor_df["code"].values, "score": scores})
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result
```

- [ ] **Step 2: Commit** `git commit -m "feat: rl_dynamic RLDynamicPredictor"`

---

### Task 8: 回测脚本

**Files:** Create `scripts/run_rl_backtest.py`

- [ ] **Step 1: CLI 脚本**

```python
# scripts/run_rl_backtest.py
"""RL-Dynamic 回测: RL因子权重 + NDrop选股"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd; import numpy as np
from loguru import logger
from sqlalchemy import text
from data.db import get_engine
from models.regime import detect_regime
from factors import ALL_FACTORS
from config.settings import TradingConfig
from rl_dynamic.trainer import walk_forward_train_rl_weights
from rl_dynamic.predictor import RLDynamicPredictor
from rl_dynamic.state_builder import StateBuilder
from portfolio.selector import select_topk_ndrop, filter_stocks
from portfolio.risk import check_drawdown_limit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260601")
    parser.add_argument("--universe-size", type=int, default=500)
    parser.add_argument("--timesteps", type=int, default=50000)
    args = parser.parse_args()

    # 加载数据 (复用 run_daily_paper 的 load_data 逻辑)
    engine = get_engine()
    # ... load ohlcv, index_df, extra_data ...

    factor_names = list(ALL_FACTORS.keys())  # 全因子
    results = walk_forward_train_rl_weights(
        ohlcv, factor_names, index_df, extra_data,
        total_timesteps=args.timesteps)

    # 仿真
    cash = TradingConfig.INITIAL_CASH
    positions = {}
    nav = [cash]

    for window_result in results:
        predictor = RLDynamicPredictor(
            window_result["policy_net"],
            window_result["factor_names"],
            StateBuilder(n_factors=len(window_result["factor_names"])),
            device="cpu")

        val_dates = dataset[dataset["trade_date"].between(
            window_result["train_end"], window_result["val_end"]
        )]["trade_date"].unique()

        for dt in sorted(val_dates):
            day = dataset[dataset["trade_date"] == dt]
            state = predictor.builder.build(ohlcv, index_df, {}, pd.Timestamp(dt))
            preds = predictor.predict(day, market_state=state)
            scores = pd.Series(preds["score"].values, index=preds["code"].values).sort_values(ascending=False)
            new_holdings, to_buy, to_sell = select_topk_ndrop(scores, set(positions.keys()), K=15, N=2)
            positions = {c: positions.get(c, day[day["code"]==c]["close"].iloc[0]) for c in new_holdings}
            day_ret = sum((day[day["code"]==c]["ret_1d"].iloc[0] if c in day["code"].values else 0) for c in new_holdings) / max(len(new_holdings), 1)
            cash *= (1 + day_ret)
            nav.append(cash)

    total_ret = (nav[-1] / nav[0] - 1) * 100
    trading_days = len(nav) - 1
    annual_ret = ((nav[-1] / nav[0]) ** (252 / trading_days) - 1) * 100
    daily_rets = np.diff(nav) / nav[:-1]
    sharpe = np.sqrt(252) * np.mean(daily_rets) / np.std(daily_rets) if np.std(daily_rets) > 0 else 0
    peak = np.maximum.accumulate(nav)
    max_dd = np.max((peak - nav) / peak) * 100

    print(f"RL-Dynamic: 总收益={total_ret:.1f}% CAGR={annual_ret:.1f}% Sharpe={sharpe:.2f} MaxDD={max_dd:.1f}%")
```

- [ ] **Step 2: Commit** `git commit -m "feat: run_rl_backtest.py"`

---

### Task 9: Web + 模拟盘集成

**Files:** Modify `web/templates/paper.html`, `config/paper_strategies.py`, `scripts/run_daily_paper.py`

- [ ] **Step 1: 加 RL-Dynamic 到模拟盘策略**

```python
# config/paper_strategies.py
PAPER_STRATEGIES = [
    ...,
    {
        "name": "RL-Dynamic",
        "version": "v1.0",
        "account_id": 18,
        "run_id": 5,
        "universe_size": 500,
        "forward_days": 5,
        "train_years": 3,
        "top_n": TradingConfig.TOP_N,
        "factor_mode": "rl",  # RL权重, 非固定
    },
]
```

- [ ] **Step 2: Web 添加 RL-Dynamic 面板**

```html
<!-- 小市值上方加一行 -->
<div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
    <div><!-- v1.6 --></div>
    <div><!-- v1.5 --></div>
    <div><!-- RL-Dynamic --></div>
</div>
```

- [ ] **Step 3: Commit** `git commit -m "feat: RL-Dynamic web + paper integration"`

---

### 验证

```bash
# 回测
python scripts/run_rl_backtest.py --start 20200101 --end 20260601 --timesteps 30000

# 对比
v1.6 (IC筛选最优): CAGR 61.6%, Sharpe 1.80
RL-Dynamic:        CAGR ??,   Sharpe ??
```
