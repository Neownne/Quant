"""策略变体数据结构 —— StrategyVariant dataclass + JSON 序列化。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from config.settings import TradingConfig

# ── 默认参数（E4 基线）──
DEFAULTS = {
    "mcap_min": 30.0, "mcap_max": 500.0,
    "price_min": 5.0, "price_max": 63.0,
    "lu_lookback": 20, "lu_count": 1,
    "min_conditions": 4, "min_listed_days": 120,
    "top_n": 5, "stop_loss_pct": 0.08,
    "weight_mode": "equal",
    "ma_fast": 5, "ma_slow": 10,
    "trailing_stop_pct": 0.10,
    "pyramid_threshold": 0.05,
    "pyramid_ratio": 0.5,
    "cooling_days": 3,
}


@dataclass
class StrategyVariant:
    """一个可回测的策略变体。"""
    name: str                         # "E4_baseline"
    family: str = "limit_up"
    description: str = ""
    source: str = "manual"            # "manual" | "web_search" | "grid_search"
    source_url: str = ""
    source_date: str = ""

    # ── 筛选条件 ──
    mcap_min: float | None = None
    mcap_max: float | None = None
    price_min: float | None = None
    price_max: float | None = None
    lu_lookback: int | None = None
    lu_count: int | None = None
    min_conditions: int | None = None
    min_listed_days: int | None = None

    # ── 新增条件（Phase 2）──
    use_volume_filter: bool = False
    volume_ratio_min: float = 1.2
    use_rsi_filter: bool = False
    rsi_min: float = 30.0
    rsi_max: float = 70.0
    ma_fast: int = 5
    ma_slow: int = 10

    # ── 评分/过滤开关 ──
    lu_score: bool = False
    lu_decay: bool = False
    lu_quality: bool = False
    lu_streak: bool = False
    trend_filter: bool = False
    no_5day_streak: bool = False

    # ── 执行参数 ──
    top_n: int = 5
    stop_loss_pct: float = 0.08
    weight_mode: str = "equal"

    # ── 移动止盈（从固定止损 → 跟随最高点）──
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.10   # 从最高点回落 X% 卖出

    # ── 金字塔加仓（盈利>阈值时追加）──
    use_pyramid: bool = False
    pyramid_threshold: float = 0.05   # 盈利 >5% 触发加仓
    pyramid_ratio: float = 0.5        # 加原仓位 50%

    # ── 入场冷却（卖出后 N 天不买回 + 可选收阳确认）──
    use_cooling: bool = False
    cooling_days: int = 3             # 卖出后 N 天内不买回
    require_positive_day: bool = False  # 买入当天必须收阳

    # ── 运行时元数据 ──
    _result: dict | None = field(default=None, repr=False)

    def effective(self, key: str) -> Any:
        """返回参数值（如果为 None 则返回 DEFAULTS 中的默认值）。"""
        val = getattr(self, key, None)
        return val if val is not None else DEFAULTS.get(key)

    def to_gen_signals_args(self) -> list[str]:
        """生成 gen_signals.py CLI 参数列表。"""
        args = []
        # 筛选参数（只传非默认值）
        for key in ["mcap_min", "mcap_max", "price_min", "price_max",
                    "lu_lookback", "lu_count", "min_conditions", "min_listed_days"]:
            val = getattr(self, key)
            if val is not None:
                args.extend([f"--{key.replace('_', '-')}", str(val)])
        # 布尔开关
        for flag in ["lu_score", "lu_decay", "lu_quality", "lu_streak",
                     "trend_filter", "no_5day_streak"]:
            if getattr(self, flag):
                args.append(f"--{flag.replace('_', '-')}")
        # mcap-proxy 总是启用
        args.append("--mcap-proxy")
        return args

    def to_bt_args(self, signals_csv: str) -> list[str]:
        """生成 bt_backtest.py CLI 参数列表。"""
        args = [
            "--top-n", str(self.top_n),
            "--exit-stop", str(self.stop_loss_pct),
            "--signals", signals_csv,
            "--exec-close",
            "--variant-label", self.name,
        ]
        # 移动止盈
        if self.use_trailing_stop:
            args.extend(["--trailing-stop", str(self.trailing_stop_pct)])
        # 金字塔加仓
        if self.use_pyramid:
            args.extend(["--pyramid", str(self.pyramid_threshold), str(self.pyramid_ratio)])
        # 入场冷却
        if self.use_cooling:
            args.append("--cooling")
            args.extend(["--cooling-days", str(self.cooling_days)])
        if self.require_positive_day:
            args.append("--require-positive-day")
        return args

    def validate(self) -> list[str]:
        """验证参数合理范围，返回错误列表。"""
        errors = []
        checks = [
            ("mcap_min", 1, 10000), ("mcap_max", 1, 10000),
            ("price_min", 0.1, 500), ("price_max", 0.1, 500),
            ("lu_lookback", 5, 120), ("lu_count", 0, 20),
            ("min_conditions", 1, 5), ("min_listed_days", 60, 500),
            ("top_n", 1, 50), ("stop_loss_pct", 0.01, 0.30),
        ]
        for key, lo, hi in checks:
            val = getattr(self, key)
            if val is not None and not (lo <= val <= hi):
                errors.append(f"{key}={val} 超出范围 [{lo}, {hi}]")
        if self.mcap_min is not None and self.mcap_max is not None and self.mcap_min >= self.mcap_max:
            errors.append(f"mcap_min({self.mcap_min}) >= mcap_max({self.mcap_max})")
        if self.price_min is not None and self.price_max is not None and self.price_min >= self.price_max:
            errors.append(f"price_min({self.price_min}) >= price_max({self.price_max})")
        return errors

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_result", None)
        return d

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyVariant":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "StrategyVariant":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @staticmethod
    def load_all(variants_dir: str = "lab/variants") -> list["StrategyVariant"]:
        """加载目录下所有 JSON 变体文件。"""
        variants = []
        d = Path(variants_dir)
        if not d.exists():
            return variants
        for f in sorted(d.glob("*.json")):
            try:
                v = StrategyVariant.from_json(str(f))
                variants.append(v)
            except Exception as e:
                print(f"  [WARN] 跳过 {f.name}: {e}")
        return variants


# ── 基线变体 ──
E4_BASELINE = StrategyVariant(
    name="E4_baseline",
    description="E4 基线：4条件去跌停，市值30-500亿，股价5-63元，MA5>MA10，近20日>1次涨停",
    source="manual",
    **{k: v for k, v in DEFAULTS.items() if k in StrategyVariant.__dataclass_fields__},
)
