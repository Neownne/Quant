import importlib
import os
import sys

from strategies.sma_cross import SMACross
from strategies.macd_strategy import MACDStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.grid_shock import GridShockStrategy

STRATEGY_REGISTRY = {
    "双均线交叉": SMACross,
    "MACD金叉死叉": MACDStrategy,
    "RSI超买超卖": RSIStrategy,
    "震荡网格(高抛低吸)": GridShockStrategy,
}

# 自定义策略目录
_CUSTOM_DIR = os.path.expanduser("~/.quant_strategies")


def load_custom_strategies() -> dict[str, type]:
    """扫描 ~/.quant_strategies/ 目录，动态加载自定义策略。"""
    custom = {}
    if not os.path.isdir(_CUSTOM_DIR):
        return custom

    if _CUSTOM_DIR not in sys.path:
        sys.path.insert(0, _CUSTOM_DIR)

    for fname in sorted(os.listdir(_CUSTOM_DIR)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        module_name = fname[:-3]
        try:
            mod = importlib.import_module(module_name)
            importlib.reload(mod)
            if hasattr(mod, "MyStrategy"):
                cls = getattr(mod, "MyStrategy")
                display = getattr(cls, "_display_name", module_name.replace("_", " ").title())
                custom[display] = cls
        except Exception:
            pass

    return custom


def get_all_strategies() -> dict[str, type]:
    """内置 + 自定义策略全集。"""
    all_s = dict(STRATEGY_REGISTRY)
    all_s.update(load_custom_strategies())
    return all_s


def list_all_strategies() -> dict[str, dict]:
    """统一策略列表：静态策略 + ML 策略。

    返回格式: {display_name: {"type": "static"|"ml", ...}}
    - 静态: {"type": "static", "class": StrategyClass}
    - ML:   {"type": "ml", "config": {...config_dict...}}
    """
    unified = {}

    # 静态策略
    for name, cls in get_all_strategies().items():
        unified[name] = {"type": "static", "class": cls}

    # ML 策略
    # [架构重构] app.utils 已移除，ML 策略列表暂不可用
    try:
        from app.utils.ml_config_manager import list_ml_configs  # noqa: F401 (removed in v2.0)
        for _, row in list_ml_configs().iterrows():
            unified[f"ML: {row['name']}"] = {"type": "ml", "config": row.to_dict()}
    except Exception:
        pass  # DB 不可用时跳过

    return unified


def is_ml_strategy(item: dict) -> bool:
    """判断策略是否为 ML 类型。"""
    return isinstance(item, dict) and item.get("type") == "ml"


def is_static_strategy(item: dict) -> bool:
    """判断策略是否为静态类型。"""
    return isinstance(item, dict) and item.get("type") == "static"
