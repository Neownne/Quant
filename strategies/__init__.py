import importlib
import os
import sys

from strategies.sma_cross import SMACross
from strategies.macd_strategy import MACDStrategy
from strategies.rsi_strategy import RSIStrategy

STRATEGY_REGISTRY = {
    "双均线交叉": SMACross,
    "MACD金叉死叉": MACDStrategy,
    "RSI超买超卖": RSIStrategy,
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
