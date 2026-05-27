import numpy as np


class OverfitChecker:
    def __init__(self, min_trades: int = 30, min_regimes: int = 2,
                 oos_ratio_threshold: float = 0.3):
        self.min_trades = min_trades
        self.min_regimes = min_regimes
        self.oos_ratio_threshold = oos_ratio_threshold

    def check(self, metrics: dict, regime_count: int = 0,
              sensitivity_stable: bool = True) -> dict:
        flags = []
        train_sr = metrics.get("train_sharpe", 0)
        val_sr = metrics.get("val_sharpe", 0)
        test_sr = metrics.get("test_sharpe", 0)
        n_trades = metrics.get("n_trades", 0)
        n_params = max(metrics.get("n_params", 1), 1)

        if val_sr > 0 and train_sr > 0 and (val_sr / train_sr) < self.oos_ratio_threshold:
            flags.append(f"样本外一致性严重不足: val/train夏普比={val_sr/train_sr:.2f}")
        if n_trades < self.min_trades:
            flags.append(f"交易次数不足: {n_trades}笔 < {self.min_trades}笔")
        if regime_count < self.min_regimes:
            flags.append(f"覆盖市场时段不足: {regime_count}种 < {self.min_regimes}种")
        if not sensitivity_stable:
            flags.append("参数敏感性过高: ±10%→结果波动>20%")

        adjusted_sharpe = test_sr * np.sqrt(n_trades / (n_params + 1))

        quality = "valid"
        if flags:
            quality = "suspect"
        if test_sr < 0:
            quality = "invalid"

        return {
            "quality": quality,
            "flags": flags,
            "adjusted_sharpe": round(float(adjusted_sharpe), 4),
        }
