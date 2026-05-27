"""Window scaling for multi-bar-per-day frequencies.

When bar_per_day > 1 (e.g., 60min data has 4 bars/day), factor lookback
windows are scaled so they cover the same calendar-time window as daily
factors.  RSI(14) on daily → RSI(56) on 60min = same 14 calendar days.
"""

_WINDOW_SCALE: float = 1.0


def w(n: int) -> int:
    """Scale a lookback window by the global _WINDOW_SCALE."""
    return max(1, round(n * _WINDOW_SCALE))


def set_window_scale(scale: float) -> None:
    global _WINDOW_SCALE
    _WINDOW_SCALE = float(scale)


def get_window_scale() -> float:
    return _WINDOW_SCALE
