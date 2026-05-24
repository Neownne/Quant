import backtrader as bt


class MACDStrategy(bt.Strategy):
    """MACD 金叉死叉策略：DIF 上穿 DEA 买入，下穿卖出。"""

    params = (
        ("fast", 12),
        ("slow", 26),
        ("signal", 9),
    )

    def __init__(self):
        self.macd = bt.ind.MACD(
            period_me1=self.params.fast,
            period_me2=self.params.slow,
            period_signal=self.params.signal,
        )
        self.crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()
