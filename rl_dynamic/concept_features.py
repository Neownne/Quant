"""概念板块特征预计算 — Top-5 热点板块动量 + 全局宽度/轮动。

从 DB 加载 concept_board + concept_stock 映射，基于 OHLCV 计算
每日板块等权收益，缓存 8 维特征供 StateBuilder 使用：

  cb_hot1_5d ~ cb_hot5_5d : 成交额 Top-5 板块的近 5 日动量
  cb_breadth              : 全板块中 5 日收益 > 0 的占比
  cb_vol                  : 全板块 20 日收益波动率均值
  cb_rotation             : 板块 5 日收益的横截面标准差 (轮动强度)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text


class ConceptBoardFeatures:
    """预计算 + 缓存概念板块日频特征。

    用法:
        cbf = ConceptBoardFeatures(ohlcv, engine)
        features = cbf.get_features(pd.Timestamp("2025-03-15"))
        # → {"cb_hot1_5d": 0.032, "cb_hot2_5d": 0.018, ...}
    """

    _defaults: dict[str, float] = {
        "cb_hot1_5d": 0.0,
        "cb_hot2_5d": 0.0,
        "cb_hot3_5d": 0.0,
        "cb_hot4_5d": 0.0,
        "cb_hot5_5d": 0.0,
        "cb_breadth": 0.5,
        "cb_vol": 0.0,
        "cb_rotation": 0.0,
    }

    def __init__(self, ohlcv: pd.DataFrame, engine, top_n: int = 5):
        """
        Args:
            ohlcv: 含 code, trade_date, close, amount, turnover 的日频数据
            engine: SQLAlchemy 数据库引擎
            top_n: 按成交额选取的热点板块数 (默认 5)
        """
        self.top_n = top_n
        self._cache: dict[pd.Timestamp, dict[str, float]] = {}
        self._ret_pivot: pd.DataFrame | None = None  # date × board 收益
        self._stock_boards: dict[str, list[str]] = {}  # stock → board codes
        self._hot_boards_cache: dict[pd.Timestamp, set[str]] = {}  # date → hot board codes
        self._precompute(ohlcv, engine)

    def get_features(self, date) -> dict[str, float]:
        """返回指定日期的 8 维概念板块特征。缓存未命中回退到最近的前一日。"""
        if isinstance(date, str):
            date = pd.Timestamp(date)
        date_key = pd.Timestamp(date.date())
        features = self._cache.get(date_key)
        if features is not None:
            return features
        # 回退
        prev_dates = sorted(d for d in self._cache.keys() if d < date_key)
        if prev_dates:
            return self._cache[prev_dates[-1]]
        logger.warning(f"概念板块特征未命中 {date.date()}")
        return dict(self._defaults)

    def get_stock_features(self, date, stock_codes: list[str]) -> pd.DataFrame:
        """返回指定日期每只股票的板块特征（可直接作为 ML 特征列）。

        Returns:
            DataFrame with columns: code, cb_s_mom5, cb_s_mom20, cb_s_hot, cb_s_nboards
        """
        import numpy as np
        import pandas as pd

        if isinstance(date, str):
            date = pd.Timestamp(date)
        dt_key = pd.Timestamp(date.date())

        hot_boards = self._hot_boards_cache.get(dt_key, set())
        ret_pivot = self._ret_pivot
        mapping = self._stock_boards

        rows = []
        for code in stock_codes:
            boards = mapping.get(code, [])
            n_boards = len(boards)

            # 板块 5日/20日动量
            mom5_vals = []
            mom20_vals = []
            in_hot = 0

            if ret_pivot is not None and dt_key in ret_pivot.index:
                pos = ret_pivot.index.get_loc(dt_key)
                for b in boards:
                    if b not in ret_pivot.columns:
                        continue
                    # 5日累计收益
                    if pos >= 4:
                        w5 = ret_pivot.iloc[max(0, pos-4):pos+1][b]
                        mom5_vals.append((1 + w5).prod() - 1)
                    # 20日累计收益
                    if pos >= 19:
                        w20 = ret_pivot.iloc[max(0, pos-19):pos+1][b]
                        mom20_vals.append((1 + w20).prod() - 1)
                    if b in hot_boards:
                        in_hot = 1

            cb_s_mom5 = float(np.mean(mom5_vals)) if mom5_vals else 0.0
            cb_s_mom20 = float(np.mean(mom20_vals)) if mom20_vals else 0.0
            cb_s_hot = in_hot
            cb_s_nboards = n_boards

            rows.append({
                "code": code,
                "cb_s_mom5": cb_s_mom5 if not np.isnan(cb_s_mom5) else 0.0,
                "cb_s_mom20": cb_s_mom20 if not np.isnan(cb_s_mom20) else 0.0,
                "cb_s_hot": cb_s_hot,
                "cb_s_nboards": cb_s_nboards,
            })

        return pd.DataFrame(rows)

    # ── 内部实现 ──────────────────────────────────────────

    def _precompute(self, ohlcv: pd.DataFrame, engine):
        """预计算所有日期的概念板块特征。"""

        # 1. 加载板块-成分股映射
        with engine.connect() as conn:
            mapping = pd.read_sql(
                "SELECT board_code, stock_code FROM concept_stock", conn,
            )
            boards = pd.read_sql("SELECT code, name FROM concept_board", conn)

        if mapping.empty:
            logger.warning("concept_stock 为空，跳过概念板块特征")
            return

        # 只保留在 ohlcv 中出现过的股票
        ohlcv_codes = set(ohlcv["code"].unique())
        mapping = mapping[mapping["stock_code"].isin(ohlcv_codes)]
        if mapping.empty:
            logger.warning("概念板块成分股均不在 OHLCV 中")
            return

        # 2. 计算每只股票每日收益率
        ohlcv = ohlcv.copy()
        ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
        ohlcv = ohlcv.sort_values(["code", "trade_date"])
        ohlcv["ret"] = ohlcv.groupby("code")["close"].pct_change()

        ret_df = ohlcv[["code", "trade_date", "ret", "amount"]].dropna(subset=["ret"])

        # 3. 板块等权日收益
        merged = ret_df.merge(
            mapping.rename(columns={"board_code": "board", "stock_code": "code"}),
            on="code", how="inner",
        )
        if merged.empty:
            logger.warning("板块-收益映射后为空")
            return

        board_daily = merged.groupby(["board", "trade_date"]).agg(
            board_ret=("ret", "mean"),
            board_amount=("amount", "sum"),
        ).reset_index()

        # 4. 转换为宽表 (date × board)
        ret_pivot = board_daily.pivot_table(
            index="trade_date", columns="board", values="board_ret", aggfunc="last",
        ).sort_index().fillna(0)

        amount_pivot = board_daily.pivot_table(
            index="trade_date", columns="board", values="board_amount", aggfunc="sum",
        ).sort_index().fillna(0)

        # 存下来供 get_stock_features 使用
        self._ret_pivot = ret_pivot
        self._stock_boards = mapping.groupby("stock_code")["board"].apply(list).to_dict()

        all_dates = sorted(ret_pivot.index)
        if len(all_dates) < 10:
            logger.warning("概念板块日期不足 10 天")
            return

        # 5. 滚动计算每日特征
        for i, dt in enumerate(all_dates):
            if i < 5:
                continue  # 需要至少 5 天历史

            # 当日成交额 Top-N 板块
            today_amount = amount_pivot.loc[dt]
            top_boards = today_amount.nlargest(self.top_n).index.tolist()
            self._hot_boards_cache[pd.Timestamp(dt.date())] = set(top_boards)

            # 板块 5 日收益 (截止当日，含当日)
            window_5 = ret_pivot.iloc[max(0, i - 4):i + 1]
            window_20 = ret_pivot.iloc[max(0, i - 19):i + 1]

            # Hot1-Hot5: Top-N 板块的 5 日累计收益
            hot_5d = {}
            for rank, board in enumerate(top_boards[:self.top_n]):
                if board in window_5.columns:
                    cum_ret = (1 + window_5[board]).prod() - 1
                    hot_5d[f"cb_hot{rank + 1}_5d"] = float(cum_ret) if not np.isnan(cum_ret) else 0.0
                else:
                    hot_5d[f"cb_hot{rank + 1}_5d"] = 0.0

            # Breadth: 全板块 5 日收益 > 0 的占比
            board_5d_rets = (1 + window_5).prod() - 1
            cb_breadth = float((board_5d_rets > 0).mean())

            # Vol: 全板块 20 日收益波动率均值
            board_20d_vols = window_20.std()
            cb_vol = float(board_20d_vols.mean()) if len(board_20d_vols) > 0 else 0.0

            # Rotation: Top-N 板块 5 日收益的横截面标准差
            top_5d_rets = board_5d_rets[top_boards].dropna()
            cb_rotation = float(top_5d_rets.std()) if len(top_5d_rets) > 1 else 0.0

            self._cache[pd.Timestamp(dt.date())] = {
                **hot_5d,
                "cb_breadth": cb_breadth if not np.isnan(cb_breadth) else 0.5,
                "cb_vol": cb_vol if not np.isnan(cb_vol) else 0.0,
                "cb_rotation": cb_rotation if not np.isnan(cb_rotation) else 0.0,
            }

        logger.info(f"概念板块特征预计算完成: {len(self._cache)} 日期, {len(ret_pivot.columns)} 板块")
