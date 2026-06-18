"""涨停板专用因子 — 连板模式识别 · 封板质量 · 板块共振 · 连板阶段。

理论来源:
  - 涨停板11因子 (中国经济出版社, 2025)
  - BigQuant AI+涨停板特征提取 (2024)
  - 天风证券 连板晋级率情绪指标 (2025.09)
  - king-pin 封板王 多因子评分引擎 (GitHub, 2025)
  - Stock Price Limit and Its Predictability (J. Forecasting, 2025)

因子签名统一: (df: pd.DataFrame) -> pd.Series
df 须包含: open, high, low, close, volume
可选列:   turnover, amount, market_cap, industry, ret (预计算收益)

板块/行业因子需通过 compute_sector_factors() 预聚合后作为 extra_data 注入。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from factors._scaling import w

# ── 涨跌停阈值（板别感知，四舍五入）──
_LIMIT_MULT = {
    "688": 1.20,  # 科创板 ±20%
    "8": 1.30,    # 北交所 ±30%
    "4": 1.30,    # 北交所 ±30%
    "300": 1.20,  # 创业板 ±20%
    "301": 1.20,  # 创业板 ±20%
}
_DEFAULT_MULT = 1.10  # 主板 ±10%


def _get_multiplier(code: str) -> float:
    """根据股票代码前缀返回涨停乘数。"""
    for prefix, mult in _LIMIT_MULT.items():
        if str(code).startswith(prefix):
            return mult
    return _DEFAULT_MULT


def is_at_limit_up(close: float, prev_close: float, code: str) -> bool:
    """A股涨停价四舍五入判断。与 TradingConfig 保持一致。"""
    import math
    if pd.isna(close) or pd.isna(prev_close) or prev_close <= 0:
        return False
    mult = _get_multiplier(str(code))
    limit_price = round(prev_close * mult, 2)
    return close >= limit_price


def is_at_limit_down(close: float, prev_close: float, code: str) -> bool:
    """A股跌停价四舍五入判断。"""
    if pd.isna(close) or pd.isna(prev_close) or prev_close <= 0:
        return False
    mult = _get_multiplier(str(code))
    limit_price = round(prev_close * (2 - mult), 2)
    return close <= limit_price


def _is_limit_up(df: pd.DataFrame) -> pd.Series:
    """检测每日是否涨停（板别感知，四舍五入）。

    前提: df 需有 'close', 'prev_close', 'code' 列。
    返回: bool Series，涨停为 True。
    """
    if "prev_close" not in df.columns:
        df = df.copy()
        df["prev_close"] = df.groupby("code")["close"].shift(1)

    code = df["code"].iloc[0] if "code" in df.columns else ""
    mult = _get_multiplier(str(code))
    return df["close"] >= round(df["prev_close"] * mult, 2)


# ══════════════════════════════════════════════════════════════════════
# Category 1: 涨停模式识别 (Limit-Up Pattern Recognition)
# ══════════════════════════════════════════════════════════════════════

def lu_streak(df: pd.DataFrame) -> pd.Series:
    """当前连板数: 截至当日已连续涨停的天数（含当日）。

    实证: 连板高度与次日溢价非线性相关，2-4板最强。
    来源: 天风证券连板晋级率研究 (2025)
    """
    is_lu = _is_limit_up(df).astype(int)
    streak = pd.Series(0, index=df.index, dtype=int)
    cnt = 0
    for i in range(len(is_lu)):
        if is_lu.iloc[i]:
            cnt += 1
        else:
            cnt = 0
        streak.iloc[i] = cnt
    return streak.astype(float)


def lu_max_streak_20d(df: pd.DataFrame) -> pd.Series:
    """20日内最大连板数。

    反映近期股性活跃度。值越大说明该股近期多次连板，投机资金关注度高。
    """
    is_lu = _is_limit_up(df).astype(int)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for i in range(len(df)):
        start = max(0, i - w(20) + 1)
        window = is_lu.iloc[start:i+1]
        max_streak = 0
        cur = 0
        for v in window:
            if v:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        result.iloc[i] = float(max_streak)
    return result


def lu_max_streak_60d(df: pd.DataFrame) -> pd.Series:
    """60日内最大连板数。更长周期看股性。"""
    is_lu = _is_limit_up(df).astype(int)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for i in range(len(df)):
        start = max(0, i - w(60) + 1)
        window = is_lu.iloc[start:i+1]
        max_streak = 0
        cur = 0
        for v in window:
            if v:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        result.iloc[i] = float(max_streak)
    return result


def lu_count_5d(df: pd.DataFrame) -> pd.Series:
    """5日内涨停次数。短期爆发力。"""
    return _is_limit_up(df).rolling(w(5), min_periods=1).sum()


def lu_count_20d(df: pd.DataFrame) -> pd.Series:
    """20日内涨停次数。中期活跃度。

    来源: BigQuant price_limit_status 因子，过去10-20日涨停次数是StockRanker核心特征。
    """
    return _is_limit_up(df).rolling(w(20), min_periods=1).sum()


def lu_count_60d(df: pd.DataFrame) -> pd.Series:
    """60日内涨停次数。长期股性。"""
    return _is_limit_up(df).rolling(w(60), min_periods=1).sum()


def lu_days_since_last(df: pd.DataFrame) -> pd.Series:
    """距上次涨停的交易日数。数值越大，冷却越充分。

    首板（大值）vs 连板途中（小值）的信号质量差异显著。
    """
    is_lu = _is_limit_up(df).astype(int)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    last_lu_idx = -999
    for i in range(len(df)):
        if is_lu.iloc[i]:
            last_lu_idx = i
        result.iloc[i] = float(i - last_lu_idx) if last_lu_idx >= 0 else np.nan
    return result


def lu_first_board(df: pd.DataFrame) -> pd.Series:
    """是否首板: 当日涨停且前5日无涨停。首板溢价潜力最大。

    实证: 首板次日溢价概率>60%，三板以上分歧加大。
    来源: king-pin 封板王 (GitHub, 2025)
    """
    is_lu = _is_limit_up(df).astype(int)
    prev_5 = is_lu.shift(1).rolling(w(5), min_periods=1).sum().fillna(0)
    return ((is_lu == 1) & (prev_5 == 0)).astype(float)


def lu_is_second_board(df: pd.DataFrame) -> pd.Series:
    """是否二板: 昨日涨停且今日涨停（一进二）。

    实证: 一进二是连板策略最关键的择时点。二板质量评分体系来自cjm99.com。
    来源: 二板质量评价体系 (7因子加权)
    """
    is_lu = _is_limit_up(df).astype(int)
    return ((is_lu == 1) & (is_lu.shift(1) == 1) &
            ((is_lu.shift(2).fillna(0)) == 0)).astype(float)


def lu_freq_accel(df: pd.DataFrame) -> pd.Series:
    """涨停频率加速度: 近10日频率 / 近30日频率 - 1。

    正值=加速（连板启动），负值=减速（热度退潮）。
    """
    is_lu = _is_limit_up(df)
    recent = is_lu.rolling(w(10), min_periods=1).sum()
    base = is_lu.rolling(w(30), min_periods=1).sum()
    return (recent / base.replace(0, np.nan)) - 1.0


def lu_seal_quality(df: pd.DataFrame) -> pd.Series:
    """封板质量: close / high。越接近1封板越强。

    一字板: close/high ≈ 1.0，最强封板。
    尾盘拉板: close/high < 0.95，次日大概率低开。

    来源: 和讯投顾 封板强度评估 (2025)
    """
    close_high = df["close"] / df["high"].replace(0, np.nan)
    is_lu = _is_limit_up(df)
    # 只在涨停日有意义，非涨停日返回 NaN
    return close_high.where(is_lu, np.nan)


def lu_vol_intensity(df: pd.DataFrame) -> pd.Series:
    """涨停日量能强度: volume / avg_vol_20。

    缩量板（<1.0）: 筹码锁定好，继续连板概率高。
    放量板（>2.0）: 分歧加大，可能是顶部信号。

    来源: 涨停板11因子 封单量与换手率因子
    """
    avg_vol = df["volume"].rolling(w(20), min_periods=5).mean()
    ratio = df["volume"] / avg_vol.replace(0, np.nan)
    is_lu = _is_limit_up(df)
    return ratio.where(is_lu, np.nan)


def lu_open_strength(df: pd.DataFrame) -> pd.Series:
    """涨停日开盘跳空强度: (open - prev_close) / prev_close。

    大幅高开（>5%）: 集合竞价抢筹，当日封板概率高。
    平开/低开涨停: 主力尾盘拉板，质量存疑。
    """
    prev_c = df["close"].shift(1)
    gap = (df["open"] - prev_c) / prev_c.replace(0, np.nan)
    is_lu = _is_limit_up(df)
    return gap.where(is_lu, np.nan)


def lu_body_ratio(df: pd.DataFrame) -> pd.Series:
    """涨停日实体占比: (close - open) / (high - low)。

    实体板（>0.8）: 多头强势，收盘价接近最高价。
    T字板（≈0.5）: 开盘即涨停但盘中曾打开。
    一字板（high=low 时退化为 NaN，配合 seal_quality 判断）。
    """
    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]) / high_low
    is_lu = _is_limit_up(df)
    return body.where(is_lu, np.nan)


def lu_upper_shadow_ratio(df: pd.DataFrame) -> pd.Series:
    """涨停日上影线占比: (high - max(open, close)) / (high - low)。

    上影线越长说明盘中涨停后被砸开（炸板），越小越好。
    0 表示收盘价=最高价（强封板）。
    """
    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    upper = (df["high"] - df[["open", "close"]].max(axis=1)) / high_low
    is_lu = _is_limit_up(df)
    return upper.where(is_lu, np.nan)


def lu_amplitude(df: pd.DataFrame) -> pd.Series:
    """涨停日振幅: (high - low) / prev_close。

    小振幅封板（<3%）: 筹码稳定，一致性高。
    大振幅封板（>8%）: 盘中分歧剧烈。
    """
    prev_c = df["close"].shift(1)
    amp = (df["high"] - df["low"]) / prev_c.replace(0, np.nan)
    is_lu = _is_limit_up(df)
    return amp.where(is_lu, np.nan)


def lu_volume_climax(df: pd.DataFrame) -> pd.Series:
    """涨停日是否量能极值: volume / (avg_vol_20 + 2*std_vol_20)。

    >1.0 表示量能异常放大，可能是主力出货信号。
    <0.5 表示缩量涨停，筹码锁定良好。
    """
    avg = df["volume"].rolling(w(20), min_periods=5).mean()
    std = df["volume"].rolling(w(20), min_periods=5).std()
    threshold = avg + 2 * std
    ratio = df["volume"] / threshold.replace(0, np.nan)
    is_lu = _is_limit_up(df)
    return ratio.where(is_lu, np.nan)


def lu_intraday_reversal(df: pd.DataFrame) -> pd.Series:
    """涨停日盘中反转: (close - low) / (high - low)。

    接近1 = 开盘低点后拉涨停，多头力量极强。
    接近0 = 冲高回落（虽然收涨停，但盘中曾大幅回落）。
    """
    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    rev = (df["close"] - df["low"]) / high_low
    is_lu = _is_limit_up(df)
    return rev.where(is_lu, np.nan)


# ══════════════════════════════════════════════════════════════════════
# Category 2: 首板前吸筹/蓄力 (Pre-Breakout Accumulation)
# ══════════════════════════════════════════════════════════════════════

def pre_lu_vol_trend(df: pd.DataFrame) -> pd.Series:
    """首板前5日量能趋势: 线性回归斜率 / 均值。

    正值表示量能在首板前温和放大（主力吸筹迹象），负值表示缩量。
    只在首板日有值（lu_first_board == 1），其余为 NaN。
    """
    vol = df["volume"].astype(float)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    is_lu = _is_limit_up(df)

    for i in range(w(5), len(df)):
        if is_lu.iloc[i]:
            # 检查是否是首板（前5日无涨停）
            if is_lu.iloc[i-w(5):i].sum() == 0:
                prev_vol = vol.iloc[i-w(5):i]
                if len(prev_vol) >= 3:
                    x = np.arange(len(prev_vol))
                    slope = np.polyfit(x, prev_vol.values, 1)[0]
                    result.iloc[i] = slope / prev_vol.mean() if prev_vol.mean() > 0 else 0
    return result


def pre_lu_ret_5d(df: pd.DataFrame) -> pd.Series:
    """首板前5日累计收益。负值=超跌反弹板，正值=趋势加速板。

    实证: 超跌首板（前5日跌>5%）的短期爆发力强于趋势首板。
    """
    prev_c5 = df["close"].shift(5)
    ret = (df["close"].shift(1) - prev_c5) / prev_c5.replace(0, np.nan)
    is_first = lu_first_board(df)
    return ret.where(is_first == 1, np.nan)


def pre_lu_turnover_cv(df: pd.DataFrame) -> pd.Series:
    """首板前10日换手率变异系数。高CV=筹码博弈激烈。

    低CV（<0.3）+ 首板缩量 = 主力控盘好，连板概率高。
    """
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    to = df["turnover"]
    cv = to.rolling(w(10)).std() / to.rolling(w(10)).mean().replace(0, np.nan)
    is_first = lu_first_board(df)
    return cv.where(is_first == 1, np.nan)


# ══════════════════════════════════════════════════════════════════════
# Category 3: 连板确认/分歧 (Continuation Confirmation)
# ══════════════════════════════════════════════════════════════════════

def lu_next_day_gap(df: pd.DataFrame) -> pd.Series:
    """涨停次日开盘缺口: (次日open - 当日close) / 当日close。

    高开（>2%）: 连板确认信号，市场继续看多。
    低开（<0%）: 分歧加大，连板可能终止。
    """
    next_open = df["open"].shift(-1)
    gap = (next_open - df["close"]) / df["close"].replace(0, np.nan)
    is_lu = _is_limit_up(df)
    return gap.where(is_lu, np.nan)


def lu_vol_contraction(df: pd.DataFrame) -> pd.Series:
    """连板途中量能收缩: 今日量 / 昨日量（仅连板日有效）。

    <1.0 = 缩量加速（健康连板），>1.5 = 放量分歧（需警惕）。
    """
    vol_ratio = df["volume"] / df["volume"].shift(1).replace(0, np.nan)
    is_lu = _is_limit_up(df)
    yesterday_lu = _is_limit_up(df).shift(1).fillna(0).astype(bool)
    # 只在连续涨停时有效
    valid = is_lu & yesterday_lu
    return vol_ratio.where(valid, np.nan)


def lu_streak_quality(df: pd.DataFrame) -> pd.Series:
    """连板质量综合评分: 连板数 × 封板质量 × 量能健康度。

    综合评估连板的质量而非单纯看连板数。
    """
    streak = lu_streak(df)
    seal = lu_seal_quality(df)
    vol_cont = lu_vol_contraction(df)
    # 量能收缩分: <1 加分，>1.5 减分
    vol_score = 1.0 / vol_cont.replace(0, np.nan).clip(0.5, 3.0)
    return streak * seal.fillna(0.9) * vol_score.fillna(1.0)


# ══════════════════════════════════════════════════════════════════════
# Category 4: 相对强度与超额收益
# ══════════════════════════════════════════════════════════════════════

def lu_excess_return_5d(df: pd.DataFrame) -> pd.Series:
    """近5日超额收益（相对20日均线位置变化）。

    用于判断涨停前的趋势强度。
    """
    ma20 = df["close"].rolling(w(20), min_periods=5).mean()
    pos_today = df["close"] / ma20.replace(0, np.nan) - 1
    pos_5d_ago = df["close"].shift(5) / ma20.shift(5).replace(0, np.nan) - 1
    is_lu = _is_limit_up(df)
    return (pos_today - pos_5d_ago).where(is_lu, np.nan)


def lu_relative_strength_20d(df: pd.DataFrame) -> pd.Series:
    """20日相对强度: 个股20日收益 vs 全市场20日收益。

    需要 extra_data 提供 mkt_ret_mean 列，否则退化为绝对动量。
    """
    ret_20 = df["close"].pct_change(w(20))
    is_lu = _is_limit_up(df)
    if "mkt_ret_mean" in df.columns:
        rs = ret_20 - df["mkt_ret_mean"]
    else:
        rs = ret_20
    return rs.where(is_lu, np.nan)


def lu_turnover_intensity(df: pd.DataFrame) -> pd.Series:
    """涨停日换手率强度: turnover / avg(turnover, 20)。

    换手率 5%-15% 最佳: 既有流动性又不至于出货。
    <2% 可能一字板买不到，>20% 可能主力出货。
    """
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    avg_to = df["turnover"].rolling(w(20), min_periods=5).mean()
    ratio = df["turnover"] / avg_to.replace(0, np.nan)
    is_lu = _is_limit_up(df)
    return ratio.where(is_lu, np.nan)


# ══════════════════════════════════════════════════════════════════════
# Category 5: 市场环境/情绪代理 (Market Regime Proxies)
# ══════════════════════════════════════════════════════════════════════

def mkt_lu_breadth_proxy(df: pd.DataFrame) -> pd.Series:
    """全市场涨停宽度代理: 该股所在日涨停的粗略估计。

    注意: 这是单只股票的视角，真正的全市场宽度需通过
    build_market_breadth_extra() 预计算后注入。
    此处通过该股的 lu_streak 状态做粗略估计。
    """
    is_lu = _is_limit_up(df).astype(int)
    # 用当日是否涨停来标记（实际使用时需替换为全市场数据）
    return is_lu.astype(float)


def lu_board_height_rank(df: pd.DataFrame) -> pd.Series:
    """该股的连板高度（用于后续在全市场股票间排名）。

    输出连板数，可在截面排名中识别"全市场最高连板"龙头。
    """
    return lu_streak(df)


# ══════════════════════════════════════════════════════════════════════
# 因子注册表
# ══════════════════════════════════════════════════════════════════════

LIMIT_UP_FACTORS: dict = {
    # 涨停模式识别 (15)
    "lu_streak": lu_streak,
    "lu_max_streak_20d": lu_max_streak_20d,
    "lu_max_streak_60d": lu_max_streak_60d,
    "lu_count_5d": lu_count_5d,
    "lu_count_20d": lu_count_20d,
    "lu_count_60d": lu_count_60d,
    "lu_days_since_last": lu_days_since_last,
    "lu_first_board": lu_first_board,
    "lu_is_second_board": lu_is_second_board,
    "lu_freq_accel": lu_freq_accel,
    "lu_seal_quality": lu_seal_quality,
    "lu_vol_intensity": lu_vol_intensity,
    "lu_open_strength": lu_open_strength,
    "lu_body_ratio": lu_body_ratio,
    "lu_upper_shadow_ratio": lu_upper_shadow_ratio,
    # 封板质量补充 (4)
    "lu_amplitude": lu_amplitude,
    "lu_volume_climax": lu_volume_climax,
    "lu_intraday_reversal": lu_intraday_reversal,
    "lu_vol_contraction": lu_vol_contraction,
    # 首板前蓄力 (3)
    "pre_lu_vol_trend": pre_lu_vol_trend,
    "pre_lu_ret_5d": pre_lu_ret_5d,
    "pre_lu_turnover_cv": pre_lu_turnover_cv,
    # 连板确认 (3)
    "lu_next_day_gap": lu_next_day_gap,
    "lu_streak_quality": lu_streak_quality,
    "lu_turnover_intensity": lu_turnover_intensity,
    # 相对强度 (2)
    "lu_excess_return_5d": lu_excess_return_5d,
    "lu_relative_strength_20d": lu_relative_strength_20d,
    # 市场/排名 (2)
    "mkt_lu_breadth_proxy": mkt_lu_breadth_proxy,
    "lu_board_height_rank": lu_board_height_rank,
}


# ── 辅助函数：从全市场数据计算行业级涨停因子 ──

def compute_sector_lu_factors(
    daily: pd.DataFrame,
    industry_map: dict[str, str],
) -> pd.DataFrame:
    """从全市场日线数据预计算行业级别涨停因子。

    返回 DataFrame: [trade_date, industry, sector_lu_n, sector_lu_ratio,
                     sector_ret_mean, sector_leader_code, ...]

    参数
    ----
    daily : 全市场日线 DataFrame，需含 code, trade_date, close, ret (预计算)
    industry_map : {code: industry} 映射
    """
    df = daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["industry"] = df["code"].map(industry_map)
    df = df.dropna(subset=["industry"])

    # 计算每只股票每日是否涨停
    if "ret" not in df.columns:
        prev = df[["code", "trade_date", "close"]].copy()
        prev["trade_date"] = prev["trade_date"] + pd.Timedelta(days=1)
        prev = prev.rename(columns={"close": "prev_close"})
        df = df.merge(prev, on=["code", "trade_date"], how="left")
        df["ret"] = (df["close"] - df["prev_close"]) / df["prev_close"]

    df["is_lu"] = df.apply(
        lambda r: r["ret"] >= _get_limit(str(r["code"])) * 0.98
        if pd.notna(r["ret"]) else False, axis=1
    )

    # 按行业×日期聚合
    sector_stats = df.groupby(["trade_date", "industry"]).agg(
        sector_lu_n=("is_lu", "sum"),
        sector_n=("code", "count"),
        sector_ret_mean=("ret", "mean"),
        sector_ret_std=("ret", "std"),
        sector_turnover_mean=("turnover", "mean") if "turnover" in df.columns else ("ret", lambda x: np.nan),
    ).reset_index()

    sector_stats["sector_lu_ratio"] = (
        sector_stats["sector_lu_n"] / sector_stats["sector_n"].replace(0, np.nan)
    )

    # 行业龙头: 行业内最早涨停(用收益最高作代理)
    # dropna 避免 skipna=True 遇到全 NA 组报错
    df_ret_valid = df.dropna(subset=["ret"])
    idx = df_ret_valid.groupby(["trade_date", "industry"])["ret"].idxmax()
    leaders = df.loc[idx.dropna(), ["trade_date", "industry", "code"]]
    leaders = leaders.rename(columns={"code": "sector_leader_code"})
    sector_stats = sector_stats.merge(leaders, on=["trade_date", "industry"], how="left")

    return sector_stats


def compute_market_lu_extra(daily: pd.DataFrame) -> pd.DataFrame:
    """从全市场日线计算每日市场级别涨停统计。

    返回 DataFrame: [trade_date, mkt_lu_total, mkt_lu_breadth, mkt_lu_streak_avg,
                     mkt_total_stocks, mkt_seal_avg, mkt_first_board_n]

    可直接作为 extra_data 注入 FactorEngine。
    """
    df = daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["code"] = df["code"].astype(str).str.zfill(6)

    if "ret" not in df.columns:
        prev = df[["code", "trade_date", "close"]].copy()
        prev["trade_date"] = prev["trade_date"] + pd.Timedelta(days=1)
        prev = prev.rename(columns={"close": "prev_close"})
        df = df.merge(prev, on=["code", "trade_date"], how="left")
        df["ret"] = (df["close"] - df["prev_close"]) / df["prev_close"]

    df["is_lu"] = df.apply(
        lambda r: r["ret"] >= _get_limit(str(r["code"])) * 0.98
        if pd.notna(r["ret"]) else False, axis=1
    )

    # 封板质量
    df["seal"] = df["close"] / df["high"].replace(0, np.nan)
    df["seal_lu"] = df["seal"].where(df["is_lu"], np.nan)

    mkt = df.groupby("trade_date").agg(
        mkt_lu_total=("is_lu", "sum"),
        mkt_total_stocks=("code", "count"),
        mkt_lu_streak_avg=("is_lu", lambda x: np.nan),  # 需要逐股算连板，此处占位
        mkt_seal_avg=("seal_lu", "mean"),
    ).reset_index()

    mkt["mkt_lu_breadth"] = mkt["mkt_lu_total"] / mkt["mkt_total_stocks"].replace(0, np.nan)

    return mkt
