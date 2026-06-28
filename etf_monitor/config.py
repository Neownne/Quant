"""ETF 监测配置：监控清单 + 信号阈值。

ETFS 从数据库动态加载（至少 60 天数据的 ETF），
不再硬编码。如需手动限制，设 ETFS_OVERRIDE。
"""
# 设为 None 则自动发现；设为 dict 则手动指定
ETFS_OVERRIDE: dict | None = None

# 最低数据天数（总历史少于 60 天的不监控）
MIN_DAYS = 60
# 最大监控数量（按近20日均成交额降序取前N只）
MAX_ETFS = 50

# 信号分级阈值
SIGNAL_HIGH = 70   # ≥70% 高确信
SIGNAL_MID = 50    # 50-70% 中等

# 因子权重
VOL_WT = 0.50
DIR_WT = 0.20
SHARE_WT = 0.30


def load_etfs(engine) -> dict[str, dict]:
    """从数据库动态加载 ETF 监控清单。"""
    if ETFS_OVERRIDE is not None:
        return ETFS_OVERRIDE

    import pandas as pd
    limit_clause = f"LIMIT {MAX_ETFS}" if MAX_ETFS else ""
    df = pd.read_sql(f"""
        SELECT code, name, market FROM (
            SELECT d.code, b.name, b.market,
                   AVG(d.amount) FILTER (WHERE d.trade_date >= (SELECT MAX(trade_date) FROM etf_daily) - 20) as avg_amount
            FROM etf_daily d
            JOIN etf_basic b ON d.code = b.code
            GROUP BY d.code, b.name, b.market
            HAVING COUNT(*) >= {MIN_DAYS}
        ) t
        WHERE avg_amount > 0
        ORDER BY avg_amount DESC
        {limit_clause}
    """, engine)

    etfs = {}
    for _, row in df.iterrows():
        code = str(row["code"])
        name = str(row["name"]) if row["name"] else code
        market = str(row["market"]) if row["market"] else "SH"
        # 推断跟踪指数
        if "300" in name: idx_name = "沪深300"
        elif "500" in name: idx_name = "中证500"
        elif "创业" in name: idx_name = "创业板"
        elif "科创" in name: idx_name = "科创50"
        elif "1000" in name: idx_name = "中证1000"
        elif "上证50" in name or "50" in name: idx_name = "上证50"
        else: idx_name = "综合"
        etfs[code] = {"name": name, "idx_name": idx_name, "market": market}

    return etfs
