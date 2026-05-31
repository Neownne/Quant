"""ETF 监测配置：监控清单 + 信号阈值。"""

# 7 只国家队增持宽基 ETF
ETFS: dict[str, dict] = {
    "510300": {"name": "沪深300ETF(华泰柏瑞)", "idx_name": "沪深300", "market": "SH"},
    "510310": {"name": "沪深300ETF(易方达)",   "idx_name": "沪深300", "market": "SH"},
    "510330": {"name": "沪深300ETF(华夏)",     "idx_name": "沪深300", "market": "SH"},
    "159919": {"name": "沪深300ETF(嘉实)",     "idx_name": "沪深300", "market": "SZ"},
    "510050": {"name": "上证50ETF",            "idx_name": "上证50",  "market": "SH"},
    "510500": {"name": "中证500ETF",           "idx_name": "中证500", "market": "SH"},
    "512100": {"name": "中证1000ETF",          "idx_name": "中证1000","market": "SH"},
}

# 信号分级阈值
SIGNAL_HIGH = 70   # ≥70% 高确信
SIGNAL_MID = 50    # 50-70% 中等

# 因子权重
VOL_WT = 0.50
DIR_WT = 0.20
SHARE_WT = 0.30
