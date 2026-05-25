"""
自定义股票池管理。
池文件保存在 ~/.quant_stock_pools/，每个 .py 文件需定义 filter_stocks 函数。
"""

import os
import sys

import pandas as pd
from loguru import logger

POOL_DIR = os.path.expanduser("~/.quant_stock_pools")

if POOL_DIR not in sys.path:
    sys.path.insert(0, POOL_DIR)

DEFAULT_TEMPLATE = '''"""
自定义股票池筛选规则。
函数签名不可更改，修改函数体后保存即可在下一次回测中生效。
"""

import pandas as pd


def filter_stocks(
    basic: pd.DataFrame,
    extra: pd.DataFrame,
    shareholder: pd.DataFrame,
) -> list[str]:
    """
    股票筛选函数。

    参数
    ----
    basic : 股票基本信息
        columns: code, name, industry, market, list_date, is_st
    extra : 估值指标（最近一个交易日）
        columns: code, trade_date, market_cap(亿), float_market_cap(亿), pe, pb, total_share, float_share
    shareholder : 股东户数（最近一期报告）
        columns: code, end_date, shareholder_count, avg_holding_value(万元), avg_holding_amount, total_market_cap

    返回
    ----
    list[str] : 筛选后的股票代码列表
    """
    # ---------- 在此编写筛选逻辑 ----------

    # 1. 排除 ST / *ST
    codes = basic[~basic["is_st"]]["code"].tolist()

    # 2. 示例：只保留流通市值 10~200 亿的小盘股（如已同步 extra 数据，取消下面注释）
    # if not extra.empty:
    #     small_cap = extra[
    #         (extra["float_market_cap"] >= 10) & (extra["float_market_cap"] <= 200)
    #     ]["code"]
    #     codes = [c for c in codes if c in small_cap.values]

    # 3. 示例：只保留股东户数 > 20000 的「散户票」
    # if not shareholder.empty:
    #     retail = shareholder[shareholder["shareholder_count"] > 20000]["code"]
    #     codes = [c for c in codes if c in retail.values]

    return codes
'''


# ---------- 池管理 ----------

def list_pools() -> list[str]:
    """列出所有自定义股票池（不含 .py 后缀）。"""
    if not os.path.isdir(POOL_DIR):
        return []
    return sorted(
        f[:-3] for f in os.listdir(POOL_DIR)
        if f.endswith(".py") and not f.startswith("_")
    )


def load_pool_code(name: str) -> str:
    """读取池文件的源代码。"""
    path = os.path.join(POOL_DIR, f"{name}.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"池文件不存在: {path}")
    with open(path) as f:
        return f.read()


def save_pool(name: str, code: str) -> str:
    """保存池文件。返回文件路径。"""
    os.makedirs(POOL_DIR, exist_ok=True)
    safe = name.strip().replace(" ", "_").replace("/", "_")
    path = os.path.join(POOL_DIR, f"{safe}.py")
    with open(path, "w") as f:
        f.write(code)
    return path


def delete_pool(name: str) -> None:
    """删除一个池文件。"""
    path = os.path.join(POOL_DIR, f"{name}.py")
    if os.path.isfile(path):
        os.remove(path)


# ---------- 编译 & 执行 ----------

def compile_pool(code: str) -> tuple[bool, str]:
    """检查代码语法并验证 filter_stocks 函数是否存在。"""
    try:
        compile(code, "<stock_pool>", "exec")
    except SyntaxError as e:
        return False, f"语法错误: {e}"

    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return False, f"执行错误: {e}"

    if "filter_stocks" not in ns:
        return False, "未找到 filter_stocks 函数"
    if not callable(ns["filter_stocks"]):
        return False, "filter_stocks 必须是函数"

    return True, "OK"


def execute_pool(
    code: str,
    basic: pd.DataFrame,
    extra: pd.DataFrame = None,
    shareholder: pd.DataFrame = None,
) -> list[str]:
    """编译并执行池代码，返回筛选后的股票代码列表。"""
    ns: dict = {}
    exec(code, ns)
    fn = ns["filter_stocks"]
    return fn(
        basic=basic,
        extra=extra if extra is not None else pd.DataFrame(),
        shareholder=shareholder if shareholder is not None else pd.DataFrame(),
    )


def load_and_execute_pool(
    name: str,
    basic: pd.DataFrame,
    extra: pd.DataFrame = None,
    shareholder: pd.DataFrame = None,
) -> list[str]:
    """加载已保存的池文件并执行筛选。"""
    code = load_pool_code(name)
    return execute_pool(code, basic, extra, shareholder)


# ---------- 数据组装（供池筛选使用）----------

def get_pool_data(engine, extra_date: str | None = None):
    """从数据库加载股票池筛选所需的数据。

    返回 (basic, extra, shareholder) 三个 DataFrame。
    extra 取最近交易日数据，shareholder 取最近报告期数据。
    """
    basic = pd.read_sql("SELECT * FROM stock_basic", engine)
    extra = pd.DataFrame()
    shareholder = pd.DataFrame()

    try:
        if extra_date:
            extra = pd.read_sql(
                f"SELECT * FROM stock_daily_extra WHERE trade_date = '{extra_date}'", engine
            )
        else:
            latest = pd.read_sql(
                "SELECT trade_date FROM stock_daily_extra ORDER BY trade_date DESC LIMIT 1", engine
            )
            if not latest.empty:
                extra = pd.read_sql(
                    f"SELECT * FROM stock_daily_extra WHERE trade_date = '{latest.iloc[0, 0]}'", engine
                )
    except Exception:
        pass

    try:
        # 每只股票取最新一期报告数据（DISTINCT ON 按 code 去重）
        shareholder = pd.read_sql(
            "SELECT DISTINCT ON (code) * FROM stock_shareholder "
            "ORDER BY code, end_date DESC",
            engine,
        )
    except Exception:
        pass

    return basic, extra, shareholder
