"""板块分类映射配置。

支持两种板块分类方式：
- broad：按市场分层（科创/北证/红利/主板大盘/主板小盘）
- sw1：按申万一级行业（19类）

板块优先级（broad模式）：科创 > 北证 > 红利 > 主板大盘 > 主板小盘
"""
from __future__ import annotations

# 板块标签
SECTOR_LABELS = ("科创", "北证", "红利", "主板大盘", "主板小盘")

# 分类方式常量
BROAD_CLASSIFICATION = "broad"
SW1_CLASSIFICATION = "sw1"


def _code_prefix(code: str) -> str:
    """提取股票代码的数字部分前3位（兼容有/无后缀格式）。"""
    # code 格式如 "688001.SH" 或 "688001"
    return code.split(".")[0][:3]


def classify_stock(
    code: str,
    csi300_members: set[str],
    dividend_stocks: set[str],
) -> str:
    """将单只股票归类到5大板块之一。

    优先级：科创 > 北证 > 红利 > 主板大盘 > 主板小盘

    参数
    ----
    code : 股票代码（如 "688001.SH"）
    csi300_members : 沪深300成分股代码集合
    dividend_stocks : 高股息率股票代码集合（全市场前20%）

    返回
    ----
    板块标签："科创" | "北证" | "红利" | "主板大盘" | "主板小盘"
    """
    prefix = _code_prefix(code)

    # 优先级1：科创板（688开头）
    if prefix == "688":
        return "科创"

    # 优先级2：北交所（8或4开头，但已在688之后）
    if prefix[0] in ("8", "4"):
        return "北证"

    # 优先级3：红利股
    if code in dividend_stocks:
        return "红利"

    # 优先级4：主板大盘（沪深300成分）
    if code in csi300_members:
        return "主板大盘"

    # 优先级5：其余
    return "主板小盘"


def build_sector_map(
    codes: list[str],
    csi300_members: set[str],
    dividend_stocks: set[str],
) -> dict[str, str]:
    """批量构建股票代码 → 板块标签的映射。

    参数
    ----
    codes : 股票代码列表
    csi300_members : 沪深300成分股代码集合
    dividend_stocks : 高股息率股票代码集合

    返回
    ----
    {code: sector_label} 字典
    """
    return {code: classify_stock(code, csi300_members, dividend_stocks) for code in codes}
