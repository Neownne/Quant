"""自定义股票池编辑器 —— 用 Python 代码定义选股规则。"""

import os

import pandas as pd
import streamlit as st

from app.utils.stock_pools import (
    POOL_DIR,
    DEFAULT_TEMPLATE,
    list_pools,
    load_pool_code,
    save_pool,
    delete_pool,
    compile_pool,
    execute_pool,
    get_pool_data,
)
from data.db import get_engine, init_db

st.set_page_config(page_title="股票池编辑器", page_icon="📦", layout="wide")
st.title("📦 自定义股票池")

os.makedirs(POOL_DIR, exist_ok=True)
init_db()

# ---- 侧边栏：池列表 ----
with st.sidebar:
    st.header("股票池")

    pools = list_pools()
    if not pools:
        st.info("暂无自定义股票池，点击「新建」")

    selected_pool = st.selectbox("选择编辑", [""] + pools, key="select_pool")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ 新建", use_container_width=True):
            st.session_state.pool_name = ""
            st.session_state.pool_code = DEFAULT_TEMPLATE
            st.rerun()
    with col2:
        if selected_pool:
            if st.button("🗑 删除", use_container_width=True):
                delete_pool(selected_pool)
                st.cache_data.clear()
                st.session_state.pop("pool_name", None)
                st.session_state.pop("pool_code", None)
                st.rerun()

    st.divider()
    st.caption("池文件保存在 ~/.quant_stock_pools/")

# ---- 主区域：编辑器 ----
if selected_pool and "pool_code" not in st.session_state:
    try:
        st.session_state.pool_code = load_pool_code(selected_pool)
        st.session_state.pool_name = selected_pool
    except FileNotFoundError:
        st.error(f"池 {selected_pool} 已不存在")
        st.stop()

if "pool_code" not in st.session_state:
    st.session_state.pool_code = DEFAULT_TEMPLATE
if "pool_name" not in st.session_state:
    st.session_state.pool_name = ""

pool_name = st.text_input(
    "池名称（英文，用于保存文件名）",
    value=st.session_state.pool_name,
    placeholder="small_cap_retail",
    key="name_input",
)

pool_code = st.text_area(
    "筛选代码",
    value=st.session_state.pool_code,
    height=500,
    key="code_input",
)

# 同步 session state
if pool_code != st.session_state.get("pool_code"):
    st.session_state.pool_code = pool_code
if pool_name != st.session_state.get("pool_name"):
    st.session_state.pool_name = pool_name

col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    if st.button("🧪 编译测试", use_container_width=True):
        ok, msg = compile_pool(pool_code)
        if ok:
            st.success("编译通过，filter_stocks 函数已就绪")
        else:
            st.error(msg)

with col2:
    if st.button("💾 保存", use_container_width=True, type="primary"):
        if not pool_name.strip():
            st.error("请输入池名称")
        else:
            ok, msg = compile_pool(pool_code)
            if not ok:
                st.error(f"编译失败，无法保存: {msg}")
            else:
                path = save_pool(pool_name, pool_code)
                st.cache_data.clear()
                st.success(f"已保存到 {path}")

# ---- 数据预览 & 实时筛选 ----
st.divider()
st.subheader("筛选预览")

if st.button("🔍 运行筛选（预览前 20 只）", use_container_width=True):
    ok, msg = compile_pool(pool_code)
    if not ok:
        st.error(msg)
    else:
        engine = get_engine()
        try:
            with st.spinner("加载数据并筛选 ..."):
                basic, extra, shareholder = get_pool_data(engine)
                st.caption(
                    f"已加载 {len(basic)} 只股票"
                    + (f"，{len(extra)} 条估值数据" if not extra.empty else "")
                    + (f"，{len(shareholder)} 条股东数据" if not shareholder.empty else "")
                )
                codes = execute_pool(pool_code, basic, extra, shareholder)
                st.success(f"筛选结果: {len(codes)} 只股票")

                if codes:
                    result_df = basic[basic["code"].isin(codes)][
                        ["code", "name", "industry", "market", "list_date"]
                    ].head(20)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    if len(codes) > 20:
                        st.caption(f"…… 共 {len(codes)} 只，仅展示前 20")
                else:
                    st.warning("筛选结果为空，请检查条件是否过于严格")
        finally:
            engine.dispose()

# ---- 编写指南 ----
with st.expander("📖 编写指南 & 数据字典"):
    st.markdown("""
### 函数签名（不可更改）

```python
def filter_stocks(basic, extra, shareholder) -> list[str]:
```

### 数据字典

**basic** — 股票基本信息 (`stock_basic` 表)

| 列 | 类型 | 说明 |
|---|---|---|
| code | str | 股票代码 |
| name | str | 名称 |
| industry | str | 行业 |
| market | str | SZ / SH / BJ |
| list_date | date | 上市日期 |
| is_st | bool | 是否 ST |

**extra** — 估值指标 (`stock_daily_extra` 表，最近交易日)

| 列 | 类型 | 单位 | 说明 |
|---|---|---|---|
| code | str | | |
| market_cap | float | 亿元 | 总市值 |
| float_market_cap | float | 亿元 | 流通市值 |
| pe | float | | 市盈率 (TTM) |
| pb | float | | 市净率 |
| total_share | float | 亿股 | 总股本 |
| float_share | float | 亿股 | 流通股本 |

**shareholder** — 股东户数 (`stock_shareholder` 表，最近报告期)

| 列 | 类型 | 单位 | 说明 |
|---|---|---|---|
| code | str | | |
| shareholder_count | int | 户 | 股东总户数 |
| avg_holding_value | float | 万元 | 户均持股市值 |
| avg_holding_amount | float | 股 | 户均持股数量 |

### 常用筛选范式

```python
# 小盘股（流通市值 10~200 亿）
small = extra[(extra['float_market_cap'] >= 10) & (extra['float_market_cap'] <= 200)]
codes = [c for c in codes if c in small['code'].values]

# 高散户参与度（股东户数 > 20000）
retail = shareholder[shareholder['shareholder_count'] > 20000]
codes = [c for c in codes if c in retail['code'].values]

# 排除特定行业
exclude = basic[basic['industry'].str.contains('银行|保险|房地产', na=False)]
codes = [c for c in codes if c not in exclude['code'].values]

# 排除上市不满 1 年的新股
from datetime import date
one_year_ago = date.today().replace(year=date.today().year - 1)
mature = basic[basic['list_date'] <= one_year_ago]
codes = [c for c in codes if c in mature['code'].values]

# 低 PE（PE > 0 排除亏损，PE < 50 不过贵）
value = extra[(extra['pe'] > 0) & (extra['pe'] < 50)]
codes = [c for c in codes if c in value['code'].values]
```
""")

with st.expander("📊 当前数据库概览"):
    if st.button("刷新概览"):
        st.cache_data.clear()
    engine = get_engine()
    try:
        basic, extra, shareholder = get_pool_data(engine)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("股票总数", len(basic))
        with col2:
            st.metric("有估值数据", len(extra))
        with col3:
            st.metric("有股东数据", len(shareholder))
        if not extra.empty:
            st.caption(
                f"市值范围: {extra['float_market_cap'].min():.1f} ~ {extra['float_market_cap'].max():.1f} 亿"
            )
    finally:
        engine.dispose()
