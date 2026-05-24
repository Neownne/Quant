import importlib
import os
import sys
import traceback

import streamlit as st

_CUSTOM_DIR = os.path.expanduser("~/.quant_strategies")

if _CUSTOM_DIR not in sys.path:
    sys.path.insert(0, _CUSTOM_DIR)

TEMPLATE = '''import backtrader as bt


class MyStrategy(bt.Strategy):
    """
    自定义策略描述。
    """
    params = (
        ("fast", 10),
        ("slow", 30),
    )

    def __init__(self):
        self.fast_ma = bt.ind.SMA(period=self.params.fast)
        self.slow_ma = bt.ind.SMA(period=self.params.slow)
        self.crossover = bt.ind.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()
'''


def list_custom_strategies() -> list[str]:
    """列出所有自定义策略文件名（不含 .py）。"""
    if not os.path.isdir(_CUSTOM_DIR):
        return []
    return sorted([
        f[:-3] for f in os.listdir(_CUSTOM_DIR)
        if f.endswith(".py") and not f.startswith("_")
    ])


def compile_strategy(code: str) -> tuple[bool, str]:
    """尝试编译策略代码，返回 (是否成功, 类名或错误信息)。"""
    try:
        compile(code, "<strategy>", "exec")
    except SyntaxError as e:
        return False, f"语法错误: {e}"

    ns = {}
    try:
        exec(code, ns)
    except Exception as e:
        return False, f"执行错误: {e}"

    if "MyStrategy" not in ns:
        return False, "未找到 MyStrategy 类（类名必须为 MyStrategy）"

    cls = ns["MyStrategy"]
    if not hasattr(cls, "params"):
        return False, "MyStrategy 必须定义 params"

    return True, "MyStrategy"


st.set_page_config(page_title="策略编辑器", page_icon="📝", layout="wide")
st.title("📝 策略编辑器")

os.makedirs(_CUSTOM_DIR, exist_ok=True)

# ---- 侧边栏：策略列表 ----
with st.sidebar:
    st.header("策略列表")

    st.caption("内置策略（只读）")
    from strategies import STRATEGY_REGISTRY
    for name in STRATEGY_REGISTRY:
        st.markdown(f"- {name}")

    st.divider()
    st.caption("自定义策略")

    custom = list_custom_strategies()

    if not custom:
        st.info("暂无自定义策略，点击「新建策略」")

    selected_custom = st.selectbox("选择编辑", [""] + custom, key="select_custom")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ 新建策略", use_container_width=True):
            st.session_state.edit_name = ""
            st.session_state.edit_code = TEMPLATE
            st.rerun()
    with col2:
        if selected_custom:
            if st.button("🗑 删除", use_container_width=True):
                os.remove(os.path.join(_CUSTOM_DIR, f"{selected_custom}.py"))
                st.cache_data.clear()
                st.session_state.pop("edit_name", None)
                st.session_state.pop("edit_code", None)
                st.rerun()

# ---- 主区域：编辑器 ----
if selected_custom and not st.session_state.get("edit_code"):
    path = os.path.join(_CUSTOM_DIR, f"{selected_custom}.py")
    with open(path) as f:
        st.session_state.edit_code = f.read()
    st.session_state.edit_name = selected_custom

if "edit_code" not in st.session_state:
    st.session_state.edit_code = TEMPLATE
if "edit_name" not in st.session_state:
    st.session_state.edit_name = ""

edit_name = st.text_input(
    "策略名称（英文，用于保存文件名）",
    value=st.session_state.edit_name,
    placeholder="my_strategy",
    key="name_input",
)

edit_code = st.text_area(
    "策略代码",
    value=st.session_state.edit_code,
    height=500,
    key="code_input",
)

# 同步到 session state
if edit_code != st.session_state.edit_code:
    st.session_state.edit_code = edit_code
if edit_name != st.session_state.edit_name:
    st.session_state.edit_name = edit_name

col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    if st.button("🧪 编译测试", use_container_width=True):
        ok, msg = compile_strategy(edit_code)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

with col2:
    if st.button("💾 保存", use_container_width=True, type="primary"):
        if not edit_name.strip():
            st.error("请输入策略名称")
        else:
            ok, msg = compile_strategy(edit_code)
            if not ok:
                st.error(f"编译失败，无法保存: {msg}")
            else:
                safe_name = edit_name.strip().replace(" ", "_").replace("/", "_")
                path = os.path.join(_CUSTOM_DIR, f"{safe_name}.py")
                with open(path, "w") as f:
                    f.write(edit_code)
                importlib.invalidate_caches()
                st.success(f"已保存到 {path}")
                # 刷新回测页面的策略缓存
                st.cache_data.clear()

with col3:
    st.caption("保存后需切换到「策略回测」页面，策略会自动出现在下拉列表中")

# ---- 编写指南 ----
with st.expander("📖 编写指南"):
    st.markdown("""
### 基本要求
- 类名必须是 `MyStrategy`，继承 `backtrader.Strategy`
- 必须定义 `params` 元组
- 交易指令：`self.buy()` / `self.sell()` / `self.close()`

### 常用指标
```python
bt.ind.SMA(period=N)         # 简单移动均线
bt.ind.EMA(period=N)         # 指数移动均线
bt.ind.MACD(fast, slow, sig) # MACD 指标 → .macd / .signal
bt.ind.RSI(period=N)         # RSI 指标
bt.ind.BollingerBands()      # 布林带 → .top / .mid / .bot
bt.ind.ATR(period=N)         # 平均真实波幅
bt.ind.CrossOver(a, b)       # a 上穿 b → +1, 下穿 → -1
```

### 常用属性
```python
self.position                # 当前持仓对象
self.position.size           # 持仓数量（0 = 空仓）
self.data.close[0]           # 当前收盘价
self.data.close[-1]          # 前一K线收盘价
self.broker.getcash()        # 可用资金
self.broker.getvalue()       # 总资产
```

### 示例：带止损的双均线
```python
class MyStrategy(bt.Strategy):
    params = (("fast", 5), ("slow", 20), ("stop_pct", 0.05))

    def __init__(self):
        self.fast_ma = bt.ind.SMA(period=self.params.fast)
        self.slow_ma = bt.ind.SMA(period=self.params.slow)
        self.crossover = bt.ind.CrossOver(self.fast_ma, self.slow_ma)
        self.entry_price = 0

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
                self.entry_price = self.data.close[0]
        else:
            if self.crossover < 0:
                self.close()
            elif self.data.close[0] < self.entry_price * (1 - self.params.stop_pct):
                self.close()
```
""")
