import importlib
import json
import os
import sys
import traceback

import streamlit as st

from factors import ALL_FACTORS
from app.utils.ml_config_manager import (
    create_ml_config, get_ml_config, get_ml_config_by_name,
    list_ml_configs, update_ml_config, delete_ml_config, seed_builtin_configs,
)

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


# -- 因子类别分组 --
FACTOR_CATEGORIES: dict[str, list[str]] = {
    "Alpha101": [],
    "自定义": [],
    "Alpha191-换手率": [],
    "Alpha191-日内": [],
    "Alpha191-资金流": [],
    "Alpha191-缺口": [],
    "Alpha191-波动率": [],
    "Alpha191-流动性": [],
    "基本面": [],
}

# 按模块来源分组因子名
import factors.alpha101 as _a101
import factors.custom as _cust
import factors.alpha191_turnover as _a191t
import factors.alpha191_intraday as _a191i
import factors.alpha191_flow as _a191f
import factors.alpha191_gap as _a191g
import factors.alpha191_vol as _a191v
import factors.alpha191_liquidity as _a191l
import factors.fundamental as _fund

FACTOR_CATEGORIES["Alpha101"] = sorted(_a101.ALPHA101_FUNCTIONS.keys())
FACTOR_CATEGORIES["自定义"] = sorted(_cust.CUSTOM_FACTORS.keys())
FACTOR_CATEGORIES["Alpha191-换手率"] = sorted(_a191t.ALPHA191_TURNOVER.keys())
FACTOR_CATEGORIES["Alpha191-日内"] = sorted(_a191i.ALPHA191_INTRADAY.keys())
FACTOR_CATEGORIES["Alpha191-资金流"] = sorted(_a191f.ALPHA191_FLOW.keys())
FACTOR_CATEGORIES["Alpha191-缺口"] = sorted(_a191g.ALPHA191_GAP.keys())
FACTOR_CATEGORIES["Alpha191-波动率"] = sorted(_a191v.ALPHA191_VOL.keys())
FACTOR_CATEGORIES["Alpha191-流动性"] = sorted(_a191l.ALPHA191_LIQUIDITY.keys())
FACTOR_CATEGORIES["基本面"] = sorted(_fund.FUNDAMENTAL_FACTORS.keys())

ML_DEFAULTS = {
    "description": "",
    "factor_names": [],
    "ic_threshold": 0.02,
    "t_threshold": 2.0,
    "orthogonal_threshold": 0.7,
    "label_mode": "binary",
    "forward_days": 1,
    "train_years": 3,
    "val_years": 1,
    "model_type": "ensemble",
    "top_n": 15,
    "rebalance_mode": "ndrop",
    "ndrop_n": 2,
    "max_single": 0.10,
    "max_industry": 0.30,
    "stop_loss_pct": 0.08,
    "atr_multiplier": 1.5,
    "atr_period": 20,
    "portfolio_dd_threshold": 0.20,
    "portfolio_dd_reduce_to": 0.50,
    "max_dd_limit": 0.25,
    "stock_pool": "",
    "stock_count": 500,
}


st.set_page_config(page_title="策略编辑器", page_icon="📝", layout="wide")
st.title("📝 策略编辑器")

os.makedirs(_CUSTOM_DIR, exist_ok=True)

tab1, tab2 = st.tabs(["📝 静态策略代码", "🧠 ML 策略配置"])

# ============================================================
# Tab 1: 静态策略编辑器
# ============================================================
with tab1:
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

# ============================================================
# Tab 2: ML 策略配置编辑器
# ============================================================
with tab2:
    # -- 确保内置预设存在 --
    seed_builtin_configs()

    # -- 侧边栏：ML 配置列表 --
    with st.sidebar:
        st.header("ML 策略配置")

        ml_configs_df = list_ml_configs()

        if st.button("🆕 新建配置", use_container_width=True):
            st.session_state.ml_edit_id = None
            st.session_state.ml_form = dict(ML_DEFAULTS)
            st.session_state.ml_form["name"] = ""
            st.rerun()

        if ml_configs_df.empty:
            st.info("暂无 ML 策略配置")
        else:
            config_names = [""] + ml_configs_df["name"].tolist()
            selected_ml_name = st.selectbox("选择编辑", config_names, key="ml_select")

            if selected_ml_name:
                cfg = get_ml_config_by_name(selected_ml_name)
                if cfg and st.session_state.get("ml_edit_id") != cfg["id"]:
                    st.session_state.ml_edit_id = cfg["id"]
                    st.session_state.ml_form = {
                        k: cfg.get(k, ML_DEFAULTS.get(k))
                        for k in ML_DEFAULTS
                    }
                    st.session_state.ml_form["name"] = cfg["name"]
                    st.rerun()

                if st.button("🗑 删除此配置", use_container_width=True):
                    delete_ml_config(cfg["id"])
                    st.cache_data.clear()
                    st.session_state.pop("ml_edit_id", None)
                    st.session_state.pop("ml_form", None)
                    st.success(f"已删除 {selected_ml_name}")
                    st.rerun()

    # -- 主区域：编辑表单 --
    if "ml_form" not in st.session_state:
        st.session_state.ml_form = dict(ML_DEFAULTS)
        st.session_state.ml_form["name"] = ""
        st.session_state.ml_edit_id = None

    form = st.session_state.ml_form
    is_edit = st.session_state.ml_edit_id is not None

    st.subheader("✏️ 编辑配置" if is_edit else "🆕 新建 ML 策略配置")

    # ---- 基本信息 ----
    st.markdown("**基本信息**")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        form["name"] = st.text_input(
            "策略名称", value=form.get("name", ""),
            placeholder="例如：ML-低回撤精选", key="ml_name",
        )
    with col_b:
        form["description"] = st.text_input(
            "描述", value=form.get("description", ""),
            placeholder="策略说明", key="ml_desc",
        )

    st.divider()

    # ---- 因子选择 ----
    st.markdown("**因子选择**（留空 = 自动使用全部非基本面因子）")
    all_selected_factors: list[str] = form.get("factor_names", []) or []
    selected_set = set(all_selected_factors)

    # 快捷操作
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("全选因子", use_container_width=True, key="ml_fac_all"):
            all_selected_factors = sorted(ALL_FACTORS.keys())
            form["factor_names"] = all_selected_factors
            st.rerun()
    with c2:
        if st.button("清空选择", use_container_width=True, key="ml_fac_clear"):
            form["factor_names"] = []
            st.rerun()
    with c3:
        if st.button("仅非基本面", use_container_width=True, key="ml_fac_nofund"):
            fund_set = set(FACTOR_CATEGORIES.get("基本面", []))
            all_selected_factors = sorted(set(ALL_FACTORS.keys()) - fund_set)
            form["factor_names"] = all_selected_factors
            st.rerun()

    # 按类别显示因子
    for cat_name, cat_factors in FACTOR_CATEGORIES.items():
        if not cat_factors:
            continue
        with st.expander(f"{cat_name} ({len(cat_factors)} 个因子)", expanded=False):
            selected_in_cat = [f for f in cat_factors if f in selected_set]
            picked = st.multiselect(
                f"选择 {cat_name} 因子",
                options=cat_factors,
                default=selected_in_cat,
                key=f"ml_factor_{cat_name}",
                label_visibility="collapsed",
            )
            # 同步：把该类别的选择合并到总列表
            for f in cat_factors:
                if f in picked and f not in all_selected_factors:
                    all_selected_factors.append(f)
                elif f not in picked and f in all_selected_factors:
                    all_selected_factors.remove(f)

    form["factor_names"] = all_selected_factors
    st.caption(f"当前已选 {len(all_selected_factors)} / {len(ALL_FACTORS)} 个因子")

    st.divider()

    # ---- 因子筛选参数 ----
    st.markdown("**因子筛选阈值**")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        form["ic_threshold"] = st.number_input(
            "IC 阈值", value=float(form.get("ic_threshold", 0.02)),
            min_value=0.0, max_value=0.5, step=0.005, format="%.3f",
            help="|mean IC| > threshold 的因子保留",
        )
    with fc2:
        form["t_threshold"] = st.number_input(
            "T 值阈值", value=float(form.get("t_threshold", 2.0)),
            min_value=0.0, max_value=10.0, step=0.1,
            help="|t-statistic| > threshold 的因子保留",
        )
    with fc3:
        form["orthogonal_threshold"] = st.number_input(
            "正交阈值", value=float(form.get("orthogonal_threshold", 0.7)),
            min_value=0.0, max_value=1.0, step=0.05,
            help="Spearman 相关 < threshold 的因子保留",
        )

    st.divider()

    # ---- 标签与训练 ----
    st.markdown("**标签与训练参数**")
    tr1, tr2, tr3, tr4, tr5 = st.columns(5)
    with tr1:
        form["label_mode"] = st.selectbox(
            "标签模式", ["binary", "quantile"],
            index=0 if form.get("label_mode", "binary") == "binary" else 1,
            help="binary: 涨跌分类; quantile: 分位数回归",
        )
    with tr2:
        form["forward_days"] = st.number_input(
            "预测天数", value=int(form.get("forward_days", 1)),
            min_value=1, max_value=20, step=1,
        )
    with tr3:
        form["train_years"] = st.number_input(
            "训练年数", value=int(form.get("train_years", 3)),
            min_value=1, max_value=10, step=1,
        )
    with tr4:
        form["val_years"] = st.number_input(
            "验证年数", value=int(form.get("val_years", 1)),
            min_value=0, max_value=5, step=1,
        )
    with tr5:
        form["model_type"] = st.selectbox(
            "模型类型", ["xgboost", "lightgbm", "ensemble"],
            index={"xgboost": 0, "lightgbm": 1, "ensemble": 2}.get(
                form.get("model_type", "ensemble"), 2,
            ),
        )

    st.divider()

    # ---- 组合参数 ----
    st.markdown("**组合参数**")
    pf1, pf2, pf3, pf4, pf5 = st.columns(5)
    with pf1:
        form["top_n"] = st.number_input(
            "持仓数", value=int(form.get("top_n", 15)),
            min_value=5, max_value=100, step=5,
        )
    with pf2:
        form["rebalance_mode"] = st.selectbox(
            "调仓模式", ["ndrop", "full"],
            index=0 if form.get("rebalance_mode", "ndrop") == "ndrop" else 1,
            help="ndrop: 增量调仓; full: 全量换仓",
        )
    with pf3:
        form["ndrop_n"] = st.number_input(
            "NDrop 数量", value=int(form.get("ndrop_n", 2)),
            min_value=1, max_value=10, step=1,
        )
    with pf4:
        form["max_single"] = st.number_input(
            "单票上限", value=float(form.get("max_single", 0.10)),
            min_value=0.01, max_value=1.0, step=0.01, format="%.2f",
        )
    with pf5:
        form["max_industry"] = st.number_input(
            "行业上限", value=float(form.get("max_industry", 0.30)),
            min_value=0.05, max_value=1.0, step=0.05, format="%.2f",
        )

    st.divider()

    # ---- 风控参数 ----
    st.markdown("**风控参数**")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        form["stop_loss_pct"] = st.number_input(
            "止损比例", value=float(form.get("stop_loss_pct", 0.08)),
            min_value=0.0, max_value=0.5, step=0.01, format="%.2f",
            help="单票亏损超过此比例则止损",
        )
    with rc2:
        form["atr_multiplier"] = st.number_input(
            "ATR 乘数", value=float(form.get("atr_multiplier", 1.5)),
            min_value=0.5, max_value=5.0, step=0.1,
        )
    with rc3:
        form["atr_period"] = st.number_input(
            "ATR 周期", value=int(form.get("atr_period", 20)),
            min_value=5, max_value=60, step=5,
        )

    rc4, rc5, rc6 = st.columns(3)
    with rc4:
        form["portfolio_dd_threshold"] = st.number_input(
            "组合回撤阈值", value=float(form.get("portfolio_dd_threshold", 0.20)),
            min_value=0.05, max_value=0.5, step=0.01, format="%.2f",
            help="超过此回撤触发减仓",
        )
    with rc5:
        form["portfolio_dd_reduce_to"] = st.number_input(
            "回撤减仓至", value=float(form.get("portfolio_dd_reduce_to", 0.50)),
            min_value=0.1, max_value=1.0, step=0.05, format="%.2f",
            help="减仓后保留的仓位比例",
        )
    with rc6:
        form["max_dd_limit"] = st.number_input(
            "最大回撤清盘", value=float(form.get("max_dd_limit", 0.25)),
            min_value=0.1, max_value=0.6, step=0.01, format="%.2f",
            help="超过此回撤则清盘",
        )

    st.divider()

    # ---- 股票池 ----
    st.markdown("**股票池**")
    sp1, sp2 = st.columns(2)
    with sp1:
        form["stock_pool"] = st.text_input(
            "股票池名称", value=form.get("stock_pool", ""),
            placeholder="留空使用全市场",
        )
    with sp2:
        form["stock_count"] = st.number_input(
            "候选股数", value=int(form.get("stock_count", 500)),
            min_value=50, max_value=5000, step=50,
            help="按流动性筛选的候选股票数量",
        )

    st.divider()

    # ---- 保存按钮 ----
    col_save1, col_save2 = st.columns([1, 4])
    with col_save1:
        if st.button("💾 保存配置", use_container_width=True, type="primary", key="ml_save"):
            if not form.get("name", "").strip():
                st.error("请输入策略名称")
            else:
                try:
                    if is_edit:
                        update_ml_config(st.session_state.ml_edit_id, **form)
                        st.success(f"已更新 {form['name']}")
                    else:
                        new_id = create_ml_config(form["name"], **{
                            k: v for k, v in form.items() if k != "name"
                        })
                        st.session_state.ml_edit_id = new_id
                        st.success(f"已创建 {form['name']} (ID: {new_id})")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败: {e}")
