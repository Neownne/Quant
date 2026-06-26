#!/usr/bin/env python
"""涨停复盘图片 OCR 分析 + 异动识别 + 策略关联。

用法:
    python scripts/analyze_hotpoint.py                    # 完整分析
    python scripts/analyze_hotpoint.py --no-ocr           # 跳过 OCR，用缓存
    python scripts/analyze_hotpoint.py --min-count 5      # 调高异动阈值
    python scripts/analyze_hotpoint.py --html-only        # 仅生成 HTML 报告
    python scripts/analyze_hotpoint.py --send-email       # 生成 HTML 并发送邮件

输入: hotpoint/*.png（每日涨停复盘简图）
输出:
    data/arsenal/hotpoint_master.csv          - 主数据（增量累积）
    data/arsenal/hotpoint_ocr_cache.json      - OCR 缓存
    data/arsenal/hotpoint_anomalies_{date}.csv - 异动关键词
    data/arsenal/hotpoint_leaders_{date}.csv   - 潜在龙头
    data/arsenal/hotpoint_strategy_{date}.csv  - 策略关联
    data/arsenal/hotpoint_report_{date}.html   - HTML 报告
"""
import sys, os, re, json, argparse, io, base64
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import pytesseract
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from sqlalchemy import text

# ── 常量 ──
HOTPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hotpoint")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "arsenal")
HOTPOINT_DIR_OUT = os.path.join(DATA_DIR, "hotpoint")
MASTER_CSV = os.path.join(HOTPOINT_DIR_OUT, "master.csv")
OCR_CACHE = os.path.join(HOTPOINT_DIR_OUT, "ocr_cache.json")

# 涨停乘数（v7.5 统一公式）
_LIMIT_MULT = {
    "688": 1.19899, "300": 1.19899, "301": 1.19899,
    "4": 1.29899, "8": 1.29899,
}
_DEFAULT_MULT = 1.09899

# ── OCR 常见错误纠正 ──
# 仅纠正明确的 OCR 字形错误（不含歧义词如 氢气/氦气）
# 原则：只在 OCR 产生无意义乱码或明显错字时才纠正
OCR_CORRECTIONS = {
    # 字形相似导致的明确错误
    "人金融": "金融",
    "大入金融": "大金融",
    "轮札": "轮机",
    "轮几": "轮机",
    "城真燃气": "城镇燃气",
    "城填燃气": "城镇燃气",
    # 常见 OCR 噪声碎片
    "Al": "AI",
    "AlI": "AI",
}

# OCR 歧义词（需要人工判断，不做自动纠正）
# 氢气/氦气、淡气/氮气、然气/燃气 等 —— 由 DB 验证环节处理

# 通用词过滤
STOP_WORDS = {
    "涨停", "封板", "首板", "连板", "涨跌", "停牌",
    "跌停", "炸板", "开板", "破板", "撬板",
    "涨停板", "跌停板", "一字板", "T字板",
    "股票", "个股", "强势", "反弹", "新高",
    "次新", "新股", "白马", "蓝筹", "龙头",
    "日线", "周线", "月线",
}

# 板块噪音词（从 web/routes/api.py 复用）
NOISE_BOARDS = {
    "沪股通", "深股通", "融资融券", "机构重仓", "标准普尔", "富时罗素",
    "MSCI", "证金", "汇金", "社保", "QFII", "举牌", "预增", "预减",
    "预亏", "解禁", "转融通", "央国企", "国资云", "国企改革",
    "沪企", "自贸", "特区", "振兴", "经济带", "大湾区", "城市群",
    "新区", "最近多板", "百日新高", "历史新高", "近期新高",
    "破发", "破增发", "深成500", "中证500", "沪深300",
}


# ══════════════════════════════════════════════════════════════════════
# Phase 1: OCR 提取（增量式）
# ══════════════════════════════════════════════════════════════════════

def load_ocr_cache():
    """加载 OCR 文本缓存。"""
    if os.path.exists(OCR_CACHE):
        with open(OCR_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ocr_cache(cache):
    os.makedirs(HOTPOINT_DIR_OUT, exist_ok=True)
    with open(OCR_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_master_dates():
    """返回主 CSV 中已有的日期集合。"""
    if os.path.exists(MASTER_CSV):
        df = pd.read_csv(MASTER_CSV, dtype={"code": str})
        return set(df["date"].unique())
    return set()


def ocr_image(filepath):
    """对单张图片执行 OCR（Tesseract），返回原始文本。"""
    img = Image.open(filepath)
    text = pytesseract.image_to_string(img, lang="chi_sim+eng")
    return text


# ── Surya OCR 后端（VLM，高精度但慢）──

_surya_predictor = None


def _get_surya():
    """延迟加载 Surya 模型（单例）。"""
    global _surya_predictor
    if _surya_predictor is None:
        import os as _os
        _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from surya.recognition import RecognitionPredictor
        from surya.inference import SuryaInferenceManager
        _surya_predictor = RecognitionPredictor(SuryaInferenceManager())
    return _surya_predictor


def ocr_image_surya(filepath):
    """用 Surya VLM 做 OCR，返回结构化记录列表（直接从 HTML 表格解析）。"""
    from html.parser import HTMLParser

    img = Image.open(filepath)
    rec = _get_surya()
    results = rec([img], full_page=True)

    all_records = []
    for page in results:
        d = page.model_dump()
        for block in d.get("blocks", []):
            if block.get("label") != "Table" or block.get("skipped"):
                continue
            html = block.get("html", "")
            if not html:
                continue
            records = _parse_surya_table_html(html)
            all_records.extend(records)

    return all_records


def _parse_surya_table_html(html):
    """从 Surya 输出的 HTML <table> 中提取结构化涨停记录。

    Surya 输出格式：
      <tr><td colspan="7"><b>医疗医药*14</b></td></tr>   ← 板块头
      <tr><td>3天3板</td><td>600851.SH</td>...           ← 数据行
    """
    from html.parser import HTMLParser

    records = []
    current_sector = "其他"

    # 简易 HTML 表格解析（不依赖第三方库）
    # 去掉 HTML 标签，逐行处理
    # 匹配 <tr>...</tr>
    tr_pattern = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
    td_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
    strip_html = re.compile(r"<[^>]+>")

    for tr_match in tr_pattern.finditer(html):
        tr_content = tr_match.group(1)
        tds = td_pattern.findall(tr_content)

        if not tds:
            continue

        # 检查 colspan
        colspan_match = re.search(r'colspan\s*=\s*["\']?7["\']?', tr_match.group(0))

        if colspan_match or len(tds) == 1:
            # 板块头行
            text = strip_html.sub("", tds[0]).strip()
            sec_m = re.match(r"^(.+?)\*(\d+)\s*$", text)
            if sec_m:
                current_sector = sec_m.group(1).strip()
            continue

        if len(tds) < 7:
            continue

        # 数据行：7 列
        # 板数 | 代码 | 名称 | 涨停时间 | 流通市值 | 成交额 | 关键词
        board_raw = strip_html.sub("", tds[0]).strip()
        code_raw = strip_html.sub("", tds[1]).strip()
        name_raw = strip_html.sub("", tds[2]).strip()
        time_raw = strip_html.sub("", tds[3]).strip()
        mcap_raw = strip_html.sub("", tds[4]).strip()
        vol_raw = strip_html.sub("", tds[5]).strip()
        kw_raw = strip_html.sub("", tds[6]).strip()

        # 解析代码
        code_m = RE_CODE.search(code_raw)
        if not code_m:
            continue
        code_6 = code_m.group(1)
        market = code_m.group(2).upper()

        # 板数
        board_count, board_desc = normalize_board_count(board_raw)

        # 数值
        mcap = _parse_num(mcap_raw)
        volume = _parse_num(vol_raw)

        # 关键词
        keywords = split_keywords(kw_raw)

        records.append({
            "date": "",  # 由调用者填充
            "code": code_6,
            "market": market,
            "name": name_raw,
            "board_count": board_count,
            "board_desc": board_desc,
            "limit_time": time_raw,
            "mcap": mcap,
            "volume": volume,
            "sector": current_sector,
            "keywords": "|".join(keywords) if keywords else "",
            "keyword_list": keywords,
            "raw_name": name_raw,
            "confidence": 0.95,  # Surya 高置信度
            "validated": False,
        })

    return records


def _parse_num(s):
    """从字符串提取数字。"""
    s = s.strip().replace(",", "").replace("，", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_all_images(skip_ocr=False, use_surya=True):
    """遍历 hotpoint/，增量 OCR 新图片。返回 {date: raw_text 或 records}。

    use_surya=True:  使用 Surya VLM（高精度，~2min/张）
    use_surya=False: 使用 Tesseract（快速，~10s/张）
    """
    cache = load_ocr_cache()
    existing_dates = load_master_dates()

    png_files = sorted([f for f in os.listdir(HOTPOINT_DIR) if f.endswith(".png")])
    logger.info(f"hotpoint/ 共 {len(png_files)} 张图片")

    for fname in png_files:
        dt = fname.replace(".png", "")
        if dt in existing_dates:
            logger.debug(f"  {dt}: 已存在主 CSV，跳过")
            continue
        # Surya 缓存了 → 跳过；Tesseract 缓存了 → 跳过
        if dt in cache:
            cached_entry = cache[dt]
            if isinstance(cached_entry, dict):
                if cached_entry.get("engine") == "surya" and use_surya:
                    logger.debug(f"  {dt}: 已有 Surya 缓存")
                    continue
                if cached_entry.get("engine") != "surya" and not use_surya:
                    logger.debug(f"  {dt}: 已有 Tesseract 缓存")
                    continue
            elif isinstance(cached_entry, str):
                # 旧缓存格式
                logger.debug(f"  {dt}: 已有 OCR 缓存")
                continue

        if skip_ocr:
            continue

        filepath = os.path.join(HOTPOINT_DIR, fname)

        if use_surya:
            logger.info(f"  Surya OCR: {dt} ({fname})")
            try:
                records = ocr_image_surya(filepath)
                # 填充日期
                for r in records:
                    r["date"] = dt
                # 缓存为 JSON（Surya 结果直接存结构化数据）
                cache[dt] = {"engine": "surya", "records": records}
                logger.info(f"    → {len(records)} 条记录")
                save_ocr_cache(cache)  # 增量保存
            except Exception as e:
                logger.error(f"  Surya 失败 {dt}: {e}，回落 Tesseract")
                text = ocr_image(filepath)
                cache[dt] = {"engine": "tesseract", "text": text}
                logger.info(f"    → {len(text)} 字符")
                save_ocr_cache(cache)
        else:
            logger.info(f"  Tesseract OCR: {dt} ({fname})")
            try:
                text = ocr_image(filepath)
                cache[dt] = {"engine": "tesseract", "text": text}
                logger.info(f"    → {len(text)} 字符")
            except Exception as e:
                logger.error(f"  OCR 失败 {dt}: {e}")
                cache[dt] = {"engine": "none", "text": ""}

    save_ocr_cache(cache)
    return cache


# ══════════════════════════════════════════════════════════════════════
# Phase 2: 结构化解析
# ══════════════════════════════════════════════════════════════════════

# Regex patterns
RE_CODE = re.compile(r"(\d{6})\.(SH|SZ|BJ)", re.IGNORECASE)
RE_SECTOR = re.compile(r"^(.+?)\*(\d+)\s*$")
RE_TIME = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")
RE_BOARD = re.compile(r"(\d+)天(\d+)板")
RE_NUMBER = re.compile(r"(\d+\.?\d*)")


def normalize_board_count(raw):
    """解析板数文本 → (板数, 描述)。"""
    raw = raw.strip()
    # "N天N板"
    m = RE_BOARD.search(raw)
    if m:
        return int(m.group(1)), m.group(0)
    # 纯数字
    m2 = re.match(r"^(\d+)$", raw)
    if m2:
        return int(m2.group(1)), raw
    # 默认首板
    return 1, "首板"


def split_keywords(text):
    """拆分关键词，过滤通用词。"""
    if not text:
        return []
    # 按 + / # ; 分割
    parts = re.split(r"[+/#;，,、\s]+", text)
    result = []
    for p in parts:
        # 清理标点符号和 OCR 噪声
        strip_chars = '“”「」『』‘’（）()\"\'*°<>《》-_.:：'
        p = p.strip().strip(strip_chars)
        # 过滤空、纯数字、纯符号、通用词
        if not p or len(p) < 2:
            continue
        if re.match(r"^[\d\.\-]+$", p):
            continue
        if p in STOP_WORDS:
            continue
        # 过滤包含过多 OCR 乱码的
        if re.search(r"[A-Za-z]{4,}", p) and not re.search(r"[一-鿿]", p):
            continue
        result.append(p)
    # OCR 字形纠错
    result = [OCR_CORRECTIONS.get(kw, kw) for kw in result]

    # 去重
    seen = set()
    uniq = []
    for kw in result:
        if kw not in seen:
            seen.add(kw)
            uniq.append(kw)
    return uniq


def parse_raw_text(date_str, raw_text):
    """从 OCR 原始文本中提取结构化记录。"""
    lines = raw_text.split("\n")
    records = []
    current_sector = "其他"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检测板块头：板块名*N
        sec_m = RE_SECTOR.match(line)
        if sec_m:
            sec_name = sec_m.group(1).strip()
            # 排除误匹配（比如包含 "天" 的板数描述）
            if not re.search(r"\d天\d板", sec_name) and len(sec_name) <= 20:
                current_sector = sec_name
            continue

        # 检测股票行：必须有 6位.SH/SZ/BJ
        code_matches = list(RE_CODE.finditer(line))
        if not code_matches:
            continue

        for cm in code_matches:
            code_raw = cm.group(0)        # 600851.SH
            code_6 = cm.group(1)          # 600851
            market = cm.group(2).upper()  # SH
            pos = cm.start()

            # ── 左侧：板数 ──
            _strip_chars = '“”「」『』‘’*°,，;； '
            left = line[:pos].strip().strip(_strip_chars)
            # 去除末尾 OCR 噪声
            left = re.sub(r'[^一-鿿\d天板块a-zA-Z]+\s*$', '', left).strip()
            board_count, board_desc = normalize_board_count(left if left else "1")

            # ── 右侧：名称 + 时间 + 市值 + 成交额 + 关键词 ──
            right = line[cm.end():].strip()

            # 提取时间
            time_match = RE_TIME.search(right)
            limit_time = ""
            if time_match:
                limit_time = time_match.group(0)
                name_part = right[:time_match.start()].strip()
                after_time = right[time_match.end():].strip()
            else:
                name_part = right
                after_time = ""

            # 清洗名称
            name = _clean_name(name_part)

            # 从 after_time 提取数值（市值、成交额）和关键词
            mcap = None
            volume = None

            # 尝试提取两个数字
            nums = RE_NUMBER.findall(after_time)
            if len(nums) >= 2:
                try:
                    # 第一个有效数字 = 市值
                    for n in nums:
                        val = float(n)
                        if 1 < val < 10000 and mcap is None:
                            mcap = val
                        elif 1 < val < 1000 and mcap is not None and volume is None:
                            volume = val
                except ValueError:
                    pass

            # 关键词：after_time 中去掉数值部分后的剩余
            kw_text = after_time
            for n in nums[:2]:
                kw_text = kw_text.replace(str(n), "", 1)
            kw_text = re.sub(r"\s+", " ", kw_text).strip()
            keywords = split_keywords(kw_text)


            records.append({
                "date": date_str,
                "code": code_6,
                "market": market,
                "name": name,
                "board_count": board_count,
                "board_desc": board_desc,
                "limit_time": limit_time,
                "mcap": mcap,
                "volume": volume,
                "sector": current_sector,
                "keywords": "|".join(keywords) if keywords else "",
                "keyword_list": keywords,
                "raw_name": name,
                "confidence": 1.0,  # 初始置信度，DB 验证后可能降低
                "validated": False,
            })

    return records


def _clean_name(name_part):
    """清洗 OCR 识别的股票名称。"""
    # 去除乱码字符
    name = re.sub(r'[^一-鿿A-Za-z0-9·\.\-\*]', '', name_part)
    # 去除前置/后置标点
    _strip_chars = '“”「」『』‘’*°,，;；.:：-_=+ '
    name = name.strip(_strip_chars)
    # 如果名称太长（>8个中文字），取前 8 个
    if len(name) > 8:
        # 尝试在常见标点处截断
        for sep in [" ", ",", "，", ".", ":", "："]:
            if sep in name[:8]:
                name = name[:name.index(sep)]
                break
        else:
            name = name[:8]
    # 去除末尾的单个英文字母/数字
    name = re.sub(r'[A-Za-z0-9]$', '', name)
    return name.strip()


# ══════════════════════════════════════════════════════════════════════
# Phase 3: DB 交叉验证与纠错
# ══════════════════════════════════════════════════════════════════════

def _get_multiplier(code_6):
    for prefix, mult in _LIMIT_MULT.items():
        if code_6.startswith(prefix):
            return mult
    return _DEFAULT_MULT


def _calc_limit_price(prev_close, code_6):
    return round(prev_close * _get_multiplier(code_6), 4)


def validate_and_correct(records, engine):
    """DB 交叉验证：代码存在性 → 名称纠错 → 涨停验证。"""
    if not records:
        return records

    # 收集所有 code
    codes = list(set(r["code"] for r in records))
    dates = sorted(set(r["date"] for r in records))

    # ── 批量查询 stock_basic ──
    with engine.connect() as conn:
        basic_df = pd.read_sql(
            text("SELECT code, name FROM stock_basic WHERE code = ANY(:codes)"),
            conn, params={"codes": codes},
        )
    db_codes = set(basic_df["code"].values)
    code_to_dbname = dict(zip(basic_df["code"], basic_df["name"]))

    # 规范化日期：records 中的 date 是 YYYYMMDD，DB 查询需要 YYYY-MM-DD
    def _norm_date(d):
        if len(d) == 8:
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return d

    # ── 批量查询 stock_daily（最近日期）──
    min_date = _norm_date(min(dates))
    max_date = _norm_date(max(dates))
    with engine.connect() as conn:
        daily_df = pd.read_sql(
            text("""
                SELECT code, trade_date, close,
                       LAG(close) OVER (PARTITION BY code ORDER BY trade_date) AS prev_close
                FROM stock_daily
                WHERE code = ANY(:codes)
                  AND trade_date BETWEEN :start AND :end
                ORDER BY code, trade_date
            """),
            conn, params={"codes": codes, "start": min_date, "end": max_date},
        )
    # 构建 (code, trade_date) → (close, prev_close) 映射
    daily_map = {}
    for _, row in daily_df.iterrows():
        daily_map[(row["code"], str(row["trade_date"]))] = (
            row["close"], row.get("prev_close")
        )

    # ── 逐条验证 ──
    corrected = []
    for r in records:
        code = r["code"]
        dt_ymd = _norm_date(r["date"])  # YYYYMMDD → YYYY-MM-DD for DB lookup
        name = r["name"]

        # 1. 代码格式检查
        if not re.match(r"^\d{6}$", code):
            logger.debug(f"  [{r['date']}] 代码格式错误: {code}")
            continue

        # 2. 代码是否存在？不在则尝试纠错
        if code not in db_codes:
            fixed = _try_fix_code(code, name, r["date"], db_codes, code_to_dbname, engine)
            if fixed:
                code = fixed
                r["code"] = code
            else:
                logger.debug(f"  [{r['date']}] 代码 {code} 不存在于 DB 且无法纠错，丢弃")
                continue

        # 3. 名称纠错
        db_name = code_to_dbname.get(code, "")
        if db_name and name != db_name:
            similarity = SequenceMatcher(None, name, db_name).ratio()
            if similarity < 0.6:
                # OCR 名称不可靠，用 DB 名称替换
                logger.debug(f"  [{r['date']}] {code} 名称纠错: '{name}' → '{db_name}' (sim={similarity:.2f})")
                r["name"] = db_name
                r["confidence"] = max(r["confidence"], 0.7)
            elif similarity < 0.9:
                r["name"] = db_name
                r["confidence"] = max(r["confidence"], 0.9)

        # 4. 涨停验证
        dkey = (code, dt_ymd)
        if dkey in daily_map:
            close, prev_close = daily_map[dkey]
            if prev_close is not None and pd.notna(prev_close) and prev_close > 0:
                limit_price = _calc_limit_price(prev_close, code)
                is_lu = close >= limit_price * 0.98  # 容差
                if not is_lu:
                    # OCR 识别了非涨停股 → 降低置信度但不丢弃
                    # （可能图片上的"涨停"定义更宽松，如盘中触板）
                    r["confidence"] = 0.5
                    logger.debug(f"  [{r['date']}] {code} {r['name']}: 涨停验证失败 "
                                f"(close={close}, limit={limit_price})")
        else:
            # 无日线数据，可能是新股或数据缺失
            r["confidence"] = 0.7

        r["validated"] = True
        corrected.append(r)

    logger.info(f"  DB 验证: {len(records)} → {len(corrected)} 条 "
                f"(丢弃 {len(records) - len(corrected)} 条)")
    return corrected


def _try_fix_code(code, name, dt, db_codes, code_to_dbname, engine):
    """代码不在 DB 中时，尝试多种方式纠错。"""
    # 策略 1：用名称反查代码
    if name and len(name) >= 2:
        for db_code, db_name in code_to_dbname.items():
            if db_name == name:
                logger.info(f"  [{dt}] 名称 '{name}' 反查代码: {code} → {db_code}")
                return db_code

    # 策略 2：用名称模糊匹配
    if name and len(name) >= 2:
        best_sim = 0
        best_code = None
        for db_code, db_name in code_to_dbname.items():
            sim = SequenceMatcher(None, name, db_name).ratio()
            if sim > best_sim:
                best_sim = sim
                best_code = db_code
        if best_sim > 0.8:
            logger.info(f"  [{dt}] 名称模糊匹配: '{name}' → '{code_to_dbname.get(best_code)}' "
                       f"(sim={best_sim:.2f}), {code} → {best_code}")
            return best_code

    # 策略 3：编辑距离相近的代码（OCR 可能识别错数字）
    if len(code) == 6:
        candidates = []
        for db_code in db_codes:
            # 同市场同前缀
            if db_code[:2] == code[:2]:
                dist = sum(1 for a, b in zip(code, db_code) if a != b)
                if dist <= 2:
                    candidates.append((dist, db_code))
        if candidates:
            candidates.sort()
            best_dist, best_code = candidates[0]
            logger.info(f"  [{dt}] 编辑距离纠错: {code} → {best_code} (dist={best_dist})")
            return best_code

    # 策略 4：与 stock_basic 所有代码做编辑距离（不限前缀）
    all_candidates = []
    for db_code in list(db_codes)[:500]:  # 采样避免性能问题
        dist = sum(1 for a, b in zip(code, db_code) if a != b)
        if dist <= 1:
            all_candidates.append((dist, db_code))
    if all_candidates:
        all_candidates.sort()
        best_dist, best_code = all_candidates[0]
        logger.info(f"  [{dt}] 全局编辑距离纠错: {code} → {best_code} (dist={best_dist})")
        return best_code

    return None


# ══════════════════════════════════════════════════════════════════════
# Phase 4: 统计分析
# ══════════════════════════════════════════════════════════════════════

def get_window_dates(master_df, window_days):
    """获取最近 N 个交易日的日期列表。"""
    all_dates = sorted(master_df["date"].unique())
    return all_dates[-window_days:]


def sector_freq(master_df, window_dates):
    """板块出现频次（按股票计数），排除"其他"，升序排列（低频优先）。"""
    df = master_df[master_df["date"].isin(window_dates)]
    df = df[df["sector"] != "其他"]
    freq = df.groupby("sector")["code"].count().sort_values(ascending=False)
    return freq


def keyword_freq(master_df, window_dates):
    """关键词出现频次（展开 keyword_list）。"""
    df = master_df[master_df["date"].isin(window_dates)]
    # 展开关键词 —— keyword_list 不存在时从 keywords 字符串恢复
    if "keyword_list" not in df.columns:
        df = df.copy()
        df["keyword_list"] = df["keywords"].apply(
            lambda x: [k for k in str(x).split("|") if k] if pd.notna(x) else []
        )
    rows = []
    for _, row in df.iterrows():
        kw_list = row.get("keyword_list") or []
        if not isinstance(kw_list, list):
            kw_list = []
        for kw in kw_list:
            rows.append({"keyword": kw, "code": row["code"], "date": row["date"], "sector": row["sector"]})
    if not rows:
        return pd.Series(dtype=int)
    kw_df = pd.DataFrame(rows)
    freq = kw_df.groupby("keyword")["code"].count().sort_values(ascending=False)
    return freq


def build_master_df(records_list):
    """将结构化记录列表转为 DataFrame，展开 keyword_list。"""
    all_records = []
    for records in records_list:
        for r in records:
            kws = r.pop("keyword_list", [])
            r["keywords"] = "|".join(kws) if kws else ""
            all_records.append(r)

    df = pd.DataFrame(all_records)
    if df.empty:
        return df

    # 按日期排序
    df = df.sort_values(["date", "sector", "code"]).reset_index(drop=True)
    return df


def _get_records_from_cache(dt, cache_entry, engine):
    """从缓存条目提取结构化记录（兼容 Surya / Tesseract 两种格式）。"""
    if isinstance(cache_entry, dict) and cache_entry.get("engine") == "surya":
        # Surya：记录已预解析，应用 OCR 纠错后验证
        records = cache_entry.get("records", [])
        for r in records:
            r["date"] = dt
            # 对已缓存的关键词做纠错
            if r.get("keyword_list"):
                r["keyword_list"] = [OCR_CORRECTIONS.get(kw, kw) for kw in r["keyword_list"]]
                r["keywords"] = "|".join(r["keyword_list"])
        return validate_and_correct(records, engine)
    elif isinstance(cache_entry, dict) and cache_entry.get("engine") in ("tesseract", "none"):
        raw_text = cache_entry.get("text", "")
    elif isinstance(cache_entry, str):
        # 旧缓存格式（纯文本）
        raw_text = cache_entry
    else:
        return []
    return validate_and_correct(parse_raw_text(dt, raw_text), engine)


def load_or_build_master(ocr_results, engine, skip_ocr=False):
    """加载已有主 CSV 或从头构建。"""
    if os.path.exists(MASTER_CSV) and not skip_ocr:
        existing = pd.read_csv(MASTER_CSV, dtype={"code": str, "date": str})
        existing["keyword_list"] = existing["keywords"].apply(
            lambda x: [k for k in str(x).split("|") if k] if pd.notna(x) else []
        )
        logger.info(f"加载已有主数据: {len(existing)} 条记录, "
                    f"{existing['date'].nunique()} 个交易日")
        return existing

    logger.info("构建主数据...")
    all_records = []
    for dt in sorted(ocr_results.keys()):
        logger.info(f"  解析: {dt}")
        records = _get_records_from_cache(dt, ocr_results[dt], engine)
        all_records.extend(records)
        logger.info(f"    → {len(records)} 条记录")

    df = build_master_df([all_records])
    if not df.empty:
        os.makedirs(HOTPOINT_DIR_OUT, exist_ok=True)
        df.to_csv(MASTER_CSV, index=False, encoding="utf-8")
        logger.info(f"主数据已保存: {MASTER_CSV} ({len(df)} 条)")

    return df


def append_new_dates(master_df, ocr_results, engine):
    """增量追加新日期的数据。"""
    existing_dates = set(master_df["date"].unique())
    new_dates = sorted(d for d in ocr_results if d not in existing_dates)

    if not new_dates:
        logger.info("无新日期需要追加")
        return master_df

    logger.info(f"追加 {len(new_dates)} 个新日期: {new_dates}")
    new_records = []
    for dt in new_dates:
        records = _get_records_from_cache(dt, ocr_results[dt], engine)
        new_records.extend(records)
        logger.info(f"  {dt}: {len(records)} 条")

    if new_records:
        new_df = build_master_df([new_records])
        master_df = pd.concat([master_df, new_df], ignore_index=True)
        master_df = master_df.sort_values(["date", "sector", "code"]).reset_index(drop=True)
        master_df.to_csv(MASTER_CSV, index=False, encoding="utf-8")
        logger.info(f"主数据已更新: {len(master_df)} 条 ({master_df['date'].nunique()} 个交易日)")

    return master_df


# ══════════════════════════════════════════════════════════════════════
# Phase 5: 异动识别
# ══════════════════════════════════════════════════════════════════════

def detect_anomalies(master_df, min_count=1):
    """识别集中冒出的关键词：前10日从未出现，近2日集中出现 ≥ min_count 次。"""
    all_dates = sorted(master_df["date"].unique())
    if len(all_dates) < 12:
        logger.warning(f"数据不足：仅 {len(all_dates)} 个交易日，需 ≥ 12")
        return []

    dates_recent = all_dates[-2:]       # 近2日
    dates_prior = all_dates[-12:-2]     # 前10日（排除近2日）

    kw_recent = keyword_freq(master_df, dates_recent)
    kw_prior = keyword_freq(master_df, dates_prior)

    if kw_recent.empty:
        return []

    anomalies = []
    for kw, c_recent in kw_recent.items():
        if c_recent < min_count:
            continue

        # 过滤 OCR 噪声：纯英文短词（<4字母且无中文）
        if len(kw) <= 3 and not re.search(r'[一-鿿]', kw):
            continue

        c_prior = kw_prior.get(kw, 0)
        if c_prior > 0:
            continue  # 前10日出现过，不算"集中冒出"

        # 找关联股票和板块
        recent_df = master_df[master_df["date"].isin(dates_recent)]
        kw_stocks = []
        for _, row in recent_df.iterrows():
            kws = row.get("keywords", "")
            if pd.isna(kws):
                continue
            if kw in str(kws).split("|"):
                kw_stocks.append({
                    "code": row["code"],
                    "name": row["name"],
                    "sector": row["sector"],
                    "date": row["date"],
                })

        anomaly = {
            "keyword": kw,
            "count_recent": int(c_recent),
            "count_prior": 0,
            "trend": "🆕集中冒出",
            "stocks": kw_stocks,
            "stock_codes": list(set(s["code"] for s in kw_stocks)),
            "stock_names": list(set(s["name"] for s in kw_stocks)),
            "sectors": list(set(s["sector"] for s in kw_stocks)),
            "recent_dates": dates_recent,
            "prior_dates": dates_prior,
        }
        anomalies.append(anomaly)

    # 上涨空间评估
    for a in anomalies:
        c = a["count_recent"]
        if c <= 2:
            a["stage"] = "🌱萌芽"
            a["stage_note"] = "刚冒头，关注度低，空间大"
        elif c <= 5:
            a["stage"] = "🔥扩散"
            a["stage_note"] = "正在发酵，跟风盘进场"
        else:
            a["stage"] = "🏔️高峰"
            a["stage_note"] = "已过热，警惕拥挤"

    # 按近2日频次降序
    anomalies.sort(key=lambda x: x["count_recent"], reverse=True)
    logger.info(f"异动检测: {len(anomalies)} 个集中冒出的关键词 (min_count={min_count})")
    return anomalies


# ══════════════════════════════════════════════════════════════════════
# Phase 6: 潜在龙头发现
# ══════════════════════════════════════════════════════════════════════

def find_leader_stocks(keyword, engine, target_date, recent_days=5):
    """从 DB 概念板块查找关键词相关的潜在龙头股。"""
    if not keyword or len(keyword) < 2:
        return []

    leaders = []
    try:
        with engine.connect() as conn:
            # 1. concept_board LIKE '%keyword%'
            boards = pd.read_sql(
                text("SELECT code, name FROM concept_board WHERE name LIKE :kw"),
                conn, params={"kw": f"%{keyword}%"},
            )

            if boards.empty:
                return []

            # 过滤噪音板块
            board_codes = []
            for _, b in boards.iterrows():
                if b["name"] not in NOISE_BOARDS:
                    board_codes.append(b["code"])

            if not board_codes:
                return []

            # 2. concept_stock → 所有成分股
            stocks = pd.read_sql(
                text("""
                    SELECT DISTINCT cs.stock_code, sb.name, sb.industry
                    FROM concept_stock cs
                    JOIN stock_basic sb ON cs.stock_code = sb.code
                    WHERE cs.board_code = ANY(:bc)
                      AND sb.is_st = FALSE
                """),
                conn, params={"bc": board_codes},
            )

            if stocks.empty:
                return []

            stock_codes = stocks["stock_code"].tolist()

            # 3. 近 N 日涨停筛选
            start_d = (pd.Timestamp(target_date) - timedelta(days=recent_days + 5)).strftime("%Y-%m-%d")
            daily = pd.read_sql(
                text("""
                    SELECT code, trade_date, close,
                           LAG(close) OVER (PARTITION BY code ORDER BY trade_date) AS prev_close
                    FROM stock_daily
                    WHERE code = ANY(:sc) AND trade_date BETWEEN :start AND :end
                    ORDER BY code, trade_date
                """),
                conn, params={"sc": stock_codes, "start": start_d, "end": target_date},
            )

        if daily.empty:
            return []

        # 计算涨停
        daily["is_lu"] = daily.apply(
            lambda r: (
                r["close"] >= _calc_limit_price(r["prev_close"], r["code"]) * 0.98
                if pd.notna(r["prev_close"]) and r["prev_close"] > 0 else False
            ), axis=1,
        )

        # 聚合：每只股票近 N 日涨停次数、最新连板
        lu_daily = daily[daily["is_lu"]]
        for code in stock_codes:
            code_lu = lu_daily[lu_daily["code"] == code]
            lu_count = len(code_lu)
            if lu_count == 0:
                continue

            # 计算连板数
            code_dates = sorted(code_lu["trade_date"].unique())
            streak = 1
            for i in range(len(code_dates) - 1, 0, -1):
                d1 = pd.Timestamp(code_dates[i])
                d2 = pd.Timestamp(code_dates[i - 1])
                if (d1 - d2).days <= 3:
                    streak += 1
                else:
                    break

            info = stocks[stocks["stock_code"] == code]
            name = info["name"].values[0] if not info.empty else ""
            industry = info["industry"].values[0] if not info.empty else ""

            leaders.append({
                "code": code,
                "name": name,
                "industry": industry,
                "lu_count_5d": lu_count,
                "streak": streak,
                "keyword": keyword,
            })

        leaders.sort(key=lambda x: (-x["streak"], -x["lu_count_5d"]))

    except Exception as e:
        logger.warning(f"  龙头发现失败 '{keyword}': {e}")

    return leaders


def discover_all_leaders(anomalies, engine, target_date):
    """对所有异动关键词发现潜在龙头。"""
    all_leaders = []
    seen_codes = set()

    for a in anomalies:
        kw = a["keyword"]
        leaders = find_leader_stocks(kw, engine, target_date)
        for l in leaders:
            if l["code"] not in seen_codes:
                seen_codes.add(l["code"])
                all_leaders.append(l)
        logger.info(f"  '{kw}': {len(leaders)} 只潜在龙头")

    return all_leaders


# ══════════════════════════════════════════════════════════════════════
# Phase 7: 策略关联
# ══════════════════════════════════════════════════════════════════════

def cross_ref_strategies(anomalies, master_df, limit_up_pool, yaogu_pool, bull_pool):
    """异动股票与三个策略池交叉比对 + 启动阶段判定。"""

    # 获取池内代码集合
    lu_codes = set()
    yaogu_codes = set()
    bull_codes = set()

    if limit_up_pool is not None and not limit_up_pool.empty:
        lu_codes = set(limit_up_pool["code"].astype(str).values) if "code" in limit_up_pool.columns else set()
    if yaogu_pool is not None and not yaogu_pool.empty:
        yaogu_codes = set(yaogu_pool["code"].astype(str).values) if "code" in yaogu_pool.columns else set()
    if bull_pool is not None and not bull_pool.empty:
        bull_codes = set(bull_pool["code"].astype(str).values) if "code" in bull_pool.columns else set()

    # 池内代码详情
    lu_info = {}
    if limit_up_pool is not None and not limit_up_pool.empty:
        for _, r in limit_up_pool.iterrows():
            lu_info[str(r.get("code", ""))] = {
                "name": r.get("名称", ""),
                "ret": r.get("今日涨幅", 0),
                "lu_20d": r.get("近20日涨停", 0),
                "industry": r.get("行业", ""),
            }

    yaogu_info = {}
    if yaogu_pool is not None and not yaogu_pool.empty:
        for _, r in yaogu_pool.iterrows():
            yaogu_info[str(r.get("code", ""))] = {
                "name": r.get("name", ""),
                "score": r.get("yaogu_score", 0),
                "streak": r.get("streak", 0),
                "yiziban": r.get("yiziban", False),
            }

    bull_info = {}
    if bull_pool is not None and not bull_pool.empty:
        for _, r in bull_pool.iterrows():
            bull_info[str(r.get("code", ""))] = {
                "name": r.get("名称", ""),
                "score": r.get("牛股评分", 0),
                "vs_ma40": r.get("vsMA40(%)", 0),
            }

    # 所有异动股票
    all_anomaly_codes = set()
    for a in anomalies:
        all_anomaly_codes.update(a["stock_codes"])

    # 获取历史日期用于启动判定
    all_dates = sorted(master_df["date"].unique())
    recent_5_dates = set(all_dates[-5:]) if len(all_dates) >= 5 else set(all_dates)
    earlier_dates = set(all_dates[:-5]) if len(all_dates) > 5 else set()

    results = []
    seen_pairs = set()

    for a in anomalies:
        kw = a["keyword"]
        for s in a["stocks"]:
            code = s["code"]
            pair_key = (kw, code)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            matches = []
            details = {}

            if code in lu_codes:
                matches.append("涨停池")
                details["涨停池"] = lu_info.get(code, {})
            if code in yaogu_codes:
                matches.append("妖股池")
                details["妖股池"] = yaogu_info.get(code, {})
            if code in bull_codes:
                matches.append("牛股池")
                details["牛股池"] = bull_info.get(code, {})

            # 启动阶段判定
            code_dates = set(
                master_df[master_df["code"] == code]["date"].values
            )
            in_recent_5 = bool(code_dates & recent_5_dates)
            in_earlier = bool(code_dates & earlier_dates)

            # 此前20日是否有连板
            earlier_records = master_df[
                (master_df["code"] == code) & (master_df["date"].isin(earlier_dates))
            ]
            had_streak = any(earlier_records["board_count"] >= 2) if not earlier_records.empty else False

            is_startup = in_recent_5 and not had_streak

            results.append({
                "keyword": kw,
                "code": code,
                "name": s["name"],
                "sector": s["sector"],
                "matched_strategies": "|".join(matches) if matches else "无",
                "matched_count": len(matches),
                "is_startup": is_startup,
                "startup_label": "✅ 启动阶段" if is_startup else ("—" if had_streak else "🔄 观察中"),
                "lu_info": details.get("涨停池", {}),
                "yaogu_info": details.get("妖股池", {}),
                "bull_info": details.get("牛股池", {}),
            })

    # 排序：匹配策略数降序 > 启动阶段优先
    results.sort(key=lambda x: (-x["matched_count"], -int(x["is_startup"])))
    logger.info(f"策略关联: {len(results)} 条 (匹配≥1池: {sum(1 for r in results if r['matched_count'] > 0)})")
    return results


# ══════════════════════════════════════════════════════════════════════
# Phase 7b: 个股再启动发现
# ══════════════════════════════════════════════════════════════════════

def find_relaunch_stocks(master_df, limit_up_pool, yaogu_pool, min_gap_days=10):
    """发现'再启动'个股：首板 + 上次出现间隔久 + 在涨停池中。

    规则：
    1. 最新日期的首板股票
    2. 上次涨停距今 ≥ min_gap_days（自然日），休息充分非连板妖股
    3. 在涨停池中（基本面过滤）
    4. 综合评分 = 涨停池加权 + 妖股加分 + 间隔天数加分
    """
    all_dates = sorted(master_df["date"].unique())
    if len(all_dates) < 2:
        return []

    latest_date = all_dates[-1]

    # 最新日期首板
    latest = master_df[(master_df["date"] == latest_date) & (master_df["board_count"] == 1)]

    # 策略池代码集
    lu_codes = set()
    yg_scores = {}
    if limit_up_pool is not None and not limit_up_pool.empty:
        lu_codes = set(str(c) for c in limit_up_pool.get("code", limit_up_pool.get("代码", pd.Series())))
    if yaogu_pool is not None and not yaogu_pool.empty:
        for _, r in yaogu_pool.iterrows():
            yg_scores[str(r.get("code", ""))] = r.get("yaogu_score", 0)

    # Standalone 模式：无策略池时，hotpoint 中的股票本身就是涨停股，全部通过
    has_pools = limit_up_pool is not None and not limit_up_pool.empty

    relaunch = []
    for _, r in latest.iterrows():
        code = r["code"]
        code_dates = sorted(master_df[master_df["code"] == code]["date"].values)

        # 首次出现也纳入（之前没涨停过）
        if len(code_dates) >= 2:
            prev_date = code_dates[-2]
            gap = (pd.Timestamp(latest_date) - pd.Timestamp(prev_date)).days
        else:
            gap = 999  # 首次出现

        if gap < min_gap_days:
            continue

        # 必须在涨停池中（standalone 模式跳过此检查）
        if has_pools and code not in lu_codes:
            continue

        # 评分
        yg_score = yg_scores.get(code, 0)
        total_appearances = len(code_dates)

        # 综合评分: 间隔(35%) + 历史节奏(35%) + 妖股质量(30%)
        # 间隔: 10-45天是甜蜜区（休息充分但未遗忘），>90天可能 OCR 遗漏
        if gap <= 90:
            gap_score = min(gap / 45, 1.0) * 35
        else:
            gap_score = 35 * 0.5  # 首次/极长间隔：不确定性高，给半价

        # 历史节奏: 2-4次最佳（有规律回归），1次或>5次偏低
        if total_appearances == 1:
            rhythm_score = 15  # 首次出现，不确定
        elif 2 <= total_appearances <= 4:
            rhythm_score = 35  # 最佳：有规律地回归
        else:
            rhythm_score = 35 * max(0, (10 - total_appearances) / 10)

        yg_score_norm = min(yg_score / 9, 1.0) * 30
        relaunch_score = round(gap_score + rhythm_score + yg_score_norm, 1)

        # 阶段标签
        if relaunch_score >= 65:
            stage = "🎯强信号"
        elif relaunch_score >= 45:
            stage = "👀关注"
        else:
            stage = "📋观察"

        relaunch.append({
            "code": code,
            "name": r["name"],
            "sector": r["sector"],
            "keywords": r.get("keywords", ""),
            "board_count": r["board_count"],
            "gap_days": gap,
            "total_appearances": total_appearances,
            "prev_date": code_dates[-2] if len(code_dates) >= 2 else "首次",
            "in_limit_up": True,
            "yaogu_score": yg_score,
            "relaunch_score": relaunch_score,
            "stage": stage,
        })

    relaunch.sort(key=lambda x: x["relaunch_score"], reverse=True)
    logger.info(f"个股再启动: {len(relaunch)} 只 (间隔≥{min_gap_days}天 + 涨停池)")
    return relaunch


# ══════════════════════════════════════════════════════════════════════
# Phase 8: 可视化
# ══════════════════════════════════════════════════════════════════════

def _setup_chinese_font():
    """设置中文字体。"""
    # macOS
    for fpath in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]:
        if os.path.exists(fpath):
            font_manager.fontManager.addfont(fpath)
            prop = font_manager.FontProperties(fname=fpath)
            plt.rcParams["font.family"] = prop.get_name()
            return
    # fallback
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def generate_charts(master_df, output_path=None):
    """生成板块 & 关键词频次图（近10日/近5日，细柱多展示）。"""
    _setup_chinese_font()

    all_dates = sorted(master_df["date"].unique())
    dates_10d = all_dates[-10:] if len(all_dates) >= 10 else all_dates
    dates_5d = all_dates[-5:] if len(all_dates) >= 5 else all_dates

    fig, axes = plt.subplots(2, 2, figsize=(20, 22))
    fig.suptitle(f"涨停复盘分析 — {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)}个交易日)",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── 左上：板块 TOP25（近10日）──
    ax = axes[0, 0]
    sf_10 = sector_freq(master_df, dates_10d).head(25)
    colors_10 = ["#2196F3" if i < 5 else "#90CAF9" for i in range(len(sf_10))]
    ax.barh(range(len(sf_10)), sf_10.values, height=0.6, color=colors_10)
    ax.set_yticks(range(len(sf_10)))
    ax.set_yticklabels(sf_10.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"板块频次 TOP25（近{len(dates_10d)}日）", fontsize=13)
    ax.set_xlabel("涨停股票数")
    for i, v in enumerate(sf_10.values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=8)

    # ── 右上：板块 TOP25（近5日）──
    ax = axes[0, 1]
    sf_5 = sector_freq(master_df, dates_5d).head(25)
    colors_5 = ["#FF5722" if i < 5 else "#FFAB91" for i in range(len(sf_5))]
    ax.barh(range(len(sf_5)), sf_5.values, height=0.6, color=colors_5)
    ax.set_yticks(range(len(sf_5)))
    ax.set_yticklabels(sf_5.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"板块频次 TOP25（近{len(dates_5d)}日）", fontsize=13)
    ax.set_xlabel("涨停股票数")
    for i, v in enumerate(sf_5.values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=8)

    # ── 左下：关键词 TOP30（近10日）──
    ax = axes[1, 0]
    kf_10 = keyword_freq(master_df, dates_10d)
    kf_10 = kf_10.head(30) if not kf_10.empty else pd.Series(dtype=int)
    if not kf_10.empty:
        ax.barh(range(len(kf_10)), kf_10.values, height=0.6, color="#4CAF50")
        ax.set_yticks(range(len(kf_10)))
        ax.set_yticklabels(kf_10.index, fontsize=9)
        ax.invert_yaxis()
        for i, v in enumerate(kf_10.values):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
    ax.set_title(f"关键词频次 TOP30（近{len(dates_10d)}日）", fontsize=13)
    ax.set_xlabel("出现次数")

    # ── 右下：关键词 TOP30（近5日）──
    ax = axes[1, 1]
    kf_5 = keyword_freq(master_df, dates_5d)
    kf_5 = kf_5.head(30) if not kf_5.empty else pd.Series(dtype=int)
    if not kf_5.empty:
        ax.barh(range(len(kf_5)), kf_5.values, height=0.6, color="#E91E63")
        ax.set_yticks(range(len(kf_5)))
        ax.set_yticklabels(kf_5.index, fontsize=9)
        ax.invert_yaxis()
        for i, v in enumerate(kf_5.values):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
    ax.set_title(f"关键词频次 TOP30（近{len(dates_5d)}日）", fontsize=13)
    ax.set_xlabel("出现次数")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # 同时保存文件
    if output_path is None:
        output_path = os.path.join(HOTPOINT_DIR_OUT, f"chart_{all_dates[-1]}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(img_b64))
    logger.info(f"图表已保存: {output_path}")

    return img_b64


# ══════════════════════════════════════════════════════════════════════
# Phase 9: HTML 报告
# ══════════════════════════════════════════════════════════════════════

def _get_limit_pool_codes(codes, target_date):
    """批量检查哪些股票满足涨停池 4 条件。返回 {code: True/False}。"""
    if not codes: return {}
    try:
        from data.db import get_engine
        from data.loader import load_daily_data
        from sqlalchemy import text

        engine = get_engine()
        str_codes = [str(c).zfill(6) for c in codes]
        td_str = str(target_date)
        pre = (pd.Timestamp(td_str) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        daily = load_daily_data(engine, str_codes, pre, td_str,
                                cols=["high","low","close","volume"])
        daily["code"] = daily["code"].astype(str).str.zfill(6)
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        daily = daily.sort_values(["code","trade_date"])

        with engine.connect() as conn:
            extra = pd.read_sql(text(
                "SELECT code, market_cap FROM stock_daily_extra WHERE code=ANY(:c) AND trade_date=:d"),
                conn, params={"c": str_codes, "d": td_str})
        extra["code"] = extra["code"].astype(str).str.zfill(6)

        daily["prev_close"] = daily.groupby("code")["close"].shift(1)
        daily["limit_price"] = daily.apply(
            lambda r: round(r["prev_close"] * 1.09899, 4)
            if pd.notna(r["prev_close"]) and r["prev_close"] > 0 else 0, axis=1)
        daily["is_lu"] = (daily["close"] >= daily["limit_price"]).astype(int)
        daily["ma5"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(5,min_periods=3).mean())
        daily["ma10"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(10,min_periods=5).mean())
        daily["lu_20d"] = daily.groupby("code")["is_lu"].transform(lambda x: x.rolling(20,min_periods=1).sum())

        td_mask = daily["trade_date"] == pd.Timestamp(td_str)
        today = daily[td_mask].set_index("code")
        extra_map = dict(zip(extra["code"], extra["market_cap"]))

        result = {}
        for code in str_codes:
            if code not in today.index:
                result[code] = False; continue
            r = today.loc[code]
            mcap = extra_map.get(code, 0)
            result[code] = bool(30 <= mcap <= 500 and 5 <= r["close"] <= 100 and
                               r["ma5"] > r["ma10"] and 2 <= r["lu_20d"] <= 4)

        engine.dispose()
        return result
    except Exception as e:
        logger.warning(f"涨停池批量检查失败 ({len(codes)}只): {e}")
        return {c: False for c in codes}


def _build_recommendation_table(master_df, anomalies, target_date=None):
    """构建 标签烙印型推荐 表格 — 综合标签稳定度 + 涨停池过滤。"""
    from collections import defaultdict

    all_dates = sorted([d for d in master_df["date"].unique() if str(d)[:2] == '20' and len(str(d)) == 8])
    # 支持指定日期（YYYY-MM-DD 或 YYYYMMDD）
    if target_date:
        target_clean = str(target_date).replace('-', '')
        if target_clean in [str(d) for d in all_dates]:
            latest = target_clean
        else:
            latest = str(all_dates[-1])
    else:
        latest = str(all_dates[-1])
    # 统一转字符串避免 pd.Timestamp(int) 解析为纳秒
    latest_str = str(latest)
    all_dates_str = [str(d) for d in all_dates]
    
    # Build keyword history
    kw_hist = defaultdict(list)
    for (code, date), grp in master_df.groupby(['code', 'date']):
        kws = set()
        for kw_str in grp['keywords']:
            if pd.notna(kw_str) and str(kw_str) != 'nan':
                kws.update(str(kw_str).split('|'))
        kw_hist[(code, date)] = kws
    
    def tag_stab(code, today_str):
        tags_by_date = []
        for (c, d), kws in kw_hist.items():
            if c == code and str(d) <= str(today_str):
                tags_by_date.append((str(d), kws))
        tags_by_date.sort()
        if len(tags_by_date) < 2: return 0.0, len(tags_by_date), 0
        latest_kws = tags_by_date[-1][1]
        earlier = set()
        for _, tags in tags_by_date[:-1]:
            earlier.update(tags)
        if not earlier or not latest_kws: return 0.0, len(tags_by_date), 0
        inter = latest_kws & earlier
        union = latest_kws | earlier
        stab = round(len(inter)/len(union), 3) if union else 0.0
        d1, d2 = tags_by_date[-1][0], tags_by_date[-2][0]
        gap = (pd.Timestamp(d1) - pd.Timestamp(d2)).days if len(tags_by_date) >= 2 else 0
        return stab, len(tags_by_date), gap
    
    # Score all stocks on latest date
    latest_df = master_df[master_df["date"] == latest]
    scored = []
    # 扩候选池：近5天内出现过的首板 + 所有历史出现≥2次的
    recent_5d = [d for d in all_dates_str if 0 <= (pd.Timestamp(latest_str) - pd.Timestamp(d)).days <= 5]
    master_df["_date_str"] = master_df["date"].astype(str)
    active_stocks = set(master_df[master_df["_date_str"].isin(recent_5d)]["code"].unique())

    # 批量检查涨停池（转纯 Python str）
    active_list = [str(c).zfill(6) for c in active_stocks]
    logger.info(f"  涨停池检查 {len(active_list)} 只候选...")
    pool_ok = _get_limit_pool_codes(active_list, str(latest))
    in_pool = {c for c, ok in pool_ok.items() if ok}
    logger.info(f"  通过涨停池: {len(in_pool)} 只")

    n_total = n_no_recent = n_not_first = n_low_stab = n_low_appear = n_scored = 0
    # 对每只活跃股票取最新一次出现信息
    for code in active_stocks:
        code_df = master_df[master_df["code"] == code].sort_values("date")
        # 取最近5天内的出现（不是全历史最后一次）
        recent_appearances = code_df[code_df["date"].astype(str).isin(recent_5d)]
        n_total += 1
        if recent_appearances.empty:
            n_no_recent += 1
            continue
        last_row = recent_appearances.iloc[-1]
        if int(last_row["board_count"]) != 1:
            n_not_first += 1
            continue

        stab, n_appear, gap = tag_stab(code, latest)
        if n_appear < 2:
            n_low_appear += 1
            continue
        if stab < 0.3:
            n_low_stab += 1
            continue

        # 涨停池过滤
        if code not in in_pool:
            continue

        kws_short = "|".join(str(last_row.get("keywords", "")).split("|")[:4])

        # 新评分：稳定度(35%) + 历史验证(30%) + 间隔收益(35%)
        # 稳定度按样本量折权：2次出现→打7折，3次→8.5折，4次+→满分
        sample_weight = min(1.0, 0.7 + n_appear * 0.075) if n_appear >= 2 else 0.5
        weighted_stab = stab * sample_weight

        if n_appear >= 5 and stab >= 0.5:
            history_score = 30  # 强验证：多次出现+标签稳定
        elif n_appear >= 3 and stab >= 0.5:
            history_score = 25
        elif n_appear >= 2 and stab >= 0.4:
            history_score = 18  # 正在形成模式
        else:
            history_score = 10

        # 间隔评分：5-15天活跃轮动(0.6) / 15-60天理想(0.6→1.0) / >60天遗忘(0.7)
        if 5 <= gap <= 15:
            gap_bonus = 0.6
        elif 15 < gap <= 60:
            gap_bonus = 0.6 + (gap - 15) / 45 * 0.4  # 0.6 → 1.0
        else:
            gap_bonus = 0.7  # 极短或极长

        score = round(weighted_stab * 35 + history_score + gap_bonus * 35, 1)

        if score >= 55:
            level = "🎯推荐"
        elif score >= 40:
            level = "👀关注"
        else:
            level = "📋观察"

        n_scored += 1
        scored.append({
            "code": code, "name": last_row["name"], "sector": last_row["sector"],
            "keywords": kws_short, "stability": stab, "appearances": n_appear,
            "gap_days": gap, "score": score, "level": level,
            "last_date": last_row["date"],
        })
    
    logger.info(f"  过滤: total={n_total} no_recent={n_no_recent} not_first={n_not_first} "
                f"low_appear={n_low_appear} low_stab={n_low_stab} scored={n_scored}")
    logger.info(f"  评分完成: {len(scored)} 只进入推荐")
    scored.sort(key=lambda x: x["score"], reverse=True)

    if not scored:
        return '<p style="color:#999;text-align:center;padding:20px;">暂无符合条件的标签烙印型股票</p>'
    
    rows = ""
    for i, r in enumerate(scored[:20]):
        sc = r["score"]
        sc_color = "#4CAF50" if sc >= 40 else ("#FF9800" if sc >= 30 else "#999")
        lv_badge = {"🎯推荐": "background:#4CAF50;color:white", "👀关注": "background:#FF9800;color:white", "📋观察": "background:#999;color:white"}.get(r["level"], "")
        rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><b>{r['code']}</b></td>
            <td>{r['name']}</td>
            <td style="font-size:11px;">{r['sector']}</td>
            <td style="font-size:11px;">{r['keywords']}</td>
            <td style="text-align:center;">{r['stability']:.2f}</td>
            <td style="text-align:center;">{r['appearances']}次</td>
            <td style="text-align:center;">{r['gap_days']}天</td>
            <td style="text-align:center;font-weight:bold;font-size:16px;color:{sc_color};">{sc:.0f}</td>
            <td><span style="{lv_badge};padding:2px 8px;border-radius:3px;font-size:11px;">{r['level']}</span></td>
        </tr>"""
    
    rec = sum(1 for r in scored if '推荐' in r['level'])
    watch = sum(1 for r in scored if '关注' in r['level'])
    return f"""
    <p style="color:#888;">🎯推荐 {rec}只 | 👀关注 {watch}只 | 📋观察 {len(scored)-rec-watch}只
    | 评分=标签稳定(40%)+间隔收益(30%)+稀缺性(30%)</p>
    <table>
    <thead><tr>
        <th>#</th><th>代码</th><th>名称</th><th>板块</th><th>关键词</th><th>稳定度</th><th>出现</th><th>间隔</th><th>评分</th><th>推荐</th>
    </tr></thead>
    <tbody>{rows}</tbody>
    </table>"""


def generate_html(anomalies, leaders, strategy_results, relaunch_stocks,
                  chart_b64, master_df, target_date, min_count=1):
    """生成完整 HTML 报告 — 聚焦推荐榜单。"""
    all_dates = sorted(master_df["date"].unique())
    total_records = len(master_df)
    total_dates = len(all_dates)
    total_stocks = master_df["code"].nunique()

    # 近2日
    recent_2 = all_dates[-2:] if len(all_dates) >= 2 else all_dates

    # 集中冒出关键词表
    anomaly_rows = ""
    for i, a in enumerate(anomalies[:25]):
        codes_str = ", ".join(a["stock_codes"][:6])
        if len(a["stock_codes"]) > 6:
            codes_str += f" ... (+{len(a['stock_codes'])-6})"
        names_str = ", ".join(a["stock_names"][:6])
        if len(a["stock_names"]) > 6:
            names_str += " ..."

        stage_icon = a.get('stage', '🆕')
        anomaly_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td style="font-weight:bold;font-size:14px;">{a['keyword']}</td>
            <td style="text-align:center;font-size:16px;font-weight:bold;color:#FF5722;">{a['count_recent']}</td>
            <td style="font-weight:bold;">{a.get('stage', '🆕')}</td>
            <td style="font-size:11px;color:#888;">{a.get('stage_note', '')}</td>
            <td style="font-size:11px;">{codes_str}</td>
            <td style="font-size:11px;">{names_str}</td>
        </tr>"""

    # 推荐表格
    rec_table = _build_recommendation_table(master_df, anomalies, target_date)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>涨停复盘异动分析 — {target_date}</title>
<style>
    body {{ font-family: -apple-system, 'PingFang SC', 'Hiragino Sans GB', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
    h2 {{ border-bottom: 3px solid #E91E63; padding-bottom: 8px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
    .summary-card {{ background: white; border-radius: 8px; padding: 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .summary-card .num {{ font-size: 28px; font-weight: bold; color: #E91E63; }}
    .summary-card .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
    .chart-container {{ text-align: center; margin: 20px 0; background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .chart-container img {{ max-width: 100%; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
    th {{ background: #f5f5f5; padding: 10px 8px; text-align: left; font-size: 12px; color: #666; border-bottom: 2px solid #e0e0e0; }}
    td {{ padding: 8px; border-bottom: 1px solid #f0f0f0; }}
    tr:hover {{ background: #fafafa; }}
    .footer {{ color: #aaa; font-size: 11px; text-align: center; margin-top: 32px; padding-top: 16px; border-top: 1px solid #eee; }}
</style>
</head>
<body>

<h1 style="color:#E91E63;">🔬 涨停复盘 · 标签烙印推荐</h1>
<p style="color:#888;">分析日期: {target_date} | 数据: {all_dates[0]} ~ {all_dates[-1]} | {total_dates}天 · {total_records}条 · {total_stocks}只</p>

<div class="summary">
    <div class="summary-card"><div class="num">{total_stocks}</div><div class="label">涉及股票</div></div>
    <div class="summary-card"><div class="num">{total_dates}</div><div class="label">交易日</div></div>
    <div class="summary-card"><div class="num">{len(anomalies)}</div><div class="label">集中冒出</div></div>
    <div class="summary-card"><div class="num">{total_records}</div><div class="label">涨停记录</div></div>
</div>

<h2>🎯 标签烙印型推荐</h2>
<p style="color:#888;">规则：近5天首板 + 标签跨期稳定(交集/并集) + 历史≥2次 |
    评分 = 稳定度(35%) + 历史验证(30%) + 间隔(35%)</p>
{rec_table}

<div class="chart-container">
    <img src="data:image/png;base64,{chart_b64}" alt="热力图">
</div>

<h2>🆕 集中冒出的关键词</h2>
<p style="color:#888;">前10日零出现（{all_dates[-12] if len(all_dates)>=12 else '—'} ~ {all_dates[-4] if len(all_dates)>=4 else '—'}），
   近2日（{recent_2[0]} ~ {recent_2[-1]}）突然出现 ≥ {min_count}次</p>
<table>
<thead><tr>
    <th>#</th><th>关键词</th><th>近2日</th><th>阶段</th><th>上涨空间</th><th>关联代码</th><th>关联名称</th>
</tr></thead>
<tbody>{anomaly_rows if anomaly_rows else '<tr><td colspan="7" style="text-align:center;color:#999;padding:20px;">✨ 暂无集中冒出的新关键词</td></tr>'}</tbody>
</table>

<div class="footer">
    <p>Quant 项目 · 涨停复盘异动分析 · 自动生成于 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p>数据来源: hotpoint/ 涨停复盘图片 | 仅供参考，不构成投资建议</p>
</div>

</body>
</html>"""
def send_html_email(html_body, subject):
    """发送 HTML 格式邮件，复用项目 SMTP 配置。"""
    try:
        from dotenv import load_dotenv
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

        user = os.getenv("SMTP_USER", "")
        to = os.getenv("SMTP_TO", "")
        if not user or not to:
            logger.warning("邮箱未配置，跳过发送")
            return False

        msg = MIMEMultipart("alternative")
        msg["From"] = os.getenv("SMTP_FROM", user)
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        host = os.getenv("SMTP_HOST", "smtp.qq.com")
        port = int(os.getenv("SMTP_PORT", "465"))
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=15)
        else:
            server = smtplib.SMTP(host, port, timeout=15)
            server.starttls()
        server.login(user, os.getenv("SMTP_PASSWORD", ""))
        server.sendmail(user, [r.strip() for r in to.split(",") if r.strip()], msg.as_string())
        server.quit()
        logger.info(f"HTML 邮件已发送到 {to}")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# Phase 11: 主入口 + 集成接口
# ══════════════════════════════════════════════════════════════════════

def run_hotpoint_analysis(engine=None, target_date=None, min_count=1,
                          skip_ocr=False, use_surya=True,
                          limit_up_pool=None, yaogu_pool=None, bull_pool=None):
    """主分析函数 —— 可被 run_daily_signals.py 调用。

    参数:
        engine: DB 连接
        target_date: 目标日期（默认最新）
        min_count: 异动最小频次阈值
        skip_ocr: 跳过 OCR，仅用缓存
        limit_up_pool: 涨停池 DataFrame（来自 run_daily_signals）
        yaogu_pool: 妖股池 DataFrame
        bull_pool: 牛股池 DataFrame

    返回:
        dict: {
            "anomalies": [...],
            "leaders": [...],
            "strategy_results": [...],
            "chart_b64": "...",
            "html": "...",
            "master_df": DataFrame,
            "target_date": str,
        }
    """
    if engine is None:
        engine = get_engine()

    # ── Step 1: OCR 提取 ──
    logger.info("=" * 60)
    logger.info("Phase 1: OCR 提取")
    ocr_results = extract_all_images(skip_ocr=skip_ocr, use_surya=use_surya)

    if not ocr_results:
        logger.error("无 OCR 数据")
        return None

    # ── Step 2-3: 解析 + 验证 ──
    logger.info("=" * 60)
    logger.info("Phase 2-3: 解析 + DB 验证")
    master_df = load_or_build_master(ocr_results, engine, skip_ocr=False)
    master_df = append_new_dates(master_df, ocr_results, engine)

    if master_df.empty:
        logger.error("主数据为空")
        return None

    # 目标日期
    if target_date is None:
        target_date = sorted(master_df["date"].unique())[-1]
    logger.info(f"目标日期: {target_date}")

    # ── Step 4: 统计分析 ──
    logger.info("=" * 60)
    logger.info("Phase 4: 统计分析")
    all_dates = sorted(master_df["date"].unique())
    dates_10d = all_dates[-10:] if len(all_dates) >= 10 else all_dates
    dates_5d = all_dates[-5:] if len(all_dates) >= 5 else all_dates

    sf_10 = sector_freq(master_df, dates_10d)
    sf_5 = sector_freq(master_df, dates_5d)
    kf_10 = keyword_freq(master_df, dates_10d)
    kf_5 = keyword_freq(master_df, dates_5d)

    logger.info(f"近{len(dates_10d)}日: {sf_10.sum()} 条, "
                f"板块 {len(sf_10)} 个, 关键词 {len(kf_10)} 个")
    logger.info(f"板块 TOP5: {', '.join(f'{k}({v})' for k, v in sf_10.head(5).items())}")
    if not kf_10.empty:
        logger.info(f"关键词 TOP5: {', '.join(f'{k}({v})' for k, v in kf_10.head(5).items())}")

    # ── Step 5: 异动识别 ──
    logger.info("=" * 60)
    logger.info(f"Phase 5: 异动识别 (min_count={min_count})")
    anomalies = detect_anomalies(master_df, min_count=min_count)
    for a in anomalies:
        logger.info(f"  {a['trend']} {a['keyword']}: 近2日={a['count_recent']}, 前10日={a['count_prior']}")

    # ── Step 6: 潜在龙头 ──
    logger.info("=" * 60)
    logger.info("Phase 6: 潜在龙头发现")
    leaders = discover_all_leaders(anomalies, engine, target_date)
    logger.info(f"共发现 {len(leaders)} 只潜在龙头")

    # ── Step 7: 策略关联 ──
    logger.info("=" * 60)
    logger.info("Phase 7: 策略关联")
    strategy_results = cross_ref_strategies(
        anomalies, master_df, limit_up_pool, yaogu_pool, bull_pool,
    )
    startup_count = sum(1 for r in strategy_results if r.get("is_startup"))
    matched_count = sum(1 for r in strategy_results if r.get("matched_count", 0) > 0)
    logger.info(f"  {len(strategy_results)} 条关联, {matched_count} 条策略匹配, {startup_count} 条启动阶段")

    # ── Step 7b: 个股再启动 ──
    logger.info("=" * 60)
    logger.info("Phase 7b: 个股再启动发现")
    relaunch_stocks = find_relaunch_stocks(master_df, limit_up_pool, yaogu_pool)
    logger.info(f"  再启动: {len(relaunch_stocks)} 只, "
                f"强信号 {sum(1 for r in relaunch_stocks if '强信号' in r['stage'])} 只")

    # ── Step 8: 图表 ──
    logger.info("=" * 60)
    logger.info("Phase 8: 生成图表")
    chart_b64 = generate_charts(master_df)

    # ── Step 9: HTML ──
    logger.info("=" * 60)
    logger.info("Phase 9: 生成 HTML 报告")
    html = generate_html(anomalies, leaders, strategy_results, relaunch_stocks,
                         chart_b64, master_df, target_date, min_count)

    report_path = os.path.join(HOTPOINT_DIR_OUT, f"report_{target_date}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML 报告已保存: {report_path}")

    # ── 保存 CSV ──
    _save_csv_outputs(anomalies, leaders, strategy_results, target_date)

    result = {
        "anomalies": anomalies,
        "leaders": leaders,
        "strategy_results": strategy_results,
        "chart_b64": chart_b64,
        "html": html,
        "master_df": master_df,
        "target_date": target_date,
    }
    return result


def _save_csv_outputs(anomalies, leaders, strategy_results, target_date):
    """保存 CSV 输出文件。"""
    os.makedirs(HOTPOINT_DIR_OUT, exist_ok=True)

    # 异动关键词
    if anomalies:
        anom_df = pd.DataFrame([{
            "keyword": a["keyword"],
            "count_recent": a["count_recent"],
            "count_prior": a["count_prior"],
            "trend": a["trend"],
            "stock_codes": "|".join(a["stock_codes"]),
            "stock_names": "|".join(a["stock_names"]),
            "sectors": "|".join(a["sectors"]),
        } for a in anomalies])
        anom_df.to_csv(os.path.join(HOTPOINT_DIR_OUT, f"anomalies_{target_date}.csv"),
                       index=False, encoding="utf-8-sig")

    # 潜在龙头
    if leaders:
        leaders_df = pd.DataFrame(leaders)
        leaders_df.to_csv(os.path.join(HOTPOINT_DIR_OUT, f"leaders_{target_date}.csv"),
                          index=False, encoding="utf-8-sig")

    # 策略关联
    if strategy_results:
        strat_df = pd.DataFrame([{
            "keyword": r["keyword"],
            "code": r["code"],
            "name": r["name"],
            "sector": r["sector"],
            "matched_strategies": r["matched_strategies"],
            "matched_count": r["matched_count"],
            "is_startup": r["is_startup"],
            "startup_label": r["startup_label"],
        } for r in strategy_results])
        strat_df.to_csv(os.path.join(HOTPOINT_DIR_OUT, f"strategy_{target_date}.csv"),
                        index=False, encoding="utf-8-sig")

    logger.info("CSV 输出已保存")


def generate_text_summary(result, limit_up_count=0, yaogu_count=0, bull_count=0):
    """生成纯文本摘要（用于嵌入 run_daily_signals 报告）。"""
    if result is None:
        return "【涨停复盘异动分析】\n暂无数据\n"

    anomalies = result.get("anomalies", [])
    leaders = result.get("leaders", [])
    strategy_results = result.get("strategy_results", [])
    master_df = result.get("master_df")

    lines = [
        "=" * 66,
        "  🔥 涨停复盘 · 异动分析",
        "=" * 66,
    ]

    if master_df is not None:
        all_dates = sorted(master_df["date"].unique())
        lines.append(f"数据范围: {all_dates[0]} ~ {all_dates[-1]} "
                    f"({len(all_dates)}个交易日, {len(master_df)}条记录)")
        lines.append("")

    # 异动关键词
    if anomalies:
        lines.append(f"【集中冒出的关键词】({len(anomalies)}个，前10日无→近2日出现)")
        lines.append(f"{'关键词':<16} {'近2日':>6} {'前10日':>6} {'趋势':<10} {'关联股数':>8}")
        lines.append("-" * 56)
        for a in anomalies[:20]:
            lines.append(
                f"  {a['keyword']:<12} {a['count_recent']:>6} {a['count_prior']:>6} "
                f"{a['trend']:<8} {len(a['stock_codes']):>8}"
            )
        lines.append("")

    # 潜在龙头
    if leaders:
        lines.append(f"【潜在龙头】({min(len(leaders), 20)}只)")
        lines.append(f"{'代码':<12} {'名称':<8} {'关键词':<14} {'5日涨停':>8} {'连板':>4}")
        lines.append("-" * 56)
        for l in leaders[:20]:
            lines.append(f"  {l['code']:<10} {l['name']:<8} {l['keyword']:<12} "
                        f"{l['lu_count_5d']:>8} {l.get('streak', '-'):>4}")
        lines.append("")

    # 策略关联
    if strategy_results:
        startup_count = sum(1 for r in strategy_results if r.get("is_startup"))
        matched = [r for r in strategy_results if r["matched_count"] > 0]
        lines.append(f"【策略关联】匹配: {len(matched)}只, 启动阶段: {startup_count}只")
        if matched:
            lines.append(f"{'关键词':<14} {'代码':<10} {'名称':<8} {'匹配策略':<20} {'启动':<8}")
            lines.append("-" * 66)
            for r in matched[:20]:
                lines.append(
                    f"  {r['keyword']:<12} {r['code']:<10} {r['name']:<8} "
                    f"{r['matched_strategies']:<20} {r['startup_label']:<8}"
                )
        lines.append("")

    lines.append("(完整图表和龙头列表见 HTML 报告)")

    return "\n".join(lines)


# ── CLI ──
def main():
    parser = argparse.ArgumentParser(description="涨停复盘 OCR 分析与异动识别")
    parser.add_argument("--no-ocr", action="store_true", help="跳过 OCR，使用缓存")
    parser.add_argument("--min-count", type=int, default=1, help="异动最小频次阈值（近2日出现次数）")
    parser.add_argument("--tesseract", action="store_true", help="使用 Tesseract OCR（默认用 Surya）")
    parser.add_argument("--html-only", action="store_true", help="仅从已有数据生成 HTML")
    parser.add_argument("--send-email", action="store_true", help="发送 HTML 邮件")
    parser.add_argument("--date", help="目标日期 YYYY-MM-DD")
    args = parser.parse_args()

    engine = get_engine()

    if args.html_only:
        # 仅生成 HTML
        if not os.path.exists(MASTER_CSV):
            logger.error(f"主数据不存在: {MASTER_CSV}，请先运行完整分析")
            sys.exit(1)
        master_df = pd.read_csv(MASTER_CSV, dtype={"code": str})
        target_date = args.date or sorted(master_df["date"].unique())[-1]
        anomalies = detect_anomalies(master_df, min_count=args.min_count)
        leaders = discover_all_leaders(anomalies, engine, target_date)
        strategy_results = cross_ref_strategies(anomalies, master_df, None, None, None)
        chart_b64 = generate_charts(master_df)
        html = generate_html(anomalies, leaders, strategy_results, [], chart_b64,
                             master_df, target_date, args.min_count)
        report_path = os.path.join(HOTPOINT_DIR_OUT, f"report_{target_date}.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML 报告已保存: {report_path}")
        _save_csv_outputs(anomalies, leaders, strategy_results, target_date)
    else:
        result = run_hotpoint_analysis(
            engine=engine, target_date=args.date,
            min_count=args.min_count, skip_ocr=args.no_ocr,
            use_surya=not args.tesseract,
        )
        if result is None:
            logger.error("分析失败")
            sys.exit(1)

    # 发送邮件
    if args.send_email:
        if args.html_only:
            report_path = os.path.join(HOTPOINT_DIR_OUT, f"report_{args.date or 'latest'}.html")
            # 找实际文件
            import glob
            candidates = sorted(glob.glob(os.path.join(HOTPOINT_DIR_OUT, "report_*.html")))
            if candidates:
                with open(candidates[-1], "r", encoding="utf-8") as f:
                    html = f.read()
                target_date = candidates[-1].split("_")[-1].replace(".html", "")
            else:
                logger.error("未找到 HTML 报告")
                sys.exit(1)
        else:
            html = result["html"]
            target_date = result["target_date"]

        send_html_email(html, f"涨停复盘异动分析 {target_date}")
        print(f"邮件已发送: 涨停复盘异动分析 {target_date}")

    engine.dispose()
    logger.info("完成")


if __name__ == "__main__":
    main()
