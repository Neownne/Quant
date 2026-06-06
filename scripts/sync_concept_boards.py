#!/usr/bin/env python
"""概念板块数据同步：拉取东方财富概念板块 + 成分股。

用法:
    python scripts/sync_concept_boards.py            # 全量同步（首次）
    python scripts/sync_concept_boards.py --update   # 增量更新
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from loguru import logger
from sqlalchemy import text
from data.db import get_engine

API_BOARDS = "https://push2.eastmoney.com/api/qt/clist/get"
API_CONS = "https://push2.eastmoney.com/api/qt/clist/get"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
BOARD_DELAY = 1.0  # seconds between board API calls
RETRY_DELAY = 5   # seconds on rate limit (快速失败, 稍后重试)


def _api_get(url, params, timeout=15):
    """带重试的 API 调用。"""
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            data = r.json()
            if data.get("data") and data["data"].get("diff"):
                return data
            logger.warning(f"API 返回空, retry {attempt+1}")
        except Exception as e:
            logger.warning(f"API 失败 ({attempt+1}/5): {e}")
        time.sleep(min(RETRY_DELAY * (attempt + 1), 300))
    return None


def sync_boards(engine):
    """同步概念板块列表（分页拉全量 486 个）。"""
    all_boards = []
    for pn in range(1, 10):  # 最多 10 页 (~486 板块)
        params = {"fid": "f3", "po": "1", "pz": "100", "pn": str(pn), "np": "1",
                  "fltt": "2", "invt": "2", "fs": "m:90+t:3", "fields": "f12,f14"}
        data = _api_get(API_BOARDS, params)
        if not data:
            break
        boards = data["data"]["diff"]
        if not boards:
            break
        all_boards.extend(boards)
        if len(boards) < 100:  # 最后一页，不足 100 条
            break
        time.sleep(1)

    logger.info(f"获取到 {len(all_boards)} 个概念板块 (共 {pn} 页)")

    with engine.begin() as c:
        for b in all_boards:
            c.execute(text("""
                INSERT INTO concept_board (code, name, stock_count, updated_date)
                VALUES (:c, :n, 0, CURRENT_DATE)
                ON CONFLICT (code) DO UPDATE SET name=:n2, updated_date=CURRENT_DATE
            """), {"c": b["f12"], "n": b["f14"], "n2": b["f14"]})

    return [(b["f12"], b["f14"]) for b in all_boards]


def sync_constituents(engine, board_code, board_name):
    """同步单个板块的成分股。"""
    params = {"fid": "f3", "po": "1", "pz": "200", "pn": "1", "np": "1",
              "fltt": "2", "invt": "2",
              "fs": f"b:{board_code}+f:!50", "fields": "f12"}
    data = _api_get(API_CONS, params)
    if not data:
        return 0

    items = data["data"]["diff"]
    count = len(items)
    with engine.begin() as c:
        # 删旧数据
        c.execute(text("DELETE FROM concept_stock WHERE board_code=:b"), {"b": board_code})
        for item in items:
            c.execute(text("""
                INSERT INTO concept_stock (board_code, stock_code)
                VALUES (:b, :s)
                ON CONFLICT DO NOTHING
            """), {"b": board_code, "s": item["f12"]})
        c.execute(text("UPDATE concept_board SET stock_count=:n, updated_date=CURRENT_DATE WHERE code=:c"),
                  {"n": count, "c": board_code})

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="只更新板块列表，跳过成分股")
    parser.add_argument("--board", default="", help="只同步指定板块代码")
    parser.add_argument("--fill", action="store_true", help="只补全 stock_count=0 的板块成分股")
    args = parser.parse_args()

    engine = get_engine()

    # Step 1: 板块列表
    boards = sync_boards(engine)
    if not boards:
        return

    if args.board:
        boards = [(args.board, "manual")]

    if args.fill:
        # 只取缺成分股的板块
        with engine.connect() as c:
            missing = c.execute(text(
                "SELECT code, name FROM concept_board WHERE stock_count=0"
            )).fetchall()
        boards = [(r[0], r[1]) for r in missing]
        logger.info(f"--fill 模式: {len(boards)} 个板块待补全")

    if args.update:
        logger.info("--update 模式，跳过成分股")
        return

    # Step 2: 成分股
    total = 0
    for i, (code, name) in enumerate(boards):
        if i > 0:
            time.sleep(BOARD_DELAY)
        logger.info(f"[{i+1}/{len(boards)}] {name}({code}) ...")
        n = sync_constituents(engine, code, name)
        total += n
        if n > 0:
            logger.info(f"  → {n} 只成分股")

    logger.info(f"完成: {len(boards)} 板块, {total} 条成分股映射")
    engine.dispose()


if __name__ == "__main__":
    main()
