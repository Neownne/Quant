"""板块过滤测试。"""
import pytest
import pandas as pd
from portfolio.sector_filter import filter_by_top_sectors


class TestFilterByTopSectors:
    """filter_by_top_sectors 测试。"""

    def test_keeps_only_stocks_from_top_n_sectors(self):
        """应只保留前N个板块的股票。"""
        stock_preds = pd.DataFrame({
            "code": ["688001.SH", "688002.SH", "600001.SH", "600002.SH",
                     "000001.SZ", "000002.SZ"],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "rank": [1, 2, 3, 4, 5, 6],
        })
        sector_scores = pd.DataFrame({
            "sector": ["科创", "主板大盘", "红利"],
            "score": [0.9, 0.6, 0.3],
            "rank": [1, 2, 3],
        })
        code_to_sector = {
            "688001.SH": "科创", "688002.SH": "科创",
            "600001.SH": "主板大盘", "600002.SH": "主板大盘",
            "000001.SZ": "红利", "000002.SZ": "红利",
        }

        # 只取前2个板块（科创 + 主板大盘）
        result = filter_by_top_sectors(stock_preds, sector_scores, code_to_sector, top_n_sectors=2)

        assert len(result) == 4  # 科创2只 + 主板大盘2只
        assert set(result["code"]) == {"688001.SH", "688002.SH", "600001.SH", "600002.SH"}

    def test_top_1_sector_keeps_only_that_sector(self):
        """top_n_sectors=1 时只保留得分最高的板块。"""
        stock_preds = pd.DataFrame({
            "code": ["688001.SH", "600001.SH", "000001.SZ"],
            "score": [0.9, 0.7, 0.5],
            "rank": [1, 2, 3],
        })
        sector_scores = pd.DataFrame({
            "sector": ["科创", "主板大盘", "红利"],
            "score": [0.9, 0.6, 0.3],
            "rank": [1, 2, 3],
        })
        code_to_sector = {
            "688001.SH": "科创", "600001.SH": "主板大盘", "000001.SZ": "红利",
        }

        result = filter_by_top_sectors(stock_preds, sector_scores, code_to_sector, top_n_sectors=1)
        assert len(result) == 1
        assert result.iloc[0]["code"] == "688001.SH"

    def test_preserves_score_and_rank_columns(self):
        """过滤后应保留原有的 score 和 rank 列。"""
        stock_preds = pd.DataFrame({
            "code": ["688001.SH", "600001.SH"],
            "score": [0.9, 0.7],
            "rank": [1, 2],
        })
        sector_scores = pd.DataFrame({
            "sector": ["科创", "主板大盘"],
            "score": [0.9, 0.6],
            "rank": [1, 2],
        })
        code_to_sector = {"688001.SH": "科创", "600001.SH": "主板大盘"}

        result = filter_by_top_sectors(stock_preds, sector_scores, code_to_sector, top_n_sectors=2)
        assert "score" in result.columns
        assert "rank" in result.columns

    def test_stocks_with_unknown_sector_are_excluded(self):
        """未知板块的股票应被过滤掉。"""
        stock_preds = pd.DataFrame({
            "code": ["688001.SH", "999999.XZ"],
            "score": [0.9, 0.8],
            "rank": [1, 2],
        })
        sector_scores = pd.DataFrame({
            "sector": ["科创"],
            "score": [0.9],
            "rank": [1],
        })
        code_to_sector = {"688001.SH": "科创"}  # 999999.XZ 不在映射中

        result = filter_by_top_sectors(stock_preds, sector_scores, code_to_sector, top_n_sectors=1)
        assert len(result) == 1
        assert result.iloc[0]["code"] == "688001.SH"

    def test_empty_sector_scores_returns_all_stocks(self):
        """板块打分为空时返回所有股票（不过滤）。"""
        stock_preds = pd.DataFrame({
            "code": ["688001.SH", "600001.SH"],
            "score": [0.9, 0.7],
            "rank": [1, 2],
        })
        sector_scores = pd.DataFrame()
        code_to_sector = {"688001.SH": "科创", "600001.SH": "主板大盘"}

        result = filter_by_top_sectors(stock_preds, sector_scores, code_to_sector, top_n_sectors=2)
        assert len(result) == 2

    def test_top_n_exceeds_available_sectors(self):
        """top_n 超过可用板块数时取所有板块。"""
        stock_preds = pd.DataFrame({
            "code": ["688001.SH", "600001.SH", "000001.SZ"],
            "score": [0.9, 0.7, 0.5],
            "rank": [1, 2, 3],
        })
        sector_scores = pd.DataFrame({
            "sector": ["科创", "主板大盘"],
            "score": [0.9, 0.6],
            "rank": [1, 2],
        })
        code_to_sector = {
            "688001.SH": "科创", "600001.SH": "主板大盘", "000001.SZ": "红利",
        }

        result = filter_by_top_sectors(stock_preds, sector_scores, code_to_sector, top_n_sectors=10)
        # 只有2个板块在打分中，红利板块不在打分中所以被排除
        assert len(result) == 2
        assert result["code"].tolist() == ["688001.SH", "600001.SH"]
