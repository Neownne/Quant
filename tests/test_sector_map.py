"""板块分类映射测试。"""
import pytest
from config.sector_map import (
    build_sector_map,
    classify_stock,
    SECTOR_LABELS,
    BROAD_CLASSIFICATION,
    SW1_CLASSIFICATION,
)


class TestClassifyStock:
    """单只股票板块分类测试。"""

    def test_kechuang_stocks_identified_by_688_prefix(self):
        """688开头的股票应归入科创板块。"""
        assert classify_stock("688001.SH", csi300_members=set(), dividend_stocks=set()) == "科创"
        assert classify_stock("688981.SH", csi300_members=set(), dividend_stocks=set()) == "科创"

    def test_beizheng_stocks_identified_by_8_or_4_prefix(self):
        """8或4开头的股票应归入北证板块（非688、非红利）。"""
        assert classify_stock("830799.BJ", csi300_members=set(), dividend_stocks=set()) == "北证"
        assert classify_stock("430047.BJ", csi300_members=set(), dividend_stocks=set()) == "北证"

    def test_dividend_stocks_take_priority_over_size_classification(self):
        """红利股优先：即使代码是主板，如果在红利名单中也归入红利。"""
        dividend = {"600036.SH", "000001.SZ"}
        assert classify_stock("600036.SH", csi300_members=set(), dividend_stocks=dividend) == "红利"

    def test_mainboard_large_stocks_in_csi300(self):
        """非科创/北证/红利 + 在沪深300中 → 主板大盘。"""
        csi300 = {"600519.SH", "000858.SZ"}
        assert classify_stock("600519.SH", csi300_members=csi300, dividend_stocks=set()) == "主板大盘"

    def test_mainboard_small_stocks_not_in_csi300(self):
        """非科创/北证/红利 + 不在沪深300中 → 主板小盘。"""
        assert classify_stock("603123.SH", csi300_members=set(), dividend_stocks=set()) == "主板小盘"
        assert classify_stock("002230.SZ", csi300_members=set(), dividend_stocks=set()) == "主板小盘"

    def test_kechuang_overrides_everything_else(self):
        """科创板优先：即使同时在沪深300和红利名单中，688开头仍归入科创。"""
        csi300 = {"688001.SH"}
        dividend = {"688001.SH"}
        assert classify_stock("688001.SH", csi300_members=csi300, dividend_stocks=dividend) == "科创"

    def test_beizheng_overrides_csi300_and_dividend(self):
        """北交所优先级仅次于科创：8/4开头即使在其他名单中仍归入北证。"""
        csi300 = {"830799.BJ"}
        dividend = {"830799.BJ"}
        assert classify_stock("830799.BJ", csi300_members=csi300, dividend_stocks=dividend) == "北证"

    def test_returns_valid_sector_label_for_any_code(self):
        """任何合法代码都应返回5个板块标签之一。"""
        valid = set(SECTOR_LABELS)
        codes = [
            "688001.SH", "830799.BJ", "430047.BJ",
            "600519.SH", "000858.SZ", "603123.SH",
            "002230.SZ", "300750.SZ", "601398.SH",
        ]
        for code in codes:
            result = classify_stock(code, csi300_members=set(), dividend_stocks=set())
            assert result in valid, f"{code} returned {result}, expected one of {valid}"


class TestBuildSectorMap:
    """批量构建板块映射测试。"""

    def test_returns_dict_mapping_code_to_sector(self):
        """应返回 code → sector_label 的字典。"""
        codes = ["688001.SH", "600519.SH", "603123.SH", "830799.BJ", "430047.BJ"]
        csi300 = {"600519.SH"}
        dividend = set()
        result = build_sector_map(codes, csi300_members=csi300, dividend_stocks=dividend)
        assert isinstance(result, dict)
        assert len(result) == 5
        assert result["688001.SH"] == "科创"
        assert result["600519.SH"] == "主板大盘"
        assert result["603123.SH"] == "主板小盘"

    def test_handles_empty_input(self):
        """空输入返回空字典。"""
        result = build_sector_map([], csi300_members=set(), dividend_stocks=set())
        assert result == {}

    def test_dividend_classification_applied_correctly(self):
        """红利股在批量映射中被正确分类。"""
        codes = ["600036.SH", "000001.SZ", "688001.SH", "603123.SH"]
        dividend = {"600036.SH", "000001.SZ"}
        result = build_sector_map(codes, csi300_members=set(), dividend_stocks=dividend)
        assert result["600036.SH"] == "红利"
        assert result["000001.SZ"] == "红利"
        assert result["688001.SH"] == "科创"  # 科创优先
        assert result["603123.SH"] == "主板小盘"


class TestSectorLabels:
    """板块标签定义测试。"""

    def test_five_sectors_defined(self):
        """应有且仅有5个板块标签。"""
        assert len(SECTOR_LABELS) == 5
        assert "科创" in SECTOR_LABELS
        assert "北证" in SECTOR_LABELS
        assert "红利" in SECTOR_LABELS
        assert "主板大盘" in SECTOR_LABELS
        assert "主板小盘" in SECTOR_LABELS

    def test_broad_and_sw1_classifications_available(self):
        """两种分类方式都应可用。"""
        assert BROAD_CLASSIFICATION == "broad"
        assert SW1_CLASSIFICATION == "sw1"
