"""Tests for factors/viz.py — evolution visualization."""

import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factors.viz import (
    plot_evolution_dashboard,
    plot_round_detail,
    plot_terminal_summary,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_db():
    """10 轮进化的样本数据库."""
    np.random.seed(42)
    history = []
    for r in range(1, 11):
        history.append({
            "round": r,
            "n_individuals": 50,
            "n_valid": np.random.randint(30, 45),
            "best_ic": round(np.random.uniform(0.02, 0.08), 4),
            "best_fitness": round(np.random.uniform(-0.05, 0.35), 4),
            "bt_annual": round(np.random.uniform(-0.1, 0.45), 4),
            "bt_mdd": round(np.random.uniform(0.05, 0.30), 4),
            "bt_sharpe": round(np.random.uniform(-0.5, 1.5), 2),
            "top_factors": [
                {"name": f"f_{r}_{i}", "ic": round(np.random.uniform(0.01, 0.07), 4)}
                for i in range(5)
            ],
        })
    return {"rounds": 10, "history": history}


@pytest.fixture
def empty_db():
    """空数据库."""
    return {"rounds": 0, "history": []}


@pytest.fixture
def minimal_report():
    """最简报告字典，包含所有面板数据."""
    np.random.seed(99)
    return {
        "round": 5,
        "top_factors": [
            {"name": "alpha_01", "depth": 3, "ic": 0.052},
            {"name": "alpha_02", "depth": 5, "ic": 0.038},
            {"name": "alpha_03", "depth": 2, "ic": 0.061},
            {"name": "alpha_04", "depth": 4, "ic": 0.029},
            {"name": "alpha_05", "depth": 6, "ic": 0.044},
        ],
        "industry_ic": {
            "科技": 0.045,
            "消费": 0.032,
            "医药": -0.018,
            "金融": 0.055,
            "周期": -0.011,
        },
        "operator_usage": {
            "ts_rank": 45,
            "ts_mean": 30,
            "ts_std": 18,
            "ts_zscore": 22,
            "correlation": 12,
            "linear_decay": 8,
        },
        "high_corr_count": 7,
    }


@pytest.fixture
def empty_report():
    """完全为空的报告."""
    return {}


# ═══════════════════════════════════════════════════════════════════════
# Tests: Dashboard
# ═══════════════════════════════════════════════════════════════════════

def test_dashboard_saves_png(sample_db):
    """仪表盘使用 sample_db 保存 PNG 且文件 > 1000 字节."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_evolution_dashboard(sample_db, save_path=path)
        assert os.path.exists(path), "Expected PNG file to exist"
        size = os.path.getsize(path)
        assert size > 1000, f"Expected PNG > 1000 bytes, got {size}"
    finally:
        os.unlink(path)


def test_dashboard_handles_empty(empty_db):
    """空数据库保存一个合法但小的 PNG（不崩溃）."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_evolution_dashboard(empty_db, save_path=path)
        assert os.path.exists(path), "Expected PNG to exist for empty db"
        size = os.path.getsize(path)
        assert size > 100, f"Expected a valid PNG file, got {size} bytes"
    finally:
        os.unlink(path)


def test_dashboard_handles_missing_keys():
    """history 条目缺少键时不崩溃."""
    db = {
        "rounds": 3,
        "history": [
            {"round": 1, "n_valid": 30},
            {"round": 2, "best_ic": 0.03},
            {"round": 3},
        ],
    }
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_evolution_dashboard(db, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 500
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════
# Tests: Terminal Summary
# ═══════════════════════════════════════════════════════════════════════

def test_terminal_summary_runs(sample_db, capsys):
    """终端摘要使用 sample_db 不崩溃."""
    plot_terminal_summary(sample_db)
    captured = capsys.readouterr()
    output = captured.out.strip()
    assert "轮" in output
    assert "IC=" in output
    assert "适应度=" in output
    assert "年化=" in output
    assert "MDD=" in output


def test_terminal_summary_empty(empty_db, capsys):
    """空数据库终端摘要不崩溃."""
    plot_terminal_summary(empty_db)
    captured = capsys.readouterr()
    output = captured.out.strip()
    assert "No evolution rounds completed yet." in output


def test_terminal_summary_nan_values(capsys):
    """含 NaN 值的条目不崩溃."""
    db = {
        "rounds": 2,
        "history": [
            {
                "round": 1,
                "best_ic": float("nan"),
                "best_fitness": float("nan"),
                "bt_annual": float("nan"),
                "bt_mdd": float("nan"),
            },
            {
                "round": 2,
                "best_ic": 0.05,
                "best_fitness": 0.25,
                "bt_annual": 0.35,
                "bt_mdd": 0.12,
            },
        ],
    }
    plot_terminal_summary(db)
    captured = capsys.readouterr()
    output = captured.out.strip()
    assert "轮" in output


# ═══════════════════════════════════════════════════════════════════════
# Tests: Round Detail
# ═══════════════════════════════════════════════════════════════════════

def test_round_detail_from_report(minimal_report):
    """完整报告字典生成有效 PNG."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_round_detail(minimal_report, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000
    finally:
        os.unlink(path)


def test_round_detail_empty_report(empty_report):
    """完全为空的报告不崩溃."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_round_detail(empty_report, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100
    finally:
        os.unlink(path)


def test_round_detail_partial_report():
    """部分填充的报告不崩溃."""
    report = {
        "top_factors": [{"name": "f1", "depth": 2, "ic": 0.04}],
        # 缺少 industry_ic、operator_usage、high_corr_count
    }
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_round_detail(report, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 500
    finally:
        os.unlink(path)


def test_dashboard_nan_in_metrics():
    """指标含 NaN 的 history 条目正常渲染."""
    db = {
        "rounds": 5,
        "history": [
            {"round": 1, "best_fitness": 0.1, "best_ic": 0.03, "bt_annual": 0.15, "bt_mdd": 0.10, "n_valid": 35},
            {"round": 2, "best_fitness": float("nan"), "best_ic": 0.04, "bt_annual": float("nan"), "bt_mdd": 0.12, "n_valid": 40},
            {"round": 3, "best_fitness": 0.25, "best_ic": float("nan"), "bt_annual": 0.30, "bt_mdd": float("nan"), "n_valid": 38},
            {"round": 4, "best_fitness": 0.15, "best_ic": 0.05, "bt_annual": 0.20, "bt_mdd": 0.08, "n_valid": 42},
            {"round": 5, "best_fitness": 0.30, "best_ic": 0.06, "bt_annual": 0.40, "bt_mdd": 0.11, "n_valid": 44},
        ],
    }
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_evolution_dashboard(db, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000
    finally:
        os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
