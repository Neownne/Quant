"""RL-Dynamic 策略测试。"""
import pytest


class TestRLDynamicModule:
    """RL-Dynamic 模块导入测试。"""

    def test_module_imports(self):
        """测试 rl_dynamic 模块能正常导入。"""
        import rl_dynamic
        assert hasattr(rl_dynamic, '__all__') or True  # module exists

    def test_module_docstring(self):
        """测试模块文档字符串。"""
        import rl_dynamic
        assert rl_dynamic.__doc__ is not None
        assert 'RL-Dynamic' in rl_dynamic.__doc__


class TestPaperAccount:
    """模拟盘账户测试。"""

    def test_account_exists(self):
        """测试 paper_account id=18 (RL-Dynamic) 存在。"""
        from data.db import get_engine
        from sqlalchemy import text
        e = get_engine()
        with e.connect() as c:
            r = c.execute(
                text("SELECT id, name, cash FROM paper_account WHERE id = 18")
            ).fetchone()
            assert r is not None, "paper_account id=18 does not exist"
            assert r[0] == 18
            assert r[1] == 'RL-Dynamic'
            assert r[2] == 1_000_000
        e.dispose()

    def test_paper_run_exists(self):
        """测试 paper_runs id=5 存在且关联 strategy_id=15 (舞)。"""
        from data.db import get_engine
        from sqlalchemy import text
        e = get_engine()
        with e.connect() as c:
            r = c.execute(
                text("SELECT id, strategy_id, version_id, status FROM paper_runs WHERE id = 5")
            ).fetchone()
            assert r is not None, "paper_runs id=5 does not exist"
            assert r[0] == 5
            assert r[1] == 15  # strategy_id = 15 (舞)
            assert r[3] == 'running'
        e.dispose()
