"""tests/test_fanout_integration.py — integration tests for the fanout path (#1036)"""
from mltgnt.scheduler import _FANOUT_PROMPT_SUFFIX


class TestFanoutPromptSuffix:
    def test_suffix_contains_ghdag_fanout_key(self):
        """_FANOUT_PROMPT_SUFFIX contains ghdag_fanout."""
        assert "ghdag_fanout" in _FANOUT_PROMPT_SUFFIX
