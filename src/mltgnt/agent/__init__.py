"""mltgnt.agent — 汎用エージェントループ。

設計: Issue #287
"""
from mltgnt.agent._runner import AgentResult, AgentRunner
from mltgnt.bridges.hooks_adapter import create_audit_writer

__all__ = ["AgentResult", "AgentRunner", "create_audit_writer"]
