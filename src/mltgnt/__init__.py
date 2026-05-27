"""mltgnt — persona × secretary の型契約とチャット入出力（OSS 向けコア）。"""

from importlib.metadata import PackageNotFoundError, version

from mltgnt.agent import AgentResult, AgentRunner
from mltgnt.bridges.ghdag_bridge import enqueue_and_wait, enqueue_dag
from mltgnt.chat.pipeline import run_pipeline
from mltgnt.interfaces.persona import PersonaProtocol
from mltgnt.interfaces.types import ChatInput, ChatOutput, Message
from mltgnt.memory import (
    compact,
    needs_compaction,
    read_memory_by_relevance,
    read_memory_iterative,
    read_memory_with_sufficiency_check,
)
from mltgnt.persona import (
    Persona,
    list_personas,
    load_persona,
    run_persona_prompt,
    validate_persona,
)
from mltgnt.scheduler import PersonaScheduler, ScheduleJob

try:
    __version__ = version("mltgnt")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    # chat
    "run_pipeline",
    # memory
    "read_memory_iterative",
    "read_memory_by_relevance",
    "read_memory_with_sufficiency_check",
    "compact",
    "needs_compaction",
    # persona
    "Persona",
    "load_persona",
    "list_personas",
    "validate_persona",
    "run_persona_prompt",
    # interfaces (types)
    "ChatInput",
    "ChatOutput",
    "Message",
    "PersonaProtocol",
    # agent
    "AgentResult",
    "AgentRunner",
    # bridges
    "enqueue_dag",
    "enqueue_and_wait",
    # scheduler
    "PersonaScheduler",
    "ScheduleJob",
    # version
    "__version__",
]
