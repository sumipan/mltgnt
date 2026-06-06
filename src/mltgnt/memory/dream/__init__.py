"""mltgnt.memory.dream — dream.json 合成 API。"""
from mltgnt.memory.dream._format import DreamSection, DreamSummary
from mltgnt.memory.dream.api import read_dream, read_global, read_global_summary, write_dream, write_global
from mltgnt.memory.dream.selector import DreamSelector
from mltgnt.memory.dream.synthesizer import Synthesizer

__all__ = [
    "DreamSection",
    "DreamSummary",
    "read_dream",
    "write_dream",
    "read_global",
    "write_global",
    "read_global_summary",
    "DreamSelector",
    "Synthesizer",
]
