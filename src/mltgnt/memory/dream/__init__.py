"""mltgnt.memory.dream — dream.json 合成 API。"""
from mltgnt.memory.dream._format import DreamSection, DreamSummary
from mltgnt.memory.dream.api import read_dream, write_dream
from mltgnt.memory.dream.selector import DreamSelector
from mltgnt.memory.dream.synthesizer import Synthesizer

__all__ = [
    "DreamSection",
    "DreamSummary",
    "read_dream",
    "write_dream",
    "DreamSelector",
    "Synthesizer",
]
