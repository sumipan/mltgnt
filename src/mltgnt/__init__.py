"""mltgnt — persona × secretary の型契約とチャット入出力（OSS 向けコア）。"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mltgnt")
except PackageNotFoundError:
    __version__ = "0.0.0"
