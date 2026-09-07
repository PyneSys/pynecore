"""
@pyne

Library for ``test_097``: two exports, one closing over a ``Persistent`` of the
library's own scope and one carrying a per-call-site ``ta`` machine. The main
body calls an export itself, so the run also pins that a later bar still
resolves the name after the definition stopped being rebuilt.
"""
from typing import Any, Protocol

from pynecore.core.pine_export import Exported, export
from pynecore.lib import close, script, ta
from pynecore.types import Persistent


class _Scaled(Protocol):
    def __call__(self, x: float) -> Any: ...


class _Smoothed(Protocol):
    def __call__(self, src: float, length: int) -> Any: ...


scaled: _Scaled = Exported()
smoothed: _Smoothed = Exported()

__all__ = ['scaled', 'smoothed']


@script.library("Export Once Lib")
def main():
    factor: Persistent[float] = 3.0

    @export
    def scaled(x: float):
        return x * factor

    @export
    def smoothed(src: float, length: int):
        return ta.sma(src, length)

    runs: Persistent[int] = 0
    runs += 1
    return {"own_scaled": scaled(2.0), "own_smoothed": smoothed(close, 3), "runs": runs}
