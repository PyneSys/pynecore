"""
@pyne

Library for ``test_181``: a two-state library with a stateful export, and an
export that calls it on every pass of a loop.
"""
from pynecore.core.pine_export import Exported, export
from pynecore.lib import script, ta
from typing import Any, Protocol


class _Band(Protocol):
    def __call__(self, source: float, length: int, mult: float) -> Any: ...


class _Bands(Protocol):
    def __call__(self, source: float, length: int) -> Any: ...


band: _Band = Exported()
bands: _Bands = Exported()

__all__ = ['band', 'bands']


@script.library("Bool mode loop lib")
def main():
    @export
    def band(source: float, length: int, mult: float):
        return ta.sma(source, length) + ta.stdev(source, length) * mult

    @export
    def bands(source: float, length: int):
        total = 0.0
        for i in range(3):
            total += band(source, length, 1.0 + i)
        return total
