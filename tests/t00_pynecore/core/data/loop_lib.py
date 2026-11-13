"""
@pyne

Library for ``test_181``: the Pine ``LoopLib``. ``multibands`` calls the
exported ``bb`` on every pass of a loop, so the loop site binds an export --
and when the importer runs in the other bool mode, that binding is reached
through the bool-mode wrapper.
"""
from typing import Any, Protocol

from pynecore.core.pine_export import Exported, export
from pynecore.lib import array, script, ta


class _Bb(Protocol):
    def __call__(self, source: float, length: int, multiplier: float) -> Any: ...


class _Multibands(Protocol):
    def __call__(self, source: float, length: int) -> Any: ...


bb: _Bb = Exported()
multibands: _Multibands = Exported()

__all__ = ['bb', 'multibands']


@script.library("LoopLib")
def main():
    @export
    def bb(source: float, length: int, multiplier: float):
        middle = ta.sma(source, length)
        return middle + ta.stdev(source, length) * multiplier

    @export
    def multibands(source: float, length: int):
        bands = array.new_float(0)
        for i in range(7):
            array.push(bands, bb(source, length, 0.5 + i * 0.5))
        return bands
