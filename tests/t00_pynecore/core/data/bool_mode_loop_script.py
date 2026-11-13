"""
@pyne

Script for ``test_181``: a three-state script reaching the two-state library's
stateful export from a loop, both its own and the library's. The same functions
defined locally give the reference: no module boundary, no bool-mode crossing.
"""
from pynecore.lib import close, plot, script, ta

import bool_mode_loop_lib as loop_lib


def band(source: float, length: int, mult: float):
    return ta.sma(source, length) + ta.stdev(source, length) * mult


def bands(source: float, length: int):
    total = 0.0
    for i in range(3):
        total += band(source, length, 1.0 + i)
    return total


@script.indicator("Bool mode loop script", na_bool=True)
def main():
    outer = 0.0
    outer_ref = 0.0
    for i in range(3):
        outer += loop_lib.band(close, 5, 1.0 + i)
        outer_ref += band(close, 5, 1.0 + i)
    plot(outer, 'outer')
    plot(outer_ref, 'outer_ref')
    plot(loop_lib.bands(close, 5), 'inner')
    plot(bands(close, 5), 'inner_ref')
