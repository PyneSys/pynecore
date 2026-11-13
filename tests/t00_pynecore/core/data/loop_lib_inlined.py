"""
@pyne

Reference for ``test_181``: the library's functions inlined into the script, so
no module boundary and no bool-mode wrapper stands between the loop and the
``ta.*`` machines it reaches.
"""
from pynecore.lib import array, close, open, plot, script, ta
from pynecore.types import Series


def bb(source: float, length: int, multiplier: float):
    middle = ta.sma(source, length)
    return middle + ta.stdev(source, length) * multiplier


def multibands(source: float, length: int):
    bands = array.new_float(0)
    for i in range(7):
        array.push(bands, bb(source, length, 0.5 + i * 0.5))
    return bands


@script.indicator("Inlined", na_bool=True)
def main():
    c: Series[bool] = close > open
    plot(1.0 if c[1] else 0.0, "c")
    b = multibands(close, 20)
    plot(array.get(b, 6), "b6")
