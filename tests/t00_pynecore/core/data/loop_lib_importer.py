"""
@pyne

Importer for ``test_181``: a v5 script reading a bool with history runs in the
three-state bool mode, while ``loop_lib`` does not -- the library's exports are
reached through the bool-mode wrapper.
"""
from pynecore.lib import array, close, open, plot, script
from pynecore.types import Series

import loop_lib as tl


@script.indicator("Importer", na_bool=True)
def main():
    c: Series[bool] = close > open
    plot(1.0 if c[1] else 0.0, "c")
    b = tl.multibands(close, 20)
    plot(array.get(b, 6), "b6")
