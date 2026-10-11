"""
@pyne

Library module for ``test_377``, shaped like ``enhanced_ta``'s ``ma`` ->
``customseries``: the library main calls ``relay`` with a const length and with
a series length, so the type pass gives the window sites one verdict per
context, and both ``relay`` and ``windows`` carry an instance vector -- ``relay``
hands its own entry on to ``windows``. A script calling ``relay`` through the
import creates it with no vector, so every window site receives
``_series_length=None``.
"""
from pynecore.core.pine_export import Exported, export
from pynecore.lib import bar_index, close, script, ta

relay = Exported()


@script.library("Vectorless Window Lib")
def main():
    def windows(src: float, n: int) -> tuple[float, float, int, int, float]:
        return (ta.highest(src, n), ta.lowest(n), ta.highestbars(src, n), ta.lowestbars(n),
                ta.linreg(src, n, 0))

    @export
    def relay(src: float, n: int) -> tuple[float, float, int, int, float]:
        return windows(src, n)

    const_hi, _, _, _, _ = relay(close, 5)
    series_hi, _, _, _, _ = relay(close, 5 if bar_index >= 0 else 6)
    return {"const_hi": const_hi, "series_hi": series_hi}
