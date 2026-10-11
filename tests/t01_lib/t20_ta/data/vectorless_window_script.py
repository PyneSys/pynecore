"""
@pyne

Script for ``test_377``: reaches the library's window helper through its
relaying export (an instance created with no vector) next to the same windows
called directly with the same const length.
"""
from pynecore.lib import close, plot, script, ta

import vectorless_window_lib as vwl


@script.indicator(title="Vectorless window instance")
def main():
    hi, lo, hib, lob, lr = vwl.relay(close, 5)
    plot(hi, "hi")
    plot(lo, "lo")
    plot(hib, "hib")
    plot(lob, "lob")
    plot(lr, "lr")
    plot(ta.highest(close, 5), "hi_direct")
    plot(ta.lowest(5), "lo_direct")
    plot(ta.highestbars(close, 5), "hib_direct")
    plot(ta.lowestbars(5), "lob_direct")
    plot(ta.linreg(close, 5, 0), "lr_direct")
