"""@pyne
SPDX-License-Identifier: Apache-2.0
Synthetic example: compute a moving average independently for each offset.
"""
from pynecore.lib import close, input, plot, script, ta


@script.indicator("Synthetic direct feature")
def main(offset=input.float(0), length=input.int(7)):
    plot(ta.sma(close, length) + offset, "value")
