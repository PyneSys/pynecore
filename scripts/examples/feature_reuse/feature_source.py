"""@pyne
SPDX-License-Identifier: Apache-2.0
Synthetic example: compute the shared moving average once.
"""
from pynecore.lib import close, plot, script, ta


@script.indicator("Synthetic shared feature")
def main():
    plot(ta.sma(close, 7), "feature")
