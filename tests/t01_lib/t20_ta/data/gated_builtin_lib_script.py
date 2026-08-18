"""
@pyne

Script for ``test_368``: reads the library's builtin-variable helper both
unconditionally and inside a gate, next to the unconditional engine series.
"""
from pynecore.lib import plot, script, ta, bar_index

import gated_builtin_lib as gbl


@script.indicator(title="Gated builtin via library")
def main():
    every_nvi = ta.nvi
    all_lib = gbl.lib_nvi()
    gated = -1.0
    if bar_index % 2 == 0:
        gated = gbl.lib_nvi()
    plot(every_nvi, "every_nvi")
    plot(all_lib, "all_lib")
    plot(gated, "gated")
