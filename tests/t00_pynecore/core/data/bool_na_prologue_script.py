"""
@pyne

Script for ``test_090``: the three-state choice, a later import of a two-state
library, and a UDT default built at import AFTER that import.
"""
from pynecore.core.pine_udt import udt
from pynecore.lib import na, plot, script
from pynecore.types.na import na_bool

FLAG = 1

import bool_na_prologue_lib as bnl  # noqa: E402


@udt
class Flag:
    f: bool = na(bool)


@script.indicator("Bool na prologue", na_bool=True)
def main():
    plot(bnl.one() if Flag.new().f is na_bool else 0.0, 'udt_seen')
