"""
@pyne

Script for ``test_091``: a three-state script importing a library whose entry
runs on every bar; main plots the library's record and its own, plus a UDT
default built at import.
"""
from pynecore.core.pine_udt import udt
from pynecore.lib import array, na, plot, script
from pynecore.types.na import na_bool

import bool_na_lib_entry_lib as bnl


@udt
class Flag:
    f: bool = na(bool)


@script.indicator("Bool na lib entry", na_bool=True)
def main():
    plot(array.get(bnl.seen, 0), 'lib_seen')
    plot(1.0 if na(bool) is na_bool else 0.0, 'main_seen')
    plot(1.0 if Flag.new().f is na_bool else 0.0, 'udt_seen')
