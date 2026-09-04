"""
@pyne

Script for ``test_092``: a three-state caller of a two-state library's UDT.
"""
from pynecore.lib import plot, script
from pynecore.types.na import na_bool

import bool_na_udt_lib as bnl


@script.indicator("Bool na UDT caller", na_bool=True)
def main():
    plot(1.0 if bnl.make().f is na_bool else 0.0, 'lib_udt_na')
