"""
@pyne

Script for ``test_092``: a two-state caller of a three-state library's UDT.
"""
from pynecore.lib import plot, script

import bool_na_udt_lib3 as bnl


@script.indicator("Bool na UDT caller")
def main():
    plot(1.0 if bnl.make().f is False else 0.0, 'lib_udt_false')
