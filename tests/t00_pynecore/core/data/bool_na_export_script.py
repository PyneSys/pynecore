"""
@pyne

Script for ``test_096``: an ordinary v6 script (two-state bool) calling the
three-state library's exported function.
"""
from pynecore.lib import na, plot, script
from pynecore.types.na import na_bool

import bool_na_export_lib as lib_three


@script.indicator("Bool na export script")
def main():
    lib_na, state = lib_three.direction(True)
    plot(lib_na, 'lib_na')
    plot(state, 'state')
    # The caller's own choice is untouched by the crossing
    plot(1.0 if na(bool) is na_bool else 0.0, 'main_na')
