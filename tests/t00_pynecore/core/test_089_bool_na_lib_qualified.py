"""
@pyne

The three-state bool switch is read off a ``lib.script`` qualified decorator too,
and it holds while this module's own class bodies run: the UDT field default
below is built at import, before the decorator executes.
"""
from pynecore import lib
from pynecore.core.pine_udt import udt
from pynecore.lib import close, open, na
from pynecore.types.na import na_bool
from pynecore.types.series import Series


@udt
class Flag:
    f: bool = na(bool)


@lib.script.indicator(title="lib-qualified three-state bool", na_bool=True)
def main():
    b: Series[bool] = close > open
    return {"prev": b[1], "udt": Flag.new().f}


def __test_the_switch_holds_for_the_module_body__(csv_reader, runner):
    """The UDT default and the history na are the bool na, not false"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            assert plot["udt"] is na_bool
            assert plot["prev"] is na_bool
            break
