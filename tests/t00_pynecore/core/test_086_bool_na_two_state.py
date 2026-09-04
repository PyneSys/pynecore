"""
@pyne

A bool has two states unless the script asks for Pine v4/v5's third one: with
the default ``na_bool=False`` (Pine v6, measured on TradingView) a bool history
before warm-up, ``na(bool)``, a fresh ``array<bool>`` element, an unset UDT
bool field and ``ta.change`` on the first bar are all plain ``false``.
"""
from pynecore.core.pine_udt import udt
from pynecore.lib import script, close, open, na, nz, array, ta, string
from pynecore.types.series import Series


@udt
class Flag:
    f: bool = na(bool)


@script.indicator(title="two-state bool")
def main():
    b: Series[bool] = close > open
    prev = b[1]
    return {
        "prev": prev,
        "na": na(bool),
        "arr": array.get(array.new_bool(1), 0),
        "udt": Flag.new().f,
        "change": ta.change(b),
        "eq": prev == False,  # noqa: E712
        "tostr": string.tostring(prev),
        "nz": nz(prev),
        "is_na": na(prev),
    }


def __test_a_bool_is_never_na__(csv_reader, runner):
    """Every bool na source answers false on the first bar"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            if i == 0:
                for key in ("prev", "na", "arr", "udt", "change", "nz"):
                    assert plot[key] is False, f"{key} is {plot[key]!r}"
                assert plot["eq"] is True
                assert plot["tostr"] == "false"
                assert plot["is_na"] is False
            else:
                assert type(plot["prev"]) is bool
                break
