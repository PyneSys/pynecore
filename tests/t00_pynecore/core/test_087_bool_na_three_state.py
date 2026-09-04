"""
@pyne

With ``na_bool=True`` the script keeps Pine v4/v5's three-state bool (measured
on TradingView): a bool history before warm-up, ``na(bool)``, a fresh
``array<bool>`` element, an unset UDT bool field and ``ta.change`` on the
first bar are na; ``==`` and ``!=`` propagate the na while ``not``, ``and``,
``or``, a branch and ``nz`` collapse it to false.
"""
from pynecore.core.pine_udt import udt
from pynecore.lib import script, close, open, na, nz, array, ta, string
from pynecore.types.na import NA, na_bool
from pynecore.types.series import Series


@udt
class Flag:
    f: bool = na(bool)


@script.indicator(title="three-state bool", na_bool=True)
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
        "ne": prev != True,  # noqa: E712
        "not": not prev,
        "and": prev and True,
        "or": prev or True,
        "branch": 1 if prev else 0,
        "tostr": string.tostring(prev),
        "nz": nz(prev),
        "nz_true": nz(prev, True),
        "is_na": na(prev),
    }


def __test_a_bool_keeps_its_third_state__(csv_reader, runner):
    """Every bool na source answers na on the first bar, the way Pine v5 does"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            if i == 0:
                for key in ("prev", "na", "arr", "udt", "change"):
                    assert plot[key] is na_bool, f"{key} is {plot[key]!r}"
                # == and != propagate the na (still false in a condition)
                assert plot["eq"] is na_bool and plot["ne"] is na_bool
                # not / and / or / a branch collapse it to false
                assert plot["not"] is True
                assert not plot["and"] and plot["or"]
                assert plot["branch"] == 0
                assert plot["tostr"] == "NaN"
                assert plot["nz"] is False and plot["nz_true"] is True
                assert plot["is_na"] is True
            else:
                assert type(plot["prev"]) is bool
                assert not isinstance(plot["change"], NA)
                break
