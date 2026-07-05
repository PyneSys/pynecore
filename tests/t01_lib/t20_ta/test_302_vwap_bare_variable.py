"""
@pyne
"""
from pynecore.lib import script, ta, hlc3, plot


@script.indicator(title="VWAP bare variable", overlay=True)
def main():
    plot(ta.vwap, "bare")
    plot(ta.vwap(hlc3), "explicit")


# noinspection PyShadowingNames
def __test_vwap_bare_equals_hlc3__(csv_reader, runner, log):
    """
    The bare ``ta.vwap`` reference is the Pine built-in variable form: it must be
    identical to the explicit ``ta.vwap(hlc3)`` function call on every bar.
    """
    from pathlib import Path
    syminfo_path = Path(__file__).parent / "data" / "vwap.toml"

    def eq(a: float, b: float) -> bool:
        a_na, b_na = (a != a), (b != b)
        if a_na or b_na:
            return a_na and b_na
        return abs(a - b) < 1e-9

    checked = 0
    with csv_reader('vwap.csv', subdir="data") as cr:
        for i, (_candle, plot) in enumerate(runner(cr, syminfo_path=syminfo_path).run_iter()):
            assert eq(plot['bare'], plot['explicit']), \
                f"bar {i}: bare={plot['bare']} explicit={plot['explicit']}"
            checked += 1
            if i > 300:
                break

    assert checked > 200
