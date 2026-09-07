"""
@pyne
"""
from pynecore import Persistent
from pynecore.lib import script, math, na, plot


@script.indicator(title="Math Sum NA Shrink", shorttitle="math_sum_na_shrink")
def main():
    # A window that shrinks by more than one offset ON AN NA BAR: nothing is
    # stored, so the buffer does not move under the window and every leaving
    # offset is its own ``src`` index. Offsets 3..4 leave here, offset 3 through
    # the fused step and offset 4 through the eviction walk; reading the walk
    # one index lower drops offset 4 out of the eviction and takes offset 3
    # twice, which leaves the accumulator permanently short on every later bar.
    # The window is small exact integers, so the expected sums below are the
    # window contents themselves — no compensated-arithmetic residue can move
    # them, and any machine that tracks the right offsets has to land on them.
    i: Persistent[int] = 0
    i += 1
    if i <= 10:
        v = float(i)
        ln = 5
    elif i == 11:
        v = na  # window 6..10 shrinks to 8..10 with nothing entering
        ln = 3
    else:
        v = float(i - 1)
        ln = 3
    plot(math.sum(v, ln), "s")


def __test_math_sum_na_shrink__(runner, dummy_ohlcv_iter):
    """ math.sum() - a window shrunk on an na bar evicts the offsets that leave """
    # Bars 1..4 are warmup (fewer than ``length`` values), 5..10 are the steady
    # 5-wide window, 11 is the na shrink and 12..14 the 3-wide tail after it
    expected = [None, None, None, None,
                15.0, 20.0, 25.0, 30.0, 35.0, 40.0,
                27.0, 30.0, 33.0, 36.0]
    values = []
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    for _ in range(len(expected)):
        _candle, plots = next(run_iter)
        values.append(plots['s'])

    for bar, (got, want) in enumerate(zip(values, expected), start=1):
        if want is None:
            assert got != got, f"bar {bar}: expected na, got {got}"
        else:
            assert got == want, f"bar {bar}: expected {want}, got {got}"
