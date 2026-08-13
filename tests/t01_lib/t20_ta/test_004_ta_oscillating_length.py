"""
@pyne
"""
from pynecore.lib import script, close, ta, bar_index


@script.indicator(title="TA Oscillating Length")
def main():
    # A length that drops and jumps back on the very next bar. An adaptive-period
    # script (``xHMA+``: a var float nudged by a percentage every bar) reaches the
    # same state more slowly; alternating every bar leaves no room for the buffer to
    # refill between the dip and the next full-length read.
    length = 40 if bar_index % 2 == 0 else 5
    return {
        "close": close,
        "length": length,
        "change": ta.change(close, length),
        "cog": ta.cog(close, length),
        "dev": ta.dev(close, length),
        "linreg": ta.linreg(close, length, 0),
        "percentrank": ta.percentrank(close, length),
        "rci": ta.rci(close, length),
        "roc": ta.roc(close, length),
        "wma": ta.wma(close, length),
    }


def __test_ta_oscillating_length__(runner, log):
    """
    Rolling ``ta.*`` functions must survive a series ``length`` that falls and rises.

    Each of these grows its own buffer to ``length`` bars. The resize used to follow
    the length in both directions, so a dip THREW AWAY the history the next increase
    needed: the slicing ones (``wma``, ``percentrank``, ``rci``) crashed with
    ``IndexError: Slice stop index out of range``, and the indexing ones silently read
    na past the shortened buffer. Every resize is monotonic now.

    Found on ``72s Strat: Backtesting Adaptive HMA+ pt.1`` in the wild corpus.
    """
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV
    from pynecore.types.na import NA

    n_bars = 400
    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    seed = 987654321
    price = 100.0
    rows = []
    for bi in range(n_bars):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        price += (seed / 0x7FFFFFFF - 0.5) * 2.0
        rows.append(OHLCV(timestamp=base_ts + bi * 1800, open=price, high=price + 1.0,
                          low=price - 1.0, close=price, volume=10.0))

    def na(v):
        return v is None or isinstance(v, NA)

    columns = ("change", "cog", "dev", "linreg", "percentrank", "rci", "roc", "wma")
    lengths_seen = set()
    checked = 0
    for i, (_c, p) in enumerate(runner(iter(rows)).run_iter()):
        lengths_seen.add(p["length"])
        if i < 80:  # the longest window (40) has warmed up well before this
            continue
        checked += 1
        for name in columns:
            assert not na(p[name]), f"{name} went na at bar {i} (length={p['length']})"

    assert len(lengths_seen) > 1, "the length never changed — the test proves nothing"
    assert checked > 0, "no bars were checked"
    log.info("Oscillating length verified on %d bars over %d distinct lengths",
             checked, len(lengths_seen))
