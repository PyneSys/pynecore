"""
@pyne
"""
from pynecore.lib import script, close, ta


@script.indicator(title="TA Long Length")
def main():
    return {
        "close": close,
        "stdev600": ta.stdev(close, 600),
        "change600": ta.change(close, 600),
        "wma600": ta.wma(close, 600),
        "cog600": ta.cog(close, 600),
        "corr600": ta.correlation(close, close, 600),
    }


def __test_ta_long_length__(runner):
    """
    Rolling ``ta.*`` functions whose window-drop reads ``series[length]`` must keep
    working for a ``length`` beyond the default 500-bar ``max_bars_back``. Before the
    buffer was grown to fit ``length``, that read fell out of the buffer and returned
    na: the accumulating ones (``variance``/``stdev``, ``wma``, ``cog``, ``correlation``)
    poisoned their running state to na (``stdev`` even settled at a bogus 0.0), while the
    stateless ones (``change``, ``roc``) simply returned na — every one collapsing right
    after warmup instead of tracking TradingView.
    """
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV
    from pynecore.types.na import NA
    import math as pymath

    n_bars = 700
    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    seed = 12345
    price = 100.0
    rows = []
    for bi in range(n_bars):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        price += (seed / 0x7FFFFFFF - 0.5) * 2.0
        rows.append(OHLCV(timestamp=base_ts + bi * 1800, open=price, high=price + 1.0,
                          low=price - 1.0, close=price, volume=10.0))

    # Independent expected values are built from the ``close`` the script actually saw
    # (the engine may quantize prices), pinning correctness, not price representation.
    seen: list[float] = []

    def na(v):
        return v is None or isinstance(v, NA)

    for i, (_c, p) in enumerate(runner(iter(rows)).run_iter()):
        if na(p.get("close")):
            continue
        seen.append(p["close"])
        if i < 599:  # 600-bar windows warm up on bar 599
            continue
        win = seen[i - 599: i + 1]
        mean = sum(win) / 600
        exp_std = pymath.sqrt(sum((x - mean) ** 2 for x in win) / 600)  # biased (population)
        assert not na(p["stdev600"]) and abs(p["stdev600"] - exp_std) < 1e-4, \
            f"stdev600 wrong at bar {i}: {p['stdev600']} vs {exp_std}"
        assert not na(p["wma600"]), f"wma600 na at bar {i}"
        assert not na(p["cog600"]), f"cog600 na at bar {i}"
        assert abs(p["corr600"] - 1.0) < 1e-6, \
            f"correlation of a series with itself must be 1.0 at bar {i}: {p['corr600']}"
        if i >= 600:  # change/roc reach back exactly 600 bars → valid from bar 600
            exp_chg = seen[i] - seen[i - 600]
            assert not na(p["change600"]) and abs(p["change600"] - exp_chg) < 1e-4, \
                f"change600 wrong at bar {i}: {p['change600']} vs {exp_chg}"
