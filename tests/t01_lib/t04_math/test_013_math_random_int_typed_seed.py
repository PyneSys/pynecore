"""
@pyne

``math.random`` takes an int-TYPED seed.

The seed is consumed by ``PineRandom``'s ``seed ^ 0x5DEECE66D``, which needs an
integer; an int-typed Pine expression carrying a fraction (``(R + z) / 8`` =
1.75) raised ``TypeError: unsupported operand type(s) for ^``.

MEASURED on TradingView (FX:EURUSD@60, ``R = input.int(14)``):
``math.random(0, 100, (R + z) / 8)`` is 73.0878190703 -- the draw of seed 1.
Each call site keeps its own generator, so the two below advance in lockstep.
"""
from pynecore.lib import script, math, bar_index


@script.indicator(title="Int-typed random seed")
def main():
    return {
        "frac": math.random(0, 100, 14 / 8),
        "exact": math.random(0, 100, 1),
        "bar": bar_index,
    }


def __test_math_random_int_typed_seed__(runner, dummy_ohlcv_iter):
    """A fractional seed draws the sequence of its truncated integer"""
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    for i in range(20):
        _candle, plot = next(run_iter)
        if i == 0:
            # The measured TradingView draw, printed there to ten decimals
            assert round(plot["frac"], 10) == 73.0878190703
        assert plot["frac"] == plot["exact"], \
            f"bar {i}: {plot['frac']} != {plot['exact']}"
