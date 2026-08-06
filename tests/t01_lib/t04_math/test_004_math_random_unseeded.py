"""
@pyne

Regression test: ``math.random`` without an explicit seed must produce numbers.

The ``seed`` parameter defaults to na, and the na used to be handed straight to
the PRNG, where it was XOR-ed into the generator state -- every draw of every
unseeded call site came back na.
"""
from pynecore.lib import math, plot, script


@script.indicator(title="Unseeded random", shorttitle="unseeded_rnd")
def main():
    plot(math.random(0, 255), "rnd")


def __test_unseeded_random_draws_numbers__(runner, dummy_ohlcv_iter):
    """ Every bar of an unseeded call site yields a number inside the requested range. """
    # The dummy candle source is an endless cycle, so take a fixed number of bars
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    values = [next(run_iter)[1]['rnd'] for _ in range(10)]

    assert all(v == v for v in values), f"na draws: {values}"
    assert all(0.0 <= v <= 255.0 for v in values), f"out of range: {values}"
    # A time-seeded generator advances every bar; a stuck state would repeat
    assert len(set(values)) > 1, f"the generator did not advance: {values}"
