"""
@pyne
"""
from pynecore.lib import bar_index, close, high, input, na, script, strategy
from pynecore.types import Persistent


@script.strategy("Coin flip", initial_capital=10000,
                 default_qty_type=strategy.fixed, default_qty_value=1)
def main(stop_entry=input.bool(False), repeat=input.bool(False), stop_limit=input.bool(False),
         replace=input.bool(False), reverse=input.bool(False), after_fill=input.bool(False),
         every=input.int(20)):
    phase = bar_index % every
    flat = strategy.position_size == 0
    stop_level: Persistent[float] = na
    target_level: Persistent[float] = na
    # The bracket is placed once the entry fills, from the side the position reads
    if after_fill and flat and phase == 5:
        stop_level = close - 2.0
        target_level = close + 4.0
        strategy.entry("L", strategy.long)
    if after_fill and not flat:
        is_long = strategy.position_size > 0
        risk = (strategy.position_avg_price - stop_level if is_long
                else stop_level - strategy.position_avg_price)
        if risk > 0:
            strategy.exit("LX", "L", stop=stop_level, limit=target_level)
    # A far limit entry rests for a bar, then a stop-limit entry replaces it
    if replace and flat and phase == 5:
        strategy.entry("L", strategy.long, limit=close - 20.0)
        strategy.exit("LX", "L", stop=close - 22.0, limit=close - 16.0)
    if replace and flat and phase == 6:
        strategy.entry("L", strategy.long, stop=high + 0.5, limit=high + 1.0)
        strategy.exit("LX", "L", stop=close - 2.0, limit=close + 4.0)
    # A market entry, then a stop-limit entry of its id: a flipped (short) trade
    # is reversed by it, an unflipped one only repeats it
    if reverse and flat and phase == 5:
        strategy.entry("L", strategy.long)
    if reverse and not flat and phase == 6:
        strategy.entry("L", strategy.long, stop=high + 0.1, limit=high + 1.0)
    if reverse and strategy.position_size > 0:
        strategy.exit("LX", "L", stop=strategy.position_avg_price - 2.0,
                      limit=strategy.position_avg_price + 4.0)
    entry_bar = flat and phase == 5 and not replace and not reverse and not after_fill
    if repeat and strategy.position_size != 0 and 5 < phase < 9:
        # The entry signal holds on while its trade is open
        strategy.entry("L", strategy.long)
        strategy.exit("LX", "L", stop=strategy.position_avg_price - 2.0,
                      limit=strategy.position_avg_price + 4.0)
    alternate = stop_limit and (bar_index // every) % 2 == 1
    if entry_bar and alternate:
        strategy.entry("L", strategy.long, stop=high + 0.5, limit=high + 1.0)
        strategy.exit("LX", "L", stop=close - 2.0, limit=close + 4.0)
    if entry_bar and not stop_entry and not alternate:
        strategy.entry("L", strategy.long)
        strategy.exit("LX", "L", stop=close - 2.0, limit=close + 4.0)
    if entry_bar and stop_entry:
        strategy.entry("L", strategy.long, stop=high + 0.5)
        strategy.exit("LX", "L", stop=close - 2.0, limit=close + 4.0)


def __test_helper_bars():
    """A sine wave of hourly bars that reaches 2 below and 4 above most entries."""
    import math
    from pynecore.types.ohlcv import OHLCV
    start = 1_735_689_600_000
    bars = []
    for i in range(600):
        mid = 100.0 + 6.0 * math.sin(i / 6.0)
        bars.append(OHLCV(timestamp=start + i * 3_600_000, open=mid, high=mid + 1.5,
                          low=mid - 1.5, close=mid + 0.2, volume=1.0))
    return bars


def __test_helper_factory(runner, **inputs):
    bars = __test_helper_bars()

    def make():
        return runner(iter(bars), inputs=inputs, last_bar_index=len(bars) - 1,
                      last_bar_time=bars[-1].timestamp)

    return make


def __test_helper_record(make, flip=None):
    from pynecore.analysis import TradeRecorder
    from pynecore.lib import strategy as engine
    r = make()
    recorder = TradeRecorder(r.script.position, r.syminfo)
    engine._entry_flip = engine._EntryFlip(flip) if flip is not None else None
    try:
        for _ in r.run_iter(copy_results=False):
            recorder.on_bar()
    finally:
        engine._entry_flip = None
    return recorder.trades, r.syminfo


def __test_flipped_entry_mirrors_its_bracket__(runner):
    """A flipped entry fills at the same moment and price; its stop is mirrored."""
    make = __test_helper_factory(runner)
    real, _ = __test_helper_record(make)
    flipped, _ = __test_helper_record(make, lambda _id, _bar: True)
    by_entry = {(t.entry_bar_index, t.entry_id): t for t in flipped}
    assert len(real) > 5
    for t in real:
        f = by_entry[(t.entry_bar_index, t.entry_id)]
        assert f.direction == -t.direction
        assert f.entry_price == t.entry_price
        assert f.initial_stop is not None and t.initial_stop is not None
        assert abs((f.initial_stop - f.entry_price) + (t.initial_stop - t.entry_price)) < 1e-6
        if f.exit_price == f.initial_stop:
            assert abs(f.r + 1.0) < 1e-6


def __test_flipped_stop_entry_keeps_its_fill__(runner):
    """A stop entry flips into a limit entry at the same level, filling identically."""
    make = __test_helper_factory(runner, stop_entry=True)
    real, _ = __test_helper_record(make)
    flipped, _ = __test_helper_record(make, lambda _id, _bar: True)
    assert len(real) > 3
    assert ([(t.entry_bar_index, t.entry_price) for t in real]
            == [(t.entry_bar_index, t.entry_price) for t in flipped])
    assert all(t.direction == -1 for t in flipped)


def __test_repeated_entry_keeps_its_flip__(runner):
    """An entry call repeated while its trade is open keeps the trade's coin and mirror."""
    make = __test_helper_factory(runner, repeat=True)
    always, _ = __test_helper_record(make, lambda _id, _bar: True)
    first_only, _ = __test_helper_record(make, lambda _id, bar: bar % 20 == 5)

    def key(trades):
        return [(t.entry_bar_index, t.exit_bar_index, t.direction, t.entry_price, t.exit_price,
                 t.initial_stop) for t in trades]

    assert len(always) > 5 and all(t.direction == -1 for t in always)
    assert key(first_only) == key(always)


def __test_stop_limit_entry_drops_an_earlier_flip__(runner):
    """A stop-limit entry reusing the id of a closed flipped trade keeps unmirrored exits."""
    make = __test_helper_factory(runner, stop_limit=True)
    real, _ = __test_helper_record(make)
    flipped, _ = __test_helper_record(make, lambda _id, _bar: True)

    def key(t):
        return t.entry_bar_index, t.exit_bar_index, t.entry_price, t.exit_price, t.initial_stop

    kept = [t for t in flipped if t.direction == 1]
    assert kept and any(t.direction == -1 for t in flipped)
    real_by_entry = {t.entry_bar_index: t for t in real}
    for t in kept:
        assert t.initial_stop is not None and t.initial_stop < t.entry_price
        assert key(t) == key(real_by_entry[t.entry_bar_index])


def __test_stop_limit_replacing_a_flipped_order_drops_its_flip__(runner):
    """A stop-limit entry replacing a pending flipped entry keeps unmirrored exits."""
    make = __test_helper_factory(runner, replace=True)
    real, _ = __test_helper_record(make)
    flipped, _ = __test_helper_record(make, lambda _id, _bar: True)

    def key(t):
        return (t.entry_bar_index, t.exit_bar_index, t.direction, t.entry_price, t.exit_price,
                t.initial_stop)

    assert len(real) > 3
    assert all(t.initial_stop is not None and t.initial_stop < t.entry_price for t in flipped)
    assert [key(t) for t in flipped] == [key(t) for t in real]


def __test_stop_limit_reversing_a_flipped_trade_keeps_its_own_exits__(runner):
    """A stop-limit entry reversing an open flipped trade of its id gets unmirrored exits."""
    make = __test_helper_factory(runner, reverse=True)
    flipped, _ = __test_helper_record(make, lambda _id, _bar: True)
    reversals = [t for t in flipped if t.direction == 1]
    assert reversals and any(t.direction == -1 for t in flipped)
    for t in reversals:
        # The leg the script stated for its pending entry, below the long's fill
        assert t.initial_stop is not None and t.initial_stop < t.entry_price
        assert min(abs(t.exit_price - (t.entry_price - 2.0)),
                   abs(t.exit_price - (t.entry_price + 4.0))) < 0.011


def __test_direction_test__(runner):
    """The fast path pairs every trade, compares the real total, repeats with its seed."""
    from pynecore.analysis.coinflip import direction_test
    from pynecore.lib import strategy as engine
    make = __test_helper_factory(runner)
    real, syminfo = __test_helper_record(make)
    a = direction_test(make, syminfo, real, 'r', seed=3, draws=500)
    b = direction_test(make, syminfo, real, 'r', seed=3, draws=500)
    assert a.totals == b.totals and a.draws == 500
    assert a.unpaired == 0 and a.trades == len(real)
    assert abs(a.real - sum(t.r for t in real)) < 1e-9
    assert 0.0 <= a.percentile <= 1.0
    assert engine._entry_flip is None


def __test_exact_direction_test__(runner):
    """The exact path runs a backtest per draw with a coin per entry."""
    from pynecore.analysis.coinflip import coin, direction_test
    make = __test_helper_factory(runner)
    real, syminfo = __test_helper_record(make)
    result = direction_test(make, syminfo, real, 'r', seed=1, exact=True, runs=12)
    assert result.method == 'runs' and result.draws == 12
    assert len(set(round(t, 9) for t in result.totals)) > 1
    assert coin(1, 2, 3, 'L') == coin(1, 2, 3, 'L')


def __test_timing_test__(runner):
    """The real trades re-enter at random moments; the script places no order."""
    from pynecore.analysis import TradeRecorder
    from pynecore.analysis.coinflip import _TimingDriver, _plans, timing_test
    from pynecore.lib import strategy as engine
    make = __test_helper_factory(runner)
    real, syminfo = __test_helper_record(make)
    plans = _plans(real)

    r = make()
    driver = _TimingDriver(plans, list(range(10, 10 + len(plans))), r.script.position,
                           float(syminfo.mintick))
    recorder = TradeRecorder(r.script.position, syminfo)
    engine._orders_blocked = True
    try:
        for _ in r.run_iter(copy_results=False):
            driver.on_bar()
            recorder.on_bar()
    finally:
        engine._orders_blocked = False
    random_trades = recorder.trades
    assert random_trades and all(t.entry_id == 'coinflip' for t in random_trades)
    for t, plan in zip(random_trades, plans):
        assert t.direction == plan.direction
        assert t.initial_stop is not None and plan.risk_fraction is not None
        fraction = abs(t.entry_price - t.initial_stop) / t.entry_price
        assert abs(fraction - plan.risk_fraction) < 0.01

    result = timing_test(make, syminfo, real, 'r', seed=2, runs=10)
    assert result.draws == 10 and result.mode == 'timing'
    assert engine._orders_blocked is False


def __test_flipped_entry_reads_its_own_direction__(runner):
    """A flipped entry's script reads its own side, so a bracket placed after the fill mirrors."""
    make = __test_helper_factory(runner, after_fill=True)
    real, _ = __test_helper_record(make)
    flipped, _ = __test_helper_record(make, lambda _id, _bar: True)

    def key(trades):
        return [(t.entry_bar_index, t.entry_price) for t in trades]

    assert len(real) > 5 and all(t.direction == 1 for t in real)
    assert key(flipped) == key(real)
    for t, f in zip(real, flipped):
        assert f.direction == -1
        assert t.initial_stop is not None and f.initial_stop is not None
        assert abs((f.initial_stop - f.entry_price) + (t.initial_stop - t.entry_price)) < 1e-6


def __test_direction_test_without_pairs_gives_no_verdict__(runner):
    """Too few paired trades leave the fast path without a verdict, and its JSON stays valid."""
    import dataclasses
    import json
    import math
    from pynecore.analysis.coinflip import direction_test
    make = __test_helper_factory(runner)
    real, syminfo = __test_helper_record(make)
    # Real trades under an id the flipped backtest never opens: nothing pairs
    renamed = [dataclasses.replace(t, entry_id='other') for t in real]
    result = direction_test(make, syminfo, renamed, 'r', seed=3, draws=500)
    assert not result.valid and result.trades == 0 and result.unpaired == len(real)
    assert result.draws == 0 and not result.totals
    assert math.isnan(result.percentile) and math.isnan(result.share_better)
    assert result.verdict.startswith('not valid for this script')
    data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert data['valid'] is False and data['percentile'] is None
    assert direction_test(make, syminfo, real, 'r', seed=3, draws=50).valid
