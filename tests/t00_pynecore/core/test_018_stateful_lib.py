"""
Unit tests for the four stateful lib participants ported to ``@pyne``
submodules (``lib/_fixnan.py``, ``lib/_math_stateful.py``,
``lib/_timeframe_change.py``).

Each ported function carries a ``__pyne_layout__``; the tests instantiate
state vectors directly and drive the bars manually. Full TV-reference
validation runs through the ``t01_lib`` behavior suites.
"""
from contextlib import contextmanager
from datetime import datetime, time, timedelta, UTC
from math import fsum, sin

from pynecore import lib
from pynecore.core.instance_state import _make_state
from pynecore.core.random import PineRandom
from pynecore.core.syminfo import SymInfoInterval, SymInfoSession
from pynecore.lib import syminfo
from pynecore.types.na import NA


@contextmanager
def _bars():
    """Drive ``lib.bar_index`` manually (series adds are bar-indexed)."""
    saved = lib.bar_index
    lib.bar_index = 0

    def next_bar():
        lib.bar_index += 1

    try:
        yield next_bar
    finally:
        lib.bar_index = saved


@contextmanager
def _synthetic_chart(period: str):
    """A 24/7 UTC exchange with daily sessions starting at midnight."""
    saved = (syminfo.period, syminfo.timezone, syminfo._session_starts,
             syminfo._opening_hours, lib.bar_index, lib._datetime)
    syminfo.period = period
    syminfo.timezone = 'UTC'
    syminfo._session_starts = [SymInfoSession(d, time(0, 0)) for d in range(7)]
    syminfo._opening_hours = [SymInfoInterval(d, time(0, 0), time(23, 59, 59)) for d in range(7)]
    lib.bar_index = 0
    try:
        yield
    finally:
        (syminfo.period, syminfo.timezone, syminfo._session_starts,
         syminfo._opening_hours, lib.bar_index, lib._datetime) = saved


### fixnan ###

def __test_fixnan_bridges_na__():
    """ NA values are replaced with the last non-NA value """
    state = _make_state(lib.fixnan.__pyne_layout__)
    assert lib.fixnan(state, 1.0) == 1.0
    assert lib.fixnan(state, NA(float)) == 1.0
    assert lib.fixnan(state, 2.0) == 2.0
    assert lib.fixnan(state, NA(float)) == 2.0


def __test_fixnan_initial_na__():
    """ Before the first non-NA value, fixnan returns NA """
    state = _make_state(lib.fixnan.__pyne_layout__)
    assert isinstance(lib.fixnan(state, NA(float)), NA)


def __test_fixnan_instances_independent__():
    """ Two state vectors track their own last non-NA values """
    a = _make_state(lib.fixnan.__pyne_layout__)
    b = _make_state(lib.fixnan.__pyne_layout__)
    assert lib.fixnan(a, 1.0) == 1.0
    assert lib.fixnan(b, 5.0) == 5.0
    assert lib.fixnan(a, NA(float)) == 1.0
    assert lib.fixnan(b, NA(float)) == 5.0


### math.random ###

def __test_math_random_reference_sequence__():
    """ The ported PRNG produces the exact PineRandom sequence """
    state = _make_state(lib.math.random.__pyne_layout__)
    ref = PineRandom(42)
    got = [lib.math.random(state, 0, 1, 42) for _ in range(5)]
    expected = [ref.random(0, 1) for _ in range(5)]
    assert got == expected


def __test_math_random_lazy_seed_once__():
    """ The PRNG is created lazily on the first call; later seeds are ignored """
    state = _make_state(lib.math.random.__pyne_layout__)
    ref = PineRandom(7)
    first = lib.math.random(state, 0, 1, 7)
    second = lib.math.random(state, 0, 1, 9999)
    assert [first, second] == [ref.random(0, 1), ref.random(0, 1)]


def __test_math_random_instances_independent__():
    """ Two state vectors hold independent PRNG streams """
    a = _make_state(lib.math.random.__pyne_layout__)
    b = _make_state(lib.math.random.__pyne_layout__)
    seq_a = [lib.math.random(a, 0, 1, 3) for _ in range(3)]
    seq_b = [lib.math.random(b, 0, 1, 3) for _ in range(3)]
    assert seq_a == seq_b  # same seed, same stream — not one shared stream


### math.sum ###

def _ref_sum_factory():
    """The TradingView rolling-sum machine on plain lists, written independently
    of ``pynecore.core.rolling_sum`` so it doubles as its regression check.

    MEASURED (2026-10-29, CAPITALCOM:BTCUSD 5m, 20.9k bars per probe): a
    Neumaier-compensated rolling sum -- every add of ``v`` is ``t = s + v``,
    ``c += (s - t) + v`` when ``|s| >= |v|`` else ``c += (v - t) + s``, ``s = t``
    -- read out as ``s + c`` and never renormalized. A stored bar evicts the raw
    value at offset ``length`` (when the window does not reach the first stored
    value) and then adds its own value; a moved length walks the change first,
    one compensated add per value: a shrink evicts the extra leaving offsets
    oldest first, a grow admits the newly covered offsets newest first down to
    the one PAST the window, which the step's own eviction then takes out again.
    An na bar stores nothing and, when its length moved, only resizes the window
    of the values already stored (a grow admits down to the window's own edge
    and evicts nothing)."""
    buf: list[float] = []  # non-na values, oldest first
    summ, comp, width = 0.0, 0.0, 0

    def add(v: float) -> None:
        nonlocal summ, comp
        t = summ + v
        if abs(summ) >= abs(v):
            comp += (summ - t) + v
        else:
            comp += (v - t) + summ
        summ = t

    def ref(source, length):
        nonlocal width
        length = int(length)
        isna = isinstance(source, NA) or source != source
        if not isna:
            buf.append(float(source))
        n = len(buf)
        new_w = min(length, n)
        if isna and new_w == width:
            return summ + comp if new_w >= length else NA(float)
        shift = 0 if isna else 1

        def offset(k: int) -> float:  # window offset k, 0 = newest stored value
            return buf[-1 - k]

        prev_top = width - 1 + shift
        if prev_top + 1 > new_w:
            for k in range(prev_top, new_w, -1):
                add(-offset(k))
        elif new_w > prev_top or (new_w < n and not isna):
            top = new_w - 1 if isna else (new_w if new_w < n else n - 1)
            for k in range(prev_top + 1, top + 1):
                add(offset(k))
        if new_w < n and (not isna or width > new_w):
            add(-offset(new_w))
        if not isna:
            add(buf[-1])
        width = new_w
        return summ + comp if new_w >= length else NA(float)

    return ref


def _assert_same(got, want, where) -> None:
    if isinstance(want, NA) or want != want:
        assert isinstance(got, NA) or got != got, where
    else:
        assert repr(got) == repr(want), (where, got, want)


def __test_math_sum_bit_exact__():
    """ The port matches the reference rolling-sum machine bit for bit, with NA
    values hitting the warmup, transition and steady branches """
    values = [1.1, 2.2, NA(float), 3.3, 0.1, NA(float), NA(float), 4.4, 1e-9, 5.5,
              0.3333333333, 7.7, NA(float), 8.8, 1e12, 0.0001, 9.9, 2.5, NA(float), 6.6]
    state = _make_state(lib.math.sum.__pyne_layout__)
    ref = _ref_sum_factory()
    with _bars() as next_bar:
        for i, v in enumerate(values):
            _assert_same(lib.math.sum(state, v, 5), ref(v, 5), i)
            next_bar()


def __test_math_sum_spike_leaves_the_compensation_live__():
    """ A huge value passing through the window leaves its rounding residue in the
    compensation register, which the readout keeps applying bar after bar: the
    reported sums are then NOT the correctly rounded window sums, and the
    reference machine says exactly which doubles they are (MEASURED, probe
    newsum1: 1e15 spikes through a length-5 window, 20.9k bars bit-exact) """
    values = [1e15 if i % 7 == 0 else 0.5 + 0.25 * sin(i * 12.9898) for i in range(400)]
    state = _make_state(lib.math.sum.__pyne_layout__)
    ref = _ref_sum_factory()
    off_exact = 0
    with _bars() as next_bar:
        for i, v in enumerate(values):
            got = lib.math.sum(state, v, 5)
            _assert_same(got, ref(v, 5), i)
            if i >= 4 and got != fsum(values[i - 4:i + 1]):
                off_exact += 1
            next_bar()
    assert off_exact > 0


def __test_math_sum_length_one_na_bar_echoes_the_window__():
    """ length == 1 gives back the source, and an na bar echoes the last non-na

    MEASURED (probe sumlen4, BINANCE:BTCUSDT 30m, 28746 bars, and again on the
    current engine, probe newsum1): a length-1 sum equals the source on every
    non-na bar, and on an na bar it reports the last NON-NA value rather than
    na -- an na bar is not stored and does not advance the window, exactly like
    every other length.
    """
    state = _make_state(lib.math.sum.__pyne_layout__)
    with _bars() as next_bar:
        assert lib.math.sum(state, 3.3, 1) == 3.3
        next_bar()
        assert lib.math.sum(state, NA(float), 1) == 3.3
        next_bar()
        assert lib.math.sum(state, 4.5, 1) == 4.5


def __test_math_sum_growing_length_keeps_history__():
    """ Growing the length reuses the already buffered samples: the first full
    window is reported as soon as ``length`` values exist, without a fresh
    warmup gap """
    state = _make_state(lib.math.sum.__pyne_layout__)
    results = []
    with _bars() as next_bar:
        for v, length in ((1.0, 2), (2.0, 2), (3.0, 5), (4.0, 5), (5.0, 5)):
            results.append(lib.math.sum(state, v, length))
            next_bar()
    assert results[1] == 3.0  # length 2 window: 1 + 2
    assert results[2] != results[2] or isinstance(results[2], NA)  # only 3 of 5
    assert results[3] != results[3] or isinstance(results[3], NA)  # only 4 of 5
    assert results[4] == 15.0  # 1 + 2 + 3 + 4 + 5, no spurious na gap


def __test_math_sum_grown_length_matches_the_constant_machine__():
    """ A length grown one bar at a time IS the constant-length machine

    MEASURED LAW (probe sumlen2, BINANCE:BTCUSDT 30m, and the current engine's
    ramp probes): ``math.sum(volume, min(bar_index + 1, 610))`` equals
    ``math.sum(volume, 610)`` bit-for-bit on every one of the 28746 exported
    bars, and so does the shorter 5-length pair. While the window is still
    filling there is nothing to evict, so the two runs are the same machine --
    hence the clamped-length idiom (``ta.sma(volume, math.min(bar_index + 1,
    len))``, Heatmap Volume [xdecow]) must not restart the accumulator.
    """
    length = 9
    # Mixed magnitudes so the compensation register carries a live residue
    values = [1e7 + i * 0.1 + (i % 13) * 1e-9 if i % 3 else 3.25e-5 * (i % 7 + 1)
              for i in range(60)]
    grown = _make_state(lib.math.sum.__pyne_layout__)
    fixed = _make_state(lib.math.sum.__pyne_layout__)
    with _bars() as next_bar:
        for i, v in enumerate(values):
            g = lib.math.sum(grown, v, min(i + 1, length))
            f = lib.math.sum(fixed, v, length)
            if i + 1 >= length:
                assert repr(g) == repr(f), f'bar {i}: {g!r} != {f!r}'
            next_bar()


def __test_math_sum_length_changes_walk_one_value_at_a_time__():
    """ A moved length enters and leaves the window ONE value per compensated add,
    in the measured order, and the walk matters

    MEASURED (probe newsum2, CAPITALCOM:BTCUSD 5m: +-5 jumps, sawtooth and
    pseudo-random lengths 2..14 over 1e15 spikes, 20.9k bars each bit-exact
    this way; folding a change into one pre-summed delta, admitting only to the
    window's own edge on a stored bar, or reversing either walk drifts off
    within a few hundred bars). The reference machine is compared here over
    the same kind of schedule, with na bars mixed in, and a folded variant must
    disagree somewhere or the comparison proves nothing.
    """
    values = [1e15 if i % 37 == 0 else 0.5 + 0.25 * sin(i * 12.9898) for i in range(400)]
    values = [NA(float) if i % 11 == 5 else v for i, v in enumerate(values)]
    lengths = [2 + (i * 7919) % 13 if i % 50 < 25 else 3 + i % 7 for i in range(400)]

    def folded():
        # The same machine with every change folded into ONE add of the delta
        buf, summ, comp, width = [], 0.0, 0.0, 0
        out = []
        for i, v in enumerate(values):
            isna = isinstance(v, NA)
            if not isna:
                buf.append(float(v))
            n = len(buf)
            new_w = min(lengths[i], n)
            if isna and new_w == width:
                out.append(summ + comp if new_w >= lengths[i] else None)
                continue
            window = buf[-new_w:] if new_w else []
            prev = (buf[-1 - width:-1] if not isna else buf[-width:]) if width else []
            delta = sum(window) - sum(prev)
            t = summ + delta
            comp += (summ - t) + delta if abs(summ) >= abs(delta) else (delta - t) + summ
            summ = t
            width = new_w
            out.append(summ + comp if new_w >= lengths[i] else None)
        return out

    state = _make_state(lib.math.sum.__pyne_layout__)
    ref = _ref_sum_factory()
    fold = folded()
    disagree = False
    with _bars() as next_bar:
        for i, v in enumerate(values):
            got = lib.math.sum(state, v, lengths[i])
            want = ref(v, lengths[i])
            _assert_same(got, want, i)
            if fold[i] is not None and not isinstance(want, NA) and fold[i] != want:
                disagree = True
            next_bar()
    assert disagree


def __test_math_sum_length_change_on_na_bar_keeps_window__():
    """ A length change on an na bar rebuilds from the na-compacted buffer
    instead of discarding a window that is already complete """
    state = _make_state(lib.math.sum.__pyne_layout__)
    with _bars() as next_bar:
        for v in (1.0, 2.0, 3.0, 4.0):
            lib.math.sum(state, v, 2)
            next_bar()
        # Length grows to 3 on an na bar: 2 + 3 + 4 is available from history
        assert lib.math.sum(state, NA(float), 3) == 9.0
        next_bar()
        assert lib.math.sum(state, 5.0, 3) == 12.0  # 3 + 4 + 5


def __test_math_sum_growth_after_long_short_length_run__():
    """ A long run at a short length must not shrink the buffer: a later growth
    still finds the full trailing window in history """
    state = _make_state(lib.math.sum.__pyne_layout__)
    with _bars() as next_bar:
        for i in range(10):
            lib.math.sum(state, float(i + 1), 2)
            next_bar()
        # 7 + 8 + 9 + 10 + 11 -- all five values are still buffered
        assert lib.math.sum(state, 11.0, 5) == 45.0


def __test_math_sum_growth_on_na_bar_after_long_run__():
    """ Same as above, but the growth lands on an na bar: the window is rebuilt
    from the last five non-na values """
    state = _make_state(lib.math.sum.__pyne_layout__)
    with _bars() as next_bar:
        for i in range(10):
            lib.math.sum(state, float(i + 1), 2)
            next_bar()
        assert lib.math.sum(state, NA(float), 5) == 40.0  # 6 + 7 + 8 + 9 + 10


def __test_math_sum_instances_independent__():
    """ Two state vectors keep separate buffers and accumulators """
    a = _make_state(lib.math.sum.__pyne_layout__)
    b = _make_state(lib.math.sum.__pyne_layout__)
    with _bars() as next_bar:
        for v in (1.0, 2.0, 3.0):
            lib.math.sum(a, v, 2)
            lib.math.sum(b, v * 10, 2)
            next_bar()
        assert lib.math.sum(a, 4.0, 2) == 7.0
        assert lib.math.sum(b, 40.0, 2) == 70.0


### timeframe.change ###

def __test_timeframe_change_intraday__():
    """ On an hourly 24/7 chart, change('240') fires every 4 hours from the
    session anchor (and never on bar 0) """
    with _synthetic_chart('60'):
        state = _make_state(lib.timeframe.change.__pyne_layout__)
        start = datetime(2026, 1, 5, tzinfo=UTC)  # Monday
        fired = []
        for i in range(48):
            lib._datetime = start + timedelta(hours=i)
            lib.bar_index = i
            if lib.timeframe.change(state, '240'):
                fired.append(i)
        assert fired == [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]


def __test_timeframe_change_daily__():
    """ change('D') fires on the first candle of each new session; the
    first-bar anchor replay (host helper) sets up the cycle state """
    with _synthetic_chart('60'):
        state = _make_state(lib.timeframe.change.__pyne_layout__)
        start = datetime(2026, 1, 5, tzinfo=UTC)  # Monday
        fired = []
        for i in range(72):
            lib._datetime = start + timedelta(hours=i)
            lib.bar_index = i
            if lib.timeframe.change(state, 'D'):
                fired.append(i)
        assert fired == [24, 48]


def __test_timeframe_change_smaller_tf_false__():
    """ A timeframe below the chart timeframe never signals """
    with _synthetic_chart('60'):
        state = _make_state(lib.timeframe.change.__pyne_layout__)
        lib._datetime = datetime(2026, 1, 5, tzinfo=UTC)
        lib.bar_index = 5
        assert lib.timeframe.change(state, '1') is False
