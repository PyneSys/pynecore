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
from fractions import Fraction
from math import isfinite

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

def _ref_fires(comp: float, x: float) -> bool:
    """Exact-arithmetic (Fraction) re-baseline condition, written independently
    of ``pynecore.core.rolling_sum`` so it doubles as its regression check:
    fire iff the realized rounding error of ``fl(x + comp)`` has the opposite
    sign to ``comp`` (ties are already resolved by round-half-even in the
    float add; exact adds never fire)."""
    if comp == 0.0 or x == 0.0 or not (isfinite(comp) and isfinite(x)):
        return False
    r = x + comp
    if r == 0.0 or not isfinite(r):
        return False
    err = Fraction(r) - (Fraction(x) + Fraction(comp))
    if err == 0:
        return False
    return (err < 0) if comp > 0.0 else (err > 0)


def _ref_sum_factory():
    """The manual TV rolling-sum machine on plain lists, as an independent
    bit-exactness reference (NA values are skipped, never buffered): fused
    two-round compensated evict-and-add, with exact-arithmetic re-baseline
    detection and newest-first linear window rebuilds."""
    buf: list[float] = []
    entries: list[float] = []
    summ, count, comp = 0.0, 0, 0.0

    def ref(source, length):
        nonlocal summ, count, comp
        if length == 1:
            return source
        length = int(length)
        isna = isinstance(source, NA) or source != source
        if isna:
            return summ if count >= length else NA(float)
        x = float(source)
        buf.append(x)
        if count < length:  # Warmup: d0 = 0, fires sum the available prefix
            count += 1
            if _ref_fires(comp, x):
                acc = x
                for v in buf[-count:-1][::-1]:
                    acc = v + acc
                summ, comp = acc, 0.0
                entries.append(x)
            else:
                y1 = -comp
                t = summ + y1
                e1 = (t - summ) - y1
                y2 = x - e1
                new_sum = t + y2
                comp = (new_sum - t) - y2
                summ = new_sum
                entries.append(y2)
            return summ if count == length else NA(float)
        old = entries.pop(0)
        if _ref_fires(comp, x):
            acc = x
            for v in buf[-length:-1][::-1]:
                acc = v + acc
            summ, comp = acc, 0.0
            entries.append(x)
        else:
            y1 = -old - comp
            t = summ + y1
            e1 = (t - summ) - y1
            y2 = x - e1
            new_sum = t + y2
            comp = (new_sum - t) - y2
            summ = new_sum
            entries.append(y2)
        return summ

    return ref


def __test_math_sum_bit_exact__():
    """ The port matches the manual rolling-sum machine bit for bit, with NA
    values hitting the warmup, transition and steady branches """
    values = [1.1, 2.2, NA(float), 3.3, 0.1, NA(float), NA(float), 4.4, 1e-9, 5.5,
              0.3333333333, 7.7, NA(float), 8.8, 1e12, 0.0001, 9.9, 2.5, NA(float), 6.6]
    state = _make_state(lib.math.sum.__pyne_layout__)
    ref = _ref_sum_factory()
    with _bars() as next_bar:
        for v in values:
            got = lib.math.sum(state, v, 5)
            want = ref(v, 5)
            if isinstance(want, NA) or want != want:
                assert isinstance(got, NA) or got != got
            else:
                assert got == want
            next_bar()


def __test_math_sum_length_one_shortcut__():
    """ length == 1 returns the source untouched, without buffering """
    state = _make_state(lib.math.sum.__pyne_layout__)
    assert lib.math.sum(state, 3.3, 1) == 3.3
    na_result = lib.math.sum(state, NA(float), 1)
    assert na_result != na_result  # the untouched na source is the native nan
    assert state[2] == 0  # the count slot stayed untouched


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
        # 7 + 8 + 9 + 10 + 11 — all five values are still buffered
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
