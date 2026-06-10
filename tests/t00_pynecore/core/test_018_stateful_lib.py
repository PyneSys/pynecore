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
    """The pre-port manual algorithm on a plain list, as an independent
    bit-exactness reference (NA values are skipped, never buffered)."""
    buf: list[float] = []
    summ, count, comp = 0.0, 0, 0.0

    def ref(source, length):
        nonlocal summ, count, comp
        if length == 1:
            return source
        length = int(length)
        isna = isinstance(source, NA)
        if not isna:
            buf.append(float(source))
        if count < length - 1:
            if not isna:
                count += 1
                corrected = float(source) - comp
                new_sum = summ + corrected
                comp = (new_sum - summ) - corrected
                summ = new_sum
            return NA(float)
        elif count == length - 1:
            if isna:
                return NA(float)
            count += 1
        else:
            if isna:
                return summ
            old = buf[-1 - length]
            corrected_old = -old - comp
            new_sum = summ + corrected_old
            comp = (new_sum - summ) - corrected_old
            summ = new_sum
        corrected = float(source) - comp
        new_sum = summ + corrected
        comp = (new_sum - summ) - corrected
        summ = new_sum
        return summ

    return ref


def __test_math_sum_bit_exact__():
    """ The port matches the manual Kahan algorithm bit for bit, with NA
    values hitting the warmup, transition and steady branches """
    values = [1.1, 2.2, NA(float), 3.3, 0.1, NA(float), NA(float), 4.4, 1e-9, 5.5,
              0.3333333333, 7.7, NA(float), 8.8, 1e12, 0.0001, 9.9, 2.5, NA(float), 6.6]
    state = _make_state(lib.math.sum.__pyne_layout__)
    ref = _ref_sum_factory()
    with _bars() as next_bar:
        for v in values:
            got = lib.math.sum(state, v, 5)
            want = ref(v, 5)
            if isinstance(want, NA):
                assert isinstance(got, NA)
            else:
                assert got == want
            next_bar()


def __test_math_sum_length_one_shortcut__():
    """ length == 1 returns the source untouched, without buffering """
    state = _make_state(lib.math.sum.__pyne_layout__)
    assert lib.math.sum(state, 3.3, 1) == 3.3
    assert isinstance(lib.math.sum(state, NA(float), 1), NA)
    assert state[2] == 0  # the count slot stayed untouched


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
