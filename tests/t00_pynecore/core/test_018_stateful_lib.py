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
from pynecore.core.rolling_sum import sum_fires
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
    of ``pynecore.core.rolling_sum`` so it doubles as its regression check.

    The engine shifts the incoming value by the compensation's MAGNITUDE and
    tests the fixed-order Fast2Sum residue of that add, WITHOUT the usual
    magnitude swap: fire iff ``fl(|c| - fl(fl(x + |c|) - x)) > 0``. Every
    rounding step is re-derived here from rationals — ``float(Fraction)`` is
    correctly rounded — so the oracle shares no float expression with the
    production code."""
    if comp == 0.0 or x == 0.0 or not (isfinite(comp) and isfinite(x)):
        return False
    b = Fraction(-comp if comp < 0.0 else comp)
    xf = Fraction(x)
    try:
        r = float(xf + b)  # fl(x + |c|)
    except OverflowError:  # the add itself overflows to +-inf
        return False
    if r == 0.0 or not isfinite(r):
        return False
    d = float(Fraction(r) - xf)  # fl(fl(x + |c|) - x), inexact when |c| > |x|
    return float(b - Fraction(d)) > 0.0


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


def _fire_rule_pairs():
    """Deterministic ``(compensation, value)`` pairs spanning ~30 binades in
    both signs — a plain LCG so the sweep is reproducible everywhere."""
    s = 12345
    for i in range(20000):
        s = (s * 1103515245 + 12345) % (1 << 31)
        c = (s / (1 << 31) - 0.5) * 10.0 ** (i % 27 - 24)
        s = (s * 1103515245 + 12345) % (1 << 31)
        x = (s / (1 << 31) - 0.5) * 10.0 ** (i % 13 - 7)
        yield c, x


def __test_rolling_sum_fire_rule_golden__():
    """ Golden bars that separate the measured rule from its plausible
    neighbours: the magnitude shift (a signed probe walks the finer downward
    grid below a binade edge) and the missing Fast2Sum magnitude swap """
    # (compensation, value, fires)
    cases = [
        # Negative compensation at a binade edge: shifting the signed value
        # instead of its magnitude labels these bars the other way.
        (-2.117582368135751e-22, 9.536743164062499e-07, True),
        (-0.001055446444665531, 0.0009760761164057203, True),
        (-1.5096162171497207, 1.3090738838615936e-06, False),
        # |c| > |x|: the fixed-order residue is inexact, and that inexact
        # value is what the engine tests — an exact 2Sum would fire here.
        (55.99389814121456, 0.3611502021234201, False),
        (-20.420464290753458, -6.986601940076164e-09, False),
        (9.109360478429427e-07, -2.7167312094343507e-08, False),
        # Degenerate operands never fire.
        (0.0, 1.0, False),
        (1.0, 0.0, False),
        (float('nan'), 1.0, False),
        (1e308, 1e308, False),
    ]
    for comp, value, want in cases:
        assert sum_fires(comp, value) is want, (comp, value)
        assert _ref_fires(comp, value) is want, (comp, value)


def __test_rolling_sum_fire_rule_matches_exact_oracle__():
    """ The float fire detector agrees with the exact-arithmetic oracle on a
    dense deterministic sweep """
    for comp, value in _fire_rule_pairs():
        assert sum_fires(comp, value) == _ref_fires(comp, value), (comp, value)


def __test_math_sum_inlined_fire_rule_matches_core__():
    """ ``lib.math.sum`` inlines the fire test instead of calling
    ``core.rolling_sum.sum_fires``; this window makes the two disagree on the
    output bars if the inlined expression ever drifts from the core one """
    values = [-1274720.6437522452, -3.341532690643485, 8.450734784875934,
              2.050491000209509e-10, -1.9922460506784567, -1844830.279632077,
              -2.9881426742597395, -1.3691775656187065e-06, -5.087091405520743e-06,
              -5.499330175468724e-06, -5.198905520846737e-06, -2.0588158189627496e-09]
    state = _make_state(lib.math.sum.__pyne_layout__)
    ref = _ref_sum_factory()
    with _bars() as next_bar:
        for v in values:
            got = lib.math.sum(state, v, 3)
            want = ref(v, 3)
            if isinstance(want, NA) or want != want:
                assert isinstance(got, NA) or got != got
            else:
                assert got == want
            next_bar()


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


def __test_math_sum_length_one_na_bar_echoes_the_window__():
    """ length == 1 gives back the source, and an na bar echoes the last non-na

    MEASURED (probe sumlen4, BINANCE:BTCUSDT 30m, 28746 bars): a length-1 sum
    equals the source on every non-na bar, and on an na bar it reports the last
    NON-NA value rather than na — an na bar is not stored and does not advance
    the window, exactly like every other length.
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

    MEASURED LAW (probe sumlen2, BINANCE:BTCUSDT 30m): ``math.sum(volume,
    min(bar_index + 1, 610))`` equals ``math.sum(volume, 610)`` bit-for-bit on
    every one of the 28746 exported bars, and so does the shorter 5-length pair.
    TradingView does not re-baseline when the length moves — while the window is
    still filling there is nothing to evict, so the two runs are the same
    machine — hence the clamped-length idiom (``ta.sma(volume, math.min(
    bar_index + 1, len))``, Heatmap Volume [xdecow]) must not restart the
    accumulator.
    """
    length = 9
    # Mixed magnitudes so the compensated accumulator carries a live residue:
    # a re-baselined linear sum drifts off the running machine within a window.
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


def __test_math_sum_length_change_walks_one_entry_at_a_time__():
    """ A moved length enters and leaves the window ONE entry per machine step

    MEASURED LAW (probes sumlen6/sumlen7, BINANCE:BTCUSDT 30m): a 5->6, 6->7,
    4->10 or 6->5 length step reproduces TradingView bit-for-bit only when every
    admitted/evicted entry runs its own compensated round -- folding the whole
    change into a single fused ``d0`` lands 1-3 ulp away. The two models are
    compared here directly, so the test fails if the implementation folds.
    """
    length_before, length_after = 4, 8
    # Magnitudes chosen so the compensated residue is live when the length moves
    values = [1e8 + i * 0.5 if i % 2 else 7.5e-7 * (i % 5 + 1) for i in range(24)]
    change_at = 16

    def lengths(i):
        return length_before if i < change_at else length_after

    def model(sequential):
        summ = 0.0
        comp = 0.0
        ent = []            # realized entries, index 0 = newest
        win = 0
        got = []
        for i, v in enumerate(values):
            length = lengths(i)
            new_w = length if i + 1 >= length else i + 1
            d0 = 0.0
            admitted = []
            if win + 1 > new_w:
                for k in range(win, new_w, -1):
                    if sequential:
                        y1 = -ent[k - 1] - comp
                        t = summ + y1
                        y2 = -((t - summ) - y1)
                        summ = t + y2
                        comp = (summ - t) - y2
                    else:
                        d0 += ent[k - 1]
                d0 = d0 + ent[new_w - 1] if not sequential else ent[new_w - 1]
            elif new_w > win + 1:
                for k in range(new_w - 1, win, -1):
                    a = values[i - k]
                    if sequential:
                        y1 = -comp
                        t = summ + y1
                        y2 = a - ((t - summ) - y1)
                        summ = t + y2
                        comp = (summ - t) - y2
                        admitted.append((k, y2))
                    else:
                        d0 -= a
                        admitted.append((k, a))
            fires = False
            if comp != 0.0 and v != 0.0:
                b = comp if comp > 0.0 else -comp
                r = v + b
                if r != 0.0 and r - r == 0.0:
                    fires = b - (r - v) > 0.0
            if fires:
                # Re-baseline: newest-first linear sum of the raw window
                summ = v
                for k in range(1, new_w):
                    summ = values[i - k] + summ
                comp = 0.0
                ent = [v] + ent
            else:
                y1 = -d0 - comp
                t = summ + y1
                y2 = v - ((t - summ) - y1)
                ns = t + y2
                comp = (ns - t) - y2
                summ = ns
                ent = [y2] + ent
            for k, e in admitted:
                if k < len(ent):
                    ent[k] = e
            win = new_w
            got.append(summ if new_w >= length else None)
        return got

    walked = model(True)
    folded = model(False)
    state = _make_state(lib.math.sum.__pyne_layout__)
    with _bars() as next_bar:
        for i, v in enumerate(values):
            got = lib.math.sum(state, v, lengths(i))
            if walked[i] is not None:
                assert repr(got) == repr(walked[i]), f'bar {i}: {got!r} != {walked[i]!r}'
            next_bar()

    # The two models must actually disagree, or the assertion above proves nothing
    assert any(a is not None and repr(a) != repr(b) for a, b in zip(walked, folded))


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
