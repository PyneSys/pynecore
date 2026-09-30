"""
The compiled ``math.sum`` machine returns the very doubles the Python one does, and
both roll back exactly.

``core/_native_rolling_sum`` is a compiled copy of ``core.rolling_sum.SumMachine``, and
it only earns its place if it is bit-identical to it on every call: the Python class is
what the venue tests pin, and it stays the implementation wherever the extension is not
built. The comparison drives both machines through the same calls -- na sources, moving
lengths, a full window wrapping around its history, same-bar re-executions from a
restored bar-start snapshot, and baselines restored across several bars -- and compares
every returned double.

The wheel build runs this test on every platform it produces a wheel for, with
``PYNE_REQUIRE_NATIVE_MATH=1`` so that a wheel whose extension silently failed to build
fails instead of skipping.
"""
import os
import random
import struct
from contextlib import contextmanager

import pytest

from pynecore import lib
from pynecore.core import instance_state, rolling_sum
from pynecore.core.series import SeriesImpl
from pynecore.types.na import NA

_MACHINES = (rolling_sum.PYTHON_SUM_MACHINE, rolling_sum.SumMachine)


@contextmanager
def __test_helper_bars():
    """Drive ``lib.bar_index`` manually (the window writes are bar-keyed)."""
    saved = lib.bar_index
    lib.bar_index = 0
    try:
        yield
    finally:
        lib.bar_index = saved


def __test_helper_bits(x) -> object:
    return 'na' if isinstance(x, NA) else struct.pack('<d', x)


def __test_helper_value(rnd: random.Random, scale: float):
    pick = rnd.random()
    if pick < 0.07:
        return float('nan')
    if pick < 0.1:
        return NA(float)
    if pick < 0.13:
        return 0.0
    if pick < 0.16:
        return rnd.randint(-5, 5)
    if pick < 0.5:
        return rnd.uniform(-1.0, 1.0) * scale
    return rnd.choice((1e8, -3e7, 1e-7, 3.3, 1e12)) * rnd.random()


def __test_helper_drive(machine_cls, seed: int, bars: int, max_length: int) -> list:
    """Every result of one seeded run: a bar is one to four executions from the
    bar-start snapshot, the way a shared loop call site runs its machine."""
    rnd = random.Random(seed)
    machine = machine_cls()
    base = rnd.randint(1, max_length)
    out = []
    with __test_helper_bars():
        for bar in range(bars):
            lib.bar_index = bar
            token = machine.__pyne_snapshot__()
            moving = rnd.random()
            for k in range(rnd.randint(1, 4) if rnd.random() < 0.4 else 1):
                if k:
                    machine.__pyne_restore__(token)
                if moving < 0.6:
                    length = base
                elif moving < 0.85:
                    length = rnd.randint(1, max_length)
                else:
                    length = min(max_length, max(1, base + rnd.randint(-3, 3)))
                if rnd.random() < 0.05:
                    length = length + 0.7
                out.append(__test_helper_bits(
                    machine.step(__test_helper_value(rnd, 100.0 * (seed % 5 + 1)), length)))
    return out


def __test_native_sum_machine_matches_python__():
    """The compiled machine agrees with the Python one bit for bit"""
    native, python = rolling_sum.SumMachine, rolling_sum.PYTHON_SUM_MACHINE
    if native is python:
        if os.environ.get('PYNE_REQUIRE_NATIVE_MATH'):
            pytest.fail("the native rolling-sum extension is not installed")
        pytest.skip("native rolling-sum extension not built")
    assert rolling_sum.rolling_sum_step is native.step
    runs = [(seed, 400, 12) for seed in range(40)]
    # A window past the 5001-value history wraps; lengths near the cap walk deep
    runs += [(100, 11000, 5), (101, 11000, 5000), (102, 3000, 400)]
    for seed, bars, max_length in runs:
        assert (__test_helper_drive(native, seed, bars, max_length)
                == __test_helper_drive(python, seed, bars, max_length)), (seed, bars, max_length)


@pytest.mark.parametrize('machine_cls', _MACHINES, ids=('python', 'native'))
def __test_sum_machine_restore_replays_the_bar__(machine_cls):
    """A restored machine computes what a machine that never took the discarded
    step computes, whatever that step was"""
    values = [1e8 + i * 0.1 if i % 3 else 3.25e-5 * (i % 7 + 1) for i in range(300)]
    with __test_helper_bars():
        clean = machine_cls()
        rolled = machine_cls()
        for bar, value in enumerate(values):
            lib.bar_index = bar
            token = rolled.__pyne_snapshot__()
            rolled.step(-value if bar % 2 else float('nan'), 3 + bar % 11)
            rolled.__pyne_restore__(token)
            length = 5 if bar < 150 else 9
            assert __test_helper_bits(rolled.step(value, length)) == \
                __test_helper_bits(clean.step(value, length)), bar


@pytest.mark.parametrize('history', (5000, 30))
@pytest.mark.parametrize('machine_cls', _MACHINES, ids=('python', 'native'))
def __test_sum_machine_restore_undoes_every_step_since__(machine_cls, history, monkeypatch):
    """A baseline is restored across any number of steps on later bars, with
    bar-start snapshots of those bars taken and restored in between -- the way a
    live lower-timeframe baseline is rolled back over a provisional chain. The
    discarded steps grow the ring, widen the history while the window still fills,
    and overwrite a full window's oldest values -- from the start with a short
    default history"""
    monkeypatch.setattr(SeriesImpl, 'DEFAULT_MAX_BARS_BACK', history)
    rnd = random.Random(7)
    with __test_helper_bars():
        clean = machine_cls()
        rolled = machine_cls()
        for bar in range(6200):
            lib.bar_index = bar
            baseline = rolled.__pyne_snapshot__()
            for ahead in range(rnd.randint(0, 5) if bar % 3 else 0):
                lib.bar_index = bar + ahead
                inner = rolled.__pyne_snapshot__()
                pick = rnd.random()
                length = 200 if pick < 0.2 and bar < 20 else rnd.randint(1, 30)
                rolled.step(float('nan') if pick > 0.9 else rnd.uniform(-1e6, 1e6), length)
                if pick > 0.5:
                    rolled.__pyne_restore__(inner)
                    rolled.step(rnd.uniform(-1.0, 1.0), rnd.randint(1, 30))
            rolled.__pyne_restore__(baseline)
            lib.bar_index = bar
            value = 1e8 + bar * 0.1 if bar % 5 else 3.25e-5 * (bar % 7 + 1)
            length = 7 if bar < 3000 else 12
            assert __test_helper_bits(rolled.step(value, length)) == \
                __test_helper_bits(clean.step(value, length)), bar


@pytest.mark.parametrize('machine_cls', _MACHINES, ids=('python', 'native'))
def __test_sum_machine_walks_a_shrink_again_over_replayed_history__(machine_cls):
    """A shrinking window is walked over the history the machine holds: bars
    replayed behind a restored baseline with other values can reach the shrink with
    the very count, window, sum and compensation the discarded bars reached it with"""
    with __test_helper_bars():
        machine = machine_cls()
        for bar, value in enumerate((1.0, 2.0, 3.0)):
            lib.bar_index = bar
            machine.step(value, 3)
        baseline = machine.__pyne_snapshot__()
        for first, second in ((4.0, 5.0), (5.0, 4.0), (4.5, 4.5)):
            machine.__pyne_restore__(baseline)
            lib.bar_index = 3
            machine.step(first, 3)
            lib.bar_index = 4
            assert machine.step(second, 3) == 12.0
            lib.bar_index = 5
            assert machine.step(6.0, 1) == 6.0, (first, second)


@pytest.mark.parametrize('machine_cls', _MACHINES, ids=('python', 'native'))
def __test_sum_machine_refuses_a_token_it_was_rolled_back_past__(machine_cls):
    """A rollback cannot be redone: a token taken after the restored one names a
    state that is gone, and so does a token of another machine"""
    with __test_helper_bars():
        machine = machine_cls()
        machine.step(1.0, 2)
        older = machine.__pyne_snapshot__()
        lib.bar_index = 1
        machine.step(2.0, 2)
        newer = machine.__pyne_snapshot__()
        lib.bar_index = 2
        machine.step(3.0, 2)
        machine.__pyne_restore__(older)
        with pytest.raises(RuntimeError):
            machine.__pyne_restore__(newer)
        other = machine_cls()
        other.step(1.0, 2)
        with pytest.raises(RuntimeError):
            machine.__pyne_restore__(other.__pyne_snapshot__())


def __test_math_sum_rolls_back_through_instance_state__():
    """The loop-site rollback and the child-subtree rollback both hand the
    machine back to its bar-start state"""
    layout = getattr(lib.math.sum, '__pyne_layout__')
    values = [1.1, 2.2, NA(float), 3.3, 0.1, 1e12, 4.4, 1e-9, 5.5, 0.3333333333, 7.7, 8.8]
    with __test_helper_bars():
        clean = instance_state._make_state(layout)
        looped = instance_state._make_state(layout)
        ticked = instance_state._make_state(layout)
        for bar, value in enumerate(values):
            lib.bar_index = bar
            want = __test_helper_bits(lib.math.sum(clean, value, 4))

            stamped = instance_state._stamp(instance_state._snap_collected([(looped, layout)]))
            lib.math.sum(looped, 1e6, 2)
            instance_state._restore_stamped(stamped)
            assert __test_helper_bits(lib.math.sum(looped, value, 4)) == want, bar

            snap = instance_state._snap_vector(ticked, layout, {})
            lib.math.sum(ticked, -1e6, 7)
            instance_state._restore_vector(ticked, snap, {})
            assert __test_helper_bits(lib.math.sum(ticked, value, 4)) == want, bar
