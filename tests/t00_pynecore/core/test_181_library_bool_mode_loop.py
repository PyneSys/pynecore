"""
A library function called in a loop keeps one ``ta.*`` step per bar when the
caller's bool mode differs.

A v5 script that reads a bool with history runs in the three-state bool mode
(``na_bool=True``); a library that does not runs in its own mode. The runtime
crosses into the library through ``pine_export.in_module_bool_mode``, which wraps
the library function's binding in a closure. The wrapper publishes the binding as
``__pyne_bound__``, and the loop site's same-bar rollback (``__bind_loop__`` ->
``_collect_bound_builtins``) and the per-bar redefinition (``_carry_state``) look
through it. An opaque wrapper hid the builtin machines: a ``ta.sma`` / ``ta.stdev``
reached on every pass of a loop advanced once per pass instead of once per bar, so
``ta.sma(close, 20)`` had a value on bar 2 and the library's ``multibands`` could
fail in ``_math_stateful.sum`` with ``IndexError: Slice stop index out of range!``.
"""
import math
import sys
from functools import partial
from pathlib import Path

from pynecore.core import instance_state
from pynecore.core.pine_export import in_module_bool_mode
from pynecore.types import na as na_types
from pynecore.types.na import set_bool_na

DATA_DIR = Path(__file__).parent / 'data'
N_BARS = 300
LENGTH = 20


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='5', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _make_ohlcv(num_bars: int):
    """Bars whose close varies, so ``ta.stdev`` is not trivially zero."""
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(num_bars):
        close = 100.0 + 10.0 * math.sin(i / 7.0) + (i % 5)
        bars.append(OHLCV(timestamp=1_704_067_200_000 + i * 300_000,
                          open=close - 0.5 if i % 2 else close + 0.5,
                          high=close + 1.0, low=close - 1.0, close=close, volume=1000.0))
    return bars


def _run(script_name: str, modules: tuple[str, ...]) -> list[dict]:
    """Run a script of ``DATA_DIR`` over the bars; ``modules`` are unloaded after."""
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    saved_libraries = list(script_core._registered_libraries)
    sys.path.insert(0, str(DATA_DIR))
    try:
        runner = ScriptRunner(DATA_DIR / f'{script_name}.py', iter(_make_ohlcv(N_BARS)),
                              _make_syminfo())
        return [dict(plot_data) for _candle, plot_data in runner.run_iter()]
    finally:
        sys.path.remove(str(DATA_DIR))
        script_core._registered_libraries[:] = saved_libraries
        for module in modules:
            sys.modules.pop(module, None)
        set_bool_na(False)


def _value(v):
    """A plot value with ``na`` as None, so bars compare."""
    from pynecore.types.na import isna_num
    return None if isna_num(v) else v


def __test_a_library_loop_matches_the_inlined_script_across_bool_modes__():
    """The importer (three-state bool) calling the library's looped ``bb`` through
    the bool-mode wrapper computes what the same functions inlined compute, on
    every one of 300 bars; ``ta.sma(close, 20)`` is na until bar 19.

    Without the wrapper publishing its binding, the loop's seven passes each
    advanced ``ta.sma`` / ``ta.stdev``: ``b6`` had a value on bar 2 and differed
    from the inlined script on 248 of the 300 bars.
    """
    importer = _run('loop_lib_importer', ('loop_lib_importer', 'loop_lib'))
    inlined = _run('loop_lib_inlined', ('loop_lib_inlined',))
    assert len(importer) == len(inlined) == N_BARS

    b6_importer = [_value(row['b6']) for row in importer]
    b6_inlined = [_value(row['b6']) for row in inlined]
    # One ta step per bar: na until the window of 20 closes is full
    assert all(v is None for v in b6_importer[:LENGTH - 1]), b6_importer[:LENGTH]
    assert all(v is not None for v in b6_importer[LENGTH - 1:])
    differing = [i for i, (a, b) in enumerate(zip(b6_importer, b6_inlined)) if a != b]
    assert not differing, f"b6 differs from the inlined script on {len(differing)} bars: " \
                          f"{differing[:10]} ({b6_importer[differing[0]]} != {b6_inlined[differing[0]]})"
    # The importer's own three-state bool is untouched by the crossing
    assert [row['c'] for row in importer] == [row['c'] for row in inlined]


def __test_the_bool_mode_wrapper_publishes_its_binding__(monkeypatch):
    """``in_module_bool_mode`` keeps the binding reachable as ``__pyne_bound__``,
    and the state walkers see the partial and its vector through it."""
    monkeypatch.setattr(na_types, '_bool_na_seen', True)
    monkeypatch.setattr(na_types, '_bool_na', True)

    layout = {'init': (0.0,), 'series': (), 'varip': (), 'children': ()}

    def accumulate(__state__, value: float):
        __state__[0] += value
        return __state__[0]
    accumulate.__pyne_layout__ = layout

    state = instance_state._make_state(layout)
    bound = partial(accumulate, state)
    wrapper = in_module_bool_mode(bound, False)
    assert wrapper is not bound, "the modes differ, so a wrapper is made"
    assert getattr(wrapper, '__pyne_bound__') is bound
    assert wrapper(2.0) == 2.0 and state[0] == 2.0

    # A per-bar redefinition carries the vector over through the wrapper
    assert instance_state._carry_state((accumulate, wrapper), layout) is state
    # A different callee (another layout) still gets fresh state
    other = {'init': (0.0,), 'series': (), 'varip': (), 'children': ()}
    assert instance_state._carry_state((accumulate, wrapper), other) is not state

    # The loop-site collector finds the same machines behind the wrapper as
    # behind the bare partial
    through_wrapper: list = []
    instance_state._collect_bound_builtins(wrapper, through_wrapper)
    bare: list = []
    instance_state._collect_bound_builtins(bound, bare)
    assert through_wrapper == bare


def __test_an_opaque_wrapper_advances_the_loop_machines_per_pass__(monkeypatch):
    """The symptom the published binding removes: with the wrapper made opaque
    again, the loop site's rollback finds no machines behind it and every pass of
    the seven advances ``ta.sma`` / ``ta.stdev``. Counting seven calls per bar,
    ``ta.stdev``'s window re-baseline fails with ``IndexError: Slice stop index
    out of range!`` in ``_math_stateful.sum`` -- or, where it survives, ``b6``
    already has a value on bar 2 and differs from the inlined script. This pins
    that the importer run really reaches the library through the wrapper."""
    from pynecore.core import instance_state as instance_state_module
    from pynecore.core import pine_export, pine_method

    real = pine_export.in_module_bool_mode

    def opaque(bound, na_bool):
        call = real(bound, na_bool)
        if call is not bound and hasattr(call, '__pyne_bound__'):
            delattr(call, '__pyne_bound__')
        return call

    for module in (pine_export, instance_state_module, pine_method):
        monkeypatch.setattr(module, 'in_module_bool_mode', opaque, raising=False)

    try:
        importer = _run('loop_lib_importer', ('loop_lib_importer', 'loop_lib'))
    except IndexError as exc:
        assert 'Slice stop index out of range' in str(exc), exc
        return
    inlined = _run('loop_lib_inlined', ('loop_lib_inlined',))
    b6_importer = [_value(row['b6']) for row in importer]
    b6_inlined = [_value(row['b6']) for row in inlined]
    assert b6_importer[2] is not None, "seven passes per bar fill the 20-bar window by bar 2"
    differing = [i for i, (a, b) in enumerate(zip(b6_importer, b6_inlined)) if a != b]
    assert len(differing) > N_BARS // 2, f"only {len(differing)} bars differ"

