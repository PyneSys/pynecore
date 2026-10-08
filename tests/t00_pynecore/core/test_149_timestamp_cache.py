"""Timestamp result reuse preserves overload, conversion and context semantics."""
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import importlib

import pytest

import pynecore.lib as lib
from pynecore.core import datetime as core_datetime
from pynecore.core.overload import Implementation, _anchored
from pynecore.core.script_runner import ScriptRunner
from pynecore.types.ohlcv import OHLCV


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    monkeypatch.setattr(lib.syminfo, 'timezone', 'UTC')
    lib._timestamp_components_cached.cache_clear()
    yield
    lib._timestamp_components_cached.cache_clear()


def __test_loaded_map_namespace_does_not_shadow_builtin__():
    namespace = importlib.import_module('pynecore.lib.map')
    assert lib.map is namespace
    bound = lib.timestamp.__pyne_bind__('iii')
    result = bound(2025, 1, 1)
    before = lib._timestamp_components_cached.cache_info()
    assert result == 1735689600000
    assert bound(2025, 1, 1) == result
    assert lib._timestamp_components_cached.cache_info() == before


@pytest.mark.parametrize('pin,args', [
    ('iii', (2025, 1, 1)),
    ('iiii', (2025, 1, 1, 2)),
    ('iiiii', (2025, 1, 1, 2, 30)),
    ('iiiiii', (2025, 1, 1, 2, 30, 40)),
    ('siii', ('UTC', 2025, 1, 1)),
    ('siiii', ('UTC', 2025, 1, 1, 2)),
    ('siiiii', ('UTC', 2025, 1, 1, 2, 30)),
    ('siiiiii', ('UTC', 2025, 1, 1, 2, 30, 40)),
    ('fff', (2025.9, 1.9, 1.9)),
])
def __test_each_component_shape_keeps_its_last_result__(pin, args):
    bound = lib.timestamp.__pyne_bind__(pin)
    first = bound(*args)
    before = lib._timestamp_components_cached.cache_info()
    assert bound(*args) == first
    assert lib._timestamp_components_cached.cache_info() == before
    assert before.misses == 1
    other_site = lib.timestamp.__pyne_bind__(pin)
    assert other_site(*args) == first
    assert lib._timestamp_components_cached.cache_info().hits == before.hits + 1


def __test_input_edits_and_restoration_use_the_shared_lru__():
    bound = lib.timestamp.__pyne_bind__('iii')
    original = bound(2025, 1, 1)
    assert bound(2025, 1, 2) == original + 86400000
    assert bound(2025, 1, 1) == original
    stats = lib._timestamp_components_cached.cache_info()
    assert (stats.misses, stats.hits) == (2, 1)


@pytest.mark.parametrize('pin,args', [
    ('iii', (2025, 7, 1)),
    ('siii', ('', 2025, 7, 1)),
])
def __test_exchange_timezone_is_read_on_every_call__(monkeypatch, pin, args):
    bound = lib.timestamp.__pyne_bind__(pin)
    utc = bound(*args)
    monkeypatch.setattr(lib.syminfo, 'timezone', 'America/New_York')
    assert bound(*args) == utc + 4 * 3600000
    monkeypatch.setattr(lib.syminfo, 'timezone', 'UTC')
    assert bound(*args) == utc
    assert lib._timestamp_components_cached.cache_info().hits == 1


def __test_timezone_resolution_invalidation_is_observed__(monkeypatch):
    bound = lib.timestamp.__pyne_bind__('iii')
    utc = bound(2025, 7, 1)
    core_datetime._parse_timezone_cached.cache_clear()
    monkeypatch.setattr(core_datetime, 'ZoneInfo', lambda name: timezone(timedelta(hours=3)))
    try:
        assert bound(2025, 7, 1) == utc - 3 * 3600000
    finally:
        core_datetime._parse_timezone_cached.cache_clear()


def __test_unknown_timezone_does_not_reuse_a_previous_result__(monkeypatch):
    bound = lib.timestamp.__pyne_bind__('iii')
    bound(2025, 1, 1)
    monkeypatch.setattr(lib.syminfo, 'timezone', 'Not/AZone')
    with pytest.raises(core_datetime.TimezoneNotFoundError, match='Unknown timezone'):
        bound(2025, 1, 1)


def __test_equivalent_components_have_one_shared_entry__():
    first = lib.timestamp('UTC', 2025.9, 1.9, 1.9, float('nan'), float('inf'), 0)
    assert lib.timestamp(year=2025, month=1, day=1) == first
    stats = lib._timestamp_components_cached.cache_info()
    assert (stats.misses, stats.hits) == (1, 1)


def __test_keyword_and_arity_changes_keep_dispatch_semantics__():
    bound = lib.timestamp.__pyne_bind__('iii')
    result = bound(2025, 1, 1)
    assert bound(year=2025, month=1, day=1) == result
    assert bound('UTC', 2025, 1, 1) == result
    assert bound(2025, 1, 1, 2) == result + 7200000
    with pytest.raises(TypeError, match='No matching implementation'):
        bound(year=2025)


def __test_replaced_implementation_invalidates_the_front_cache__(monkeypatch):
    implementation = next(impl for impl in lib.timestamp.__pyne_impls__
                          if tuple(impl.sig.parameters)[0] == 'year')
    bound = lib.timestamp.__pyne_bind__('iii')
    bound(2025, 1, 1)
    monkeypatch.setattr(implementation, 'func', lambda *args: 42.0)
    assert bound(2025, 1, 1) == 42.0


def __test_changed_overload_group_uses_the_fallback__():
    implementation = next(impl for impl in lib.timestamp.__pyne_impls__
                          if tuple(impl.sig.parameters)[0] == 'year')
    group = [implementation]
    calls = []

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return 42.0

    fallback.__pyne_cache__ = {}
    bound = lib._bind_timestamp_components(fallback, implementation, group, 3,
                                           with_timezone=False)
    bound(2025, 1, 1)
    group.append(implementation)
    assert bound(2025, 1, 1) == 42.0
    assert len(calls) == 1


def __test_disabling_pins_keeps_value_driven_dispatch__(monkeypatch):
    monkeypatch.setenv('PYNE_NO_TYPE_PIN', '1')
    bound = lib.timestamp.__pyne_bind__('iii')
    assert bound.__name__ == 'dispatch'
    assert bound(2025, 1, 1) == 1735689600000


def __test_wildcard_pins_keep_ordinary_verification__():
    bound = lib.timestamp.__pyne_bind__('ii*')
    assert bound.__name__ != 'cached'
    assert bound(2025, 1, 1) == 1735689600000


def __test_mutable_numeric_subclasses_are_converted_on_every_call__():
    class ChangingNumber(float):
        __hash__ = None

        def __int__(self):
            return self.current

    value = ChangingNumber(1)
    value.current = 1
    bound = lib.timestamp.__pyne_bind__('iii')
    first = bound(2025, 1, 1)
    assert bound(2025, 1, value) == first
    value.current = 2
    assert bound(2025, 1, value) == first + 86400000
    value.current = 3
    assert lib.timestamp(2025, 1, value) == first + 2 * 86400000


def __test_empty_date_strings_remain_wall_clock_reads__(monkeypatch):
    dates = iter((datetime(2025, 1, 1, tzinfo=timezone.utc),
                  datetime(2025, 1, 2, tzinfo=timezone.utc)))
    monkeypatch.setattr(lib, '_parse_datestring', lambda value: next(dates))
    bound = lib.timestamp.__pyne_bind__('s')
    assert bound('') == 1735689600000
    assert bound('') == 1735776000000
    assert lib._timestamp_components_cached.cache_info().currsize == 0


def __test_shared_cache_is_bounded__():
    for minute in range(1100):
        lib.timestamp('UTC', 2025, 1, 1, 0, minute)
    assert lib._timestamp_components_cached.cache_info().currsize == 1024


def __test_runner_resets_inputs_and_context_do_not_leak__(tmp_path, monkeypatch, syminfo):
    monkeypatch.setenv('PYNE_SAVE_SCRIPT_TOML', '0')
    path = tmp_path / 'timestamp_probe.py'
    path.write_text('''"""@pyne"""
from pynecore.lib import script, input, timestamp, bar_index
@script.indicator("Timestamp context probe")
def main(day=input.int(1)):
    fixed = timestamp(2025, 7, day, 0, 0)
    varying = timestamp("UTC", 2025, 7, day, 0, bar_index)
    return {"fixed": fixed, "varying": varying}
''')
    bars = [OHLCV(1735689600000 + i * 300000, 10, 12, 9, 11, 100) for i in range(8)]

    def run(zone, day, switch=False):
        runner = ScriptRunner(path, bars, replace(syminfo, timezone=zone), inputs={'day': day})
        values = []
        for index, (_, output) in enumerate(runner.run_iter()):
            expected_zone = 'America/New_York' if switch and index > 2 else zone
            assert output['fixed'] == lib.timestamp(expected_zone, 2025, 7, day)
            assert output['varying'] == lib.timestamp('UTC', 2025, 7, day, 0, index)
            values.append(dict(output))
            if switch and index == 2:
                lib.syminfo.timezone = 'America/New_York'
        return values

    original = run('UTC', 1)
    assert run('America/New_York', 1) != original
    assert run('UTC', 2) != original
    assert run('UTC', 1) == original
    assert run('UTC', 1, True)[3]['fixed'] != original[3]['fixed']


def __test_stateful_implementations_cannot_take_the_stateless_hook__():
    def function(__state__, value: int):
        __state__[0] += value
        return __state__[0]

    function.__pyne_layout__ = {'init': (0,), 'series': (), 'varip': (), 'children': ()}

    def forbidden_factory(*args):
        raise AssertionError('A stateful implementation must retain ordinary binding')

    function.__pyne_pinned_bind__ = forbidden_factory
    bound = _anchored([Implementation(function)], 'stateful_probe', pin='i')
    assert bound(1) == 1
    assert bound(2) == 3
