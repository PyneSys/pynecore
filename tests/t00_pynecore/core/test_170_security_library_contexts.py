"""Security contexts of an imported library live in the library's own module.

The transformer numbers a context's clone (``__sec_main_<n>__``) and its merge
``group`` per module, while the runner merges the contexts of the script and of
every library it imports into one map. Both have to stay tied to the module that
declared the context.
"""
import sys
from dataclasses import replace

import pytest

from pynecore.core import script as script_mod
from pynecore.core.ohlcv import OHLCVWriter
from pynecore.core.script_runner import ScriptRunner, import_script
from pynecore.core.security_process import _child_entries
from pynecore.types.ohlcv import OHLCV

LIBRARY = '''"""@pyne"""
from pynecore.lib import bar_index, close, request, script, syminfo, ta

@script.library('T170 Library')
def main():
    x = ta.sma(close, 2)
    if bar_index > 1:
        a = request.security(syminfo.tickerid, '60', x)
        b = request.security(syminfo.tickerid, '60', x + 1)
'''

SCRIPT = '''"""@pyne"""
{imports}
from pynecore.lib import bar_index, close, plot, request, script, syminfo, ta

@script.indicator('T170 Script')
def main():
    x = ta.ema(close, 3)
    a = 0.0
    b = 0.0
    if bar_index > 1:
        a = request.security(syminfo.tickerid, '240', x)
        b = request.security(syminfo.tickerid, '240', x * 2)
    plot(a + b, 'ab')
'''

START = 1_735_689_600_000


@pytest.fixture(autouse=True)
def __test_helper_clean_state():
    libraries = list(script_mod._registered_libraries)
    modules = set(sys.modules)
    yield
    script_mod._registered_libraries[:] = libraries
    for name in set(sys.modules) - modules:
        if name.startswith('t170_'):
            sys.modules.pop(name, None)


def __test_helper_write(tmp_path, monkeypatch, *, with_library: bool):
    (tmp_path / 't170_lib.py').write_text(LIBRARY, encoding='utf-8')
    path = tmp_path / ('t170_with_lib.py' if with_library else 't170_alone.py')
    path.write_text(SCRIPT.format(imports='import t170_lib' if with_library else ''),
                    encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    return path


def __test_helper_contexts(module) -> dict[str, dict]:
    return module.__security_contexts__


def __test_clone_runs_in_place_of_its_own_modules_main__(tmp_path, monkeypatch):
    """ A library context's clone replaces the library's main, never the script's """
    path = __test_helper_write(tmp_path, monkeypatch, with_library=True)
    module = import_script(path)
    library = sys.modules['t170_lib']
    script_sid, script_ctx = next(iter(__test_helper_contexts(module).items()))
    lib_sid, lib_ctx = next(iter(__test_helper_contexts(library).items()))
    # The precondition of the bug: both modules number their first clone 0
    assert script_ctx['slice_main'] == lib_ctx['slice_main']
    clone_name = lib_ctx['slice_main']
    libraries = script_mod.imported_libraries(module)
    sec_modules = [module, library]

    entry, lib_entries = _child_entries(module, libraries, sec_modules, lib_sid, clone_name)
    assert entry is module.main
    assert lib_entries == [getattr(library, clone_name)]

    entry, lib_entries = _child_entries(module, libraries, sec_modules, script_sid, clone_name)
    assert entry is getattr(module, clone_name)
    assert lib_entries == [library.main]


def __test_same_ordinal_groups_of_two_modules_stay_apart__(tmp_path, monkeypatch, syminfo):
    """ Group 0 of the script and group 0 of a library are different feeds """
    feeds = {}
    for timeframe, minutes in (('60', 60), ('240', 240)):
        feed = tmp_path / f'feed_{timeframe}.ohlcv'
        with OHLCVWriter(feed, timeframe) as writer:
            for index in range(8):
                price = 10.0 + index
                writer.write(OHLCV(timestamp=START + index * minutes * 60_000, open=price,
                                   high=price + 1, low=price - 1, close=price, volume=1.0))
        replace(syminfo, period=timeframe).save_toml(feed.with_suffix('.toml'))
        feeds[timeframe] = feed
    chart = replace(syminfo, period='5')
    bars = [OHLCV(timestamp=START + index * 300_000, open=10.0 + index, high=11.0 + index,
                  low=9.0 + index, close=10.0 + index, volume=1.0) for index in range(240)]

    results = {}
    for with_library in (True, False):
        path = __test_helper_write(tmp_path, monkeypatch, with_library=with_library)
        if with_library:
            module = import_script(path)
            library = sys.modules['t170_lib']
            groups = {ctx['group'] for ctx in __test_helper_contexts(module).values()}
            assert groups == {ctx['group'] for ctx in __test_helper_contexts(library).values()}
            assert groups != {None}
        rows = list(ScriptRunner(path, bars, chart, security_data=feeds).run_iter())
        results[with_library] = [repr(row[1]['ab']) for row in rows]

    assert results[True] == results[False]
    assert set(results[True]) - {'0.0', 'nan'}, "the security values must reach the chart"
