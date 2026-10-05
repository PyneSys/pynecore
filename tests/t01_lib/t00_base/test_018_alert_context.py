"""Alert output follows the caller's execution context, including library helpers."""
import sys

import pytest

from pynecore import lib
from pynecore.core import script as script_core
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.lib.alert import alert
from pynecore.providers.ccxt import CCXTProvider
from pynecore.types.ohlcv import OHLCV


@pytest.mark.parametrize('context', ['main', 'library', 'security'])
@pytest.mark.parametrize('fallback', [False, True])
def __test_alert_output_context__(monkeypatch, capsys, context, fallback):
    monkeypatch.setattr(lib, '_lib_semaphore', context != 'main')
    monkeypatch.setattr(lib, '_in_lib_main', context == 'library')
    monkeypatch.setattr(lib, '_in_security', context == 'security')
    if fallback:
        monkeypatch.setitem(sys.modules, 'typer', None)
    assert alert('context probe') is None
    output = capsys.readouterr()
    assert output.err == ''
    assert output.out == ('🚨 ALERT: context probe\n' if context == 'main' else '')


def _syminfo():
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(prefix='TEST', description='Test', ticker='TEST', currency='USD',
                   period='5', type='crypto', mintick=0.01, pricescale=100,
                   minmove=1, pointvalue=1, timezone='UTC', volumetype='base',
                   mincontract=0.0001, opening_hours=opening_hours,
                   session_starts=session_starts, session_ends=session_ends)


@pytest.mark.parametrize('mode', ['library', 'indicator', 'strategy'])
def __test_alert_library_runner__(tmp_path, capsys, mode):
    library = tmp_path / 'alert_context_lib.py'
    library.write_text('''"""@pyne"""
from pynecore.lib import script, alert
from pynecore.core.pine_export import Exported, export
notify = Exported()

@script.library("Alert context")
def main():
    def helper(message):
        alert(message)
    @export
    def notify(message):
        helper(message)
    notify("demo")
''')
    host = tmp_path / 'alert_context_host.py'
    declaration = ('strategy("Alert host", calc_on_order_fills=True)' if mode == 'strategy'
                   else 'indicator("Alert host")')
    host.write_text('''"""@pyne"""
from pynecore.lib import script, alert, strategy, bar_index
import alert_context_lib

@script.''' + declaration + '''
def main():
    alert_context_lib.notify("export")
    alert("host")
''' + ('''    if bar_index == 0:
        strategy.entry("entry", strategy.long)
''' if mode == 'strategy' else ''))
    saved = list(script_core._registered_libraries)
    script_core._registered_libraries.clear()
    bars = [OHLCV(timestamp=1_704_067_200_000 + i * 300_000, open=1, high=2,
                  low=0.5, close=1, volume=1) for i in range(3)]
    try:
        runner = ScriptRunner(library if mode == 'library' else host, iter(bars), _syminfo())
        list(runner.run_iter())
        output = capsys.readouterr().out
        if mode == 'library':
            assert output.count('ALERT: demo') == len(bars)
            assert 'ALERT: host' not in output
        else:
            assert 'ALERT: demo' not in output
            count = output.count('ALERT: host')
            assert count == output.count('ALERT: export')
            if mode == 'strategy':
                assert count > len(bars)
            else:
                assert count == len(bars)
        assert not lib._lib_semaphore
        assert not lib._in_lib_main
    finally:
        script_core._registered_libraries[:] = saved
        sys.modules.pop('alert_context_lib', None)
        sys.modules.pop('alert_context_host', None)
