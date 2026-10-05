"""Imported library examples allocate no drawings; main and export calls still draw."""
import sys

import pytest

from pynecore import lib
from pynecore.core import script as script_core
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.core.viz import drawings_snapshot, reset_state
from pynecore.lib import line, label, box, table, polyline, linefill, chart, position, color
from pynecore.lib.hline import hline
from pynecore.providers.ccxt import CCXTProvider
from pynecore.types.na import NA
from pynecore.types.ohlcv import OHLCV


@pytest.fixture
def clean_state(monkeypatch):
    monkeypatch.setattr(lib, '_in_lib_main', False)
    monkeypatch.setattr(lib, '_lib_semaphore', False)
    monkeypatch.setattr(lib, '_in_security', False)
    monkeypatch.setattr(lib, '_script', None)
    reset_state()
    yield
    reset_state()


def __test_suppressed_constructors_and_copies__(clean_state, monkeypatch):
    first = line.new(0, 1, 1, 2)
    second = line.new(0, 2, 1, 3)
    lab = label.new(0, 1, 'visible')
    rect = box.new(0, 2, 1, 1)
    tbl = table.new(position.top_right, 1, 1)
    table.cell(tbl, 0, 0, 'visible')
    fill = linefill.new(first, second, color.red)
    before = drawings_snapshot()
    metadata = dict(lib._plot_meta)
    seq = dict(lib._viz_seq)
    monkeypatch.setattr(lib, '_in_lib_main', True)

    def allocation_forbidden():
        pytest.fail('A suppressed drawing allocated a visual ID')

    for module in (line, label, box, table, polyline, linefill):
        monkeypatch.setattr(module, 'next_vid', allocation_forbidden)
    handles = [line.new(0, 1, 1, 2), label.new(0, 1), box.new(0, 2, 1, 1),
               table.new(position.top_right, 1, 1),
               polyline.new([chart.point.from_index(0, 1), chart.point.from_index(1, 2)]),
               linefill.new(first, second, color.blue), hline(10),
               line.copy(first), label.copy(lab), box.copy(rect)]
    assert all(isinstance(handle, NA) for handle in handles)
    line.set_y1(handles[0], 3)
    label.set_text(handles[1], 'hidden')
    box.set_top(handles[2], 9)
    table.cell(handles[3], 0, 0, 'hidden')
    table.set_position(handles[3], position.top_right)
    linefill.set_color(handles[5], color.blue)
    for module, handle in zip((line, label, box, table, polyline, linefill), handles):
        module.delete(handle)
    assert label.get_x(handles[1]) != label.get_x(handles[1])
    assert drawings_snapshot() == before
    assert fill in linefill._registry
    assert lib._plot_meta == metadata and lib._viz_seq == seq


def __test_security_semaphore_does_not_discard_computation_objects__(clean_state, monkeypatch):
    monkeypatch.setattr(lib, '_in_security', True)
    monkeypatch.setattr(lib, '_lib_semaphore', True)
    handle = line.new(0, 2, 1, 4)
    assert line.get_y2(handle) == 4


def _syminfo():
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(prefix='TEST', description='Test', ticker='TEST', currency='USD',
                   period='5', type='crypto', mintick=0.01, pricescale=100,
                   minmove=1, pointvalue=1, timezone='UTC', volumetype='base',
                   mincontract=0.0001, opening_hours=opening_hours,
                   session_starts=session_starts, session_ends=session_ends)


@pytest.mark.parametrize('standalone', [False, True])
def __test_library_main_and_export_drawing_context__(tmp_path, clean_state, standalone):
    library = tmp_path / 'drawing_context_lib.py'
    library.write_text('''"""@pyne"""
from pynecore.lib import script, label, table, position, bar_index, high
from pynecore.types import Persistent, Label, Table
from pynecore.core.pine_export import Exported, export
make_label = Exported()

@script.library("Drawing context")
def main():
    @export
    def make_label():
        return label.new(bar_index, high, "export")
    demo: Persistent[Label] = make_label()
    label.set_text(demo, "demo")
    tbl: Persistent[Table] = table.new(position.top_right, 1, 1)
    table.cell(tbl, 0, 0, "demo")
''')
    host = tmp_path / 'drawing_context_host.py'
    host.write_text('''"""@pyne"""
from pynecore.lib import script, table, position
from pynecore.types import Persistent, Label, Table
import drawing_context_lib

@script.indicator("Host")
def main():
    tbl: Persistent[Table] = table.new(position.top_right, 1, 1)
    table.cell(tbl, 0, 0, "host")
    own: Persistent[Label] = drawing_context_lib.make_label()
''')
    saved = list(script_core._registered_libraries)
    script_core._registered_libraries.clear()
    bars = [OHLCV(timestamp=1_704_067_200_000 + i * 300_000, open=1, high=2,
                  low=0, close=1, volume=1) for i in range(3)]
    try:
        runner = ScriptRunner(library if standalone else host, iter(bars), _syminfo())
        list(runner.run_iter())
        snapshot = runner.drawings()
        assert len(snapshot['labels']) == 1
        assert snapshot['labels'][0]['text'] == ('demo' if standalone else 'export')
        assert len(snapshot['tables']) == 1
        assert snapshot['tables'][0]['cells'][0]['text'] == ('demo' if standalone else 'host')
        assert not lib._in_lib_main
    finally:
        script_core._registered_libraries[:] = saved
        sys.modules.pop('drawing_context_lib', None)
        sys.modules.pop('drawing_context_host', None)
