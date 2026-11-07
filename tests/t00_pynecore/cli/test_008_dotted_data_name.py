"""
Data names that contain a dot.

A data file is named after its symbol, and exchange symbols often contain a dot
(a perpetual like ``BTCUSD.P``). In ``bybit_BTCUSD.P_1D`` the ``.P_1D`` tail is part
of the name, so a bare name must still resolve to ``<name>.ohlcv`` in the data dir.
"""
import sys
from datetime import time as dt_time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pynecore.cli import app
from pynecore.cli.app import app_state
from pynecore.core.data_converter import DataConverter
from pynecore.core.data_path import data_extension, with_data_extension
from pynecore.core.ohlcv import OHLCVWriter
from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from pynecore.types.ohlcv import OHLCV

__test_helper_name = 'bybit_BTCUSD.P_1D'

__test_helper_script = '''"""
@pyne
"""
from pynecore.lib import script, close, plot


@script.indicator("Dotted data name")
def main():
    plot(close, "close")
'''


@pytest.fixture
def __test_helper_workdir__(tmp_path, monkeypatch) -> Path:
    """A workdir whose data dir holds ``bybit_BTCUSD.P_1D.ohlcv`` and its ``.toml``.

    The CLI callback re-points the global ``app_state`` and installs an exception
    hook; both are restored after the test.
    """
    monkeypatch.setattr(app_state, '_workdir', app_state._workdir)
    monkeypatch.setattr(sys, 'excepthook', sys.excepthook)
    monkeypatch.delenv('PYNE_DATA_DIR', raising=False)
    monkeypatch.setenv('PYNE_SAVE_SCRIPT_TOML', '0')

    workdir = tmp_path / 'workdir'
    data_dir = workdir / 'data'
    data_dir.mkdir(parents=True)
    (workdir / '.pyne').write_text('workdir_version = "1.0"\n', encoding='utf-8')

    days = range(7)
    syminfo = SymInfo(
        prefix='BYBIT', description='BTCUSD perpetual', ticker='BTCUSD.P',
        currency='USD', basecurrency='BTC', period='1D', type='crypto',
        volumetype='base', mintick=0.1, pricescale=10, minmove=1, pointvalue=1.0,
        mincontract=0.001,
        opening_hours=[SymInfoInterval(day=d, start=dt_time(0, 0), end=dt_time(23, 59, 59))
                       for d in days],
        session_starts=[SymInfoSession(day=d, time=dt_time(0, 0)) for d in days],
        session_ends=[SymInfoSession(day=d, time=dt_time(23, 59, 59)) for d in days],
        timezone='UTC',
    )
    base = data_dir / __test_helper_name
    syminfo.save_toml(Path(f'{base}.toml'))
    with OHLCVWriter(Path(f'{base}.ohlcv'), syminfo.period, truncate=True) as writer:
        for i in range(10):
            price = 100.0 + i
            writer.write(OHLCV(1_767_225_600 + i * 86_400, price, price + 1, price - 1, price, 10.0))
    return workdir


def __test_dotted_tail_is_not_an_extension__():
    """Only a known data extension counts as the extension of a data name."""
    assert data_extension('bybit_BTCUSD.P_1D') == ''
    assert data_extension('bybit_BTCUSD.P_1D.ohlcv') == '.ohlcv'
    assert data_extension('bybit_BTCUSD.P_1D.CSV') == '.CSV'
    assert with_data_extension('data/bybit_BTCUSD.P_1D', '.ohlcv') == Path(
        'data/bybit_BTCUSD.P_1D.ohlcv')
    assert with_data_extension('bybit_BTCUSD.P_1D.csv', '.ohlcv') == Path('bybit_BTCUSD.P_1D.ohlcv')
    assert with_data_extension('bybit_BTCUSD.P_1D.ohlcv', '.toml') == Path('bybit_BTCUSD.P_1D.toml')
    assert with_data_extension('bybit_BTCUSD.P_1D.ohlcv', '') == Path('bybit_BTCUSD.P_1D')


def __test_converter_keeps_dotted_name__():
    """The converter derives its outputs and the symbol from the full dotted name."""
    source = Path('bybit_BTCUSD.P_1D.csv')
    assert DataConverter.guess_symbol_from_filename(Path('bybit_BTCUSD.P_1D'))[0] == \
        DataConverter.guess_symbol_from_filename(source)[0]
    assert with_data_extension(source, '.extra.csv') == Path('bybit_BTCUSD.P_1D.extra.csv')


def __test_run_resolves_bare_dotted_name__(__test_helper_workdir__, tmp_path):
    """``pyne run SCRIPT bybit_BTCUSD.P_1D`` runs on ``data/bybit_BTCUSD.P_1D.ohlcv``."""
    script_path = tmp_path / 'dotted_name.py'
    script_path.write_text(__test_helper_script, encoding='utf-8')
    plot_path = tmp_path / 'plot.csv'

    result = CliRunner().invoke(app, [
        '--workdir', str(__test_helper_workdir__),
        'run', str(script_path), __test_helper_name, '--plot', str(plot_path),
    ])

    assert result.exit_code == 0, result.output
    rows = plot_path.read_text(encoding='utf-8').splitlines()
    assert len(rows) == 1 + 10


def __test_data_convert_to_resolves_bare_dotted_name__(__test_helper_workdir__):
    """``pyne data convert-to bybit_BTCUSD.P_1D`` exports ``data/bybit_BTCUSD.P_1D.ohlcv``."""
    result = CliRunner().invoke(app, [
        '--workdir', str(__test_helper_workdir__),
        'data', 'convert-to', __test_helper_name,
    ])

    assert result.exit_code == 0, result.output
    assert (__test_helper_workdir__ / 'data' / f'{__test_helper_name}.csv').exists()
