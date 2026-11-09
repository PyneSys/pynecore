"""
The ``pyne robustness`` command.

It runs a strategy once and writes its robustness results as CSV files, or analyses a
trade CSV that ``pyne run`` wrote earlier.
"""
import csv
import math
import sys
from datetime import time as dt_time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pynecore.cli import app
from pynecore.cli.app import app_state
from pynecore.core.ohlcv import OHLCVWriter
from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from pynecore.types.ohlcv import OHLCV

__test_helper_data = 'robustness_1h'

__test_helper_script = '''"""
@pyne
"""
from pynecore.lib import bar_index, close, script, strategy


@script.strategy("Robustness CLI", initial_capital=10000,
                 default_qty_type=strategy.fixed, default_qty_value=1)
def main():
    if bar_index % 20 == 5 and strategy.position_size == 0:
        strategy.entry("L", strategy.long)
        strategy.exit("LX", "L", stop=close - 2.0, limit=close + 4.0)
'''


@pytest.fixture
def __test_helper_workdir__(tmp_path, monkeypatch) -> Path:
    """A workdir with an hourly sine-wave data file and a strategy next to it."""
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
        prefix='TEST', description='Sine', ticker='SINE', currency='USD', basecurrency='SIN',
        period='60', type='crypto', volumetype='base', mintick=0.01, pricescale=100, minmove=1,
        pointvalue=1.0, mincontract=0.001,
        opening_hours=[SymInfoInterval(day=d, start=dt_time(0, 0), end=dt_time(23, 59, 59))
                       for d in days],
        session_starts=[SymInfoSession(day=d, time=dt_time(0, 0)) for d in days],
        session_ends=[SymInfoSession(day=d, time=dt_time(23, 59, 59)) for d in days],
        timezone='UTC',
    )
    base = data_dir / __test_helper_data
    syminfo.save_toml(Path(f'{base}.toml'))
    with OHLCVWriter(Path(f'{base}.ohlcv'), syminfo.period, truncate=True) as writer:
        for i in range(600):
            mid = round(100.0 + 6.0 * math.sin(i / 6.0), 2)
            writer.write(OHLCV(1_735_689_600 + i * 3600, mid, mid + 1.5, mid - 1.5, mid + 0.2, 1.0))
    (tmp_path / 'robustness_cli.py').write_text(__test_helper_script, encoding='utf-8')
    return workdir


def __test_helper_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def __test_run_writes_the_csv_files__(__test_helper_workdir__, tmp_path):
    """A backtest run measured in R, written as one CSV per kind of result."""
    out = tmp_path / 'out'
    result = CliRunner().invoke(app, [
        '--workdir', str(__test_helper_workdir__),
        'robustness', str(tmp_path / 'robustness_cli.py'), __test_helper_data,
        '--draws', '200', '--target', '2', '--output', str(out), '--json',
    ])
    assert result.exit_code == 0, result.output
    for suffix in ('robustness.csv', 'montecarlo.csv', 'costs.csv', 'periods.csv',
                   'coinflip_direction.csv', 'robustness.json'):
        assert (out / f'robustness_cli_{suffix}').exists()
    rows = __test_helper_rows(out / 'robustness_cli_robustness.csv')
    total = next(r for r in rows if r['Test'] == 'Backtest' and r['Metric'] == 'Total')
    assert total['Unit'] == 'R'
    assert any(r['Test'] == 'Account Monte Carlo' and r['Metric'].startswith('Paths reaching')
               for r in rows)
    flip = next(r for r in rows if r['Test'] == 'Coin flip (direction)')
    assert float(flip['Value']) == float(total['Value'])


def __test_timing_coin_flip__(__test_helper_workdir__, tmp_path):
    """``--coinflip timing`` runs one backtest per draw and writes their totals."""
    out = tmp_path / 'out'
    result = CliRunner().invoke(app, [
        '--workdir', str(__test_helper_workdir__),
        'robustness', str(tmp_path / 'robustness_cli.py'), __test_helper_data,
        '--draws', '200', '--coinflip', 'timing', '--coinflip-runs', '10', '--output', str(out),
    ])
    assert result.exit_code == 0, result.output
    assert len(__test_helper_rows(out / 'robustness_cli_coinflip_timing.csv')) == 10
    assert not (out / 'robustness_cli_coinflip_direction.csv').exists()


def __test_parallel_backtests_give_the_same_totals__(__test_helper_workdir__, tmp_path):
    """``--jobs`` spreads the draws over worker processes without changing any of them."""
    totals = {}
    for jobs in ('1', '3'):
        out = tmp_path / f'out_{jobs}'
        result = CliRunner().invoke(app, [
            '--workdir', str(__test_helper_workdir__),
            'robustness', str(tmp_path / 'robustness_cli.py'), __test_helper_data,
            '--draws', '200', '--coinflip', 'both', '--exact', '--coinflip-runs', '10',
            '--early-stop', 'off', '--jobs', jobs, '--output', str(out),
        ])
        assert result.exit_code == 0, result.output
        totals[jobs] = [__test_helper_rows(out / f'robustness_cli_coinflip_{mode}.csv')
                        for mode in ('direction', 'timing')]
    assert totals['1'] == totals['3']
    assert all(len(rows) == 10 for rows in totals['1'])


def __test_parallel_early_stop_stops_the_workers__(__test_helper_workdir__, tmp_path):
    """An early stop with draws still running in the workers ends, with the draws of one job."""
    totals = {}
    for jobs in ('1', '4'):
        out = tmp_path / f'out_{jobs}'
        result = CliRunner().invoke(app, [
            '--workdir', str(__test_helper_workdir__),
            'robustness', str(tmp_path / 'robustness_cli.py'), __test_helper_data,
            '--draws', '200', '--coinflip', 'direction', '--exact', '--coinflip-runs', '200',
            '--early-stop', 'better', '--jobs', jobs, '--output', str(out),
        ])
        assert result.exit_code == 0, result.output
        totals[jobs] = __test_helper_rows(out / 'robustness_cli_coinflip_direction.csv')
    assert totals['1'] == totals['4']
    assert len(totals['1']) < 200


__test_helper_security_script = '''"""
@pyne
"""
from pynecore.lib import bar_index, close, request, script, strategy, syminfo, time, timenow


@script.strategy("Robustness CLI security", initial_capital=10000,
                 default_qty_type=strategy.fixed, default_qty_value=1)
def main():
    htf = request.security(syminfo.tickerid, "240", close)
    recent = time >= timenow - 300 * 3_600_000
    if recent and close > htf - 100 and bar_index % 20 == 5 and strategy.position_size == 0:
        strategy.entry("L", strategy.long)
        strategy.exit("LX", "L", stop=close - 2.0, limit=close + 4.0)
'''


def __test_parallel_backtests_with_security_and_timenow__(__test_helper_workdir__, tmp_path):
    """A worker starts the security children and keeps the replay's ``timenow``."""
    data_dir = __test_helper_workdir__ / 'data'
    name = 'robustness_ms_1h'
    (data_dir / f'{name}.toml').write_bytes((data_dir / f'{__test_helper_data}.toml').read_bytes())
    # Hourly bars in milliseconds, so the 4-hour bars close and ``timenow`` is a real instant
    with OHLCVWriter(data_dir / f'{name}.ohlcv', '60', truncate=True) as writer:
        for i in range(600):
            mid = round(100.0 + 6.0 * math.sin(i / 6.0), 2)
            writer.write(OHLCV(1_735_689_600_000 + i * 3_600_000, mid, mid + 1.5, mid - 1.5,
                               mid + 0.2, 1.0))
    script = tmp_path / 'robustness_security.py'
    script.write_text(__test_helper_security_script, encoding='utf-8')
    totals = {}
    for jobs in ('1', '2'):
        out = tmp_path / f'out_{jobs}'
        result = CliRunner().invoke(app, [
            '--workdir', str(__test_helper_workdir__),
            'robustness', str(script), name,
            '--draws', '200', '--coinflip', 'direction', '--exact', '--coinflip-runs', '10',
            '--early-stop', 'off', '--jobs', jobs, '--security', f'240={name}',
            '--output', str(out),
        ])
        assert result.exit_code == 0, result.output
        totals[jobs] = __test_helper_rows(out / 'robustness_security_coinflip_direction.csv')
    assert totals['1'] == totals['2']
    assert len(totals['1']) == 10


def __test_from_trades_measures_in_currency__(__test_helper_workdir__, tmp_path):
    """A trade CSV carries no stops, so its results are in account currency."""
    trade_path = tmp_path / 'trades.csv'
    run = CliRunner().invoke(app, [
        '--workdir', str(__test_helper_workdir__),
        'run', str(tmp_path / 'robustness_cli.py'), __test_helper_data, '--no-plot',
        '--trade', str(trade_path), '--strat', str(tmp_path / 'strat.csv'),
    ])
    assert run.exit_code == 0, run.output

    out = tmp_path / 'out'
    result = CliRunner().invoke(app, [
        '--workdir', str(__test_helper_workdir__),
        'robustness', 'robustness_cli', '--from-trades', str(trade_path),
        '--draws', '200', '--output', str(out),
    ])
    assert result.exit_code == 0, result.output
    rows = __test_helper_rows(out / 'robustness_cli_robustness.csv')
    total = next(r for r in rows if r['Test'] == 'Backtest' and r['Metric'] == 'Total')
    assert total['Unit'] == 'currency'
