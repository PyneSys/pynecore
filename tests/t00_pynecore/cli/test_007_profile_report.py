"""
The terminal summary of ``pyne run --profile``.

The summary reads the profile file the run wrote: the categories, then the slowest lines
with their share of the script's time.
"""
from pathlib import Path

from rich.console import Console

from pynecore.cli.utils.profile_report import print_profile_report

_SCRIPT = '''def main():
    fast = 1
    slow = values["[red]x[/red]"]
    return fast + slow
'''

_LIBRARY = '''def helper():
    return 2
'''


def _profile(tmp_path: Path) -> tuple[dict, Path]:
    script = tmp_path / 'script.py'
    library = tmp_path / 'library.py'
    script.write_text(_SCRIPT, encoding='utf-8')
    library.write_text(_LIBRARY, encoding='utf-8')
    profile = {
        'bars': 1000,
        'wall_ns': 4_000_000,
        'script_ns': 2_000_000,
        'categories': {'script': 1_500_000, 'libraries': 500_000, 'engine': 0,
                       'output': 1_000_000, 'data': 1_000_000, 'security': 0, 'runner': 0},
        'files': [
            {'path': str(script), 'role': 'script',
             'lines': {'1': [100_000, 1000], '2': [200_000, 1000], '3': [1_000_000, 1000],
                       '4': [50_000, 1000]},
             'blocks': [{'kind': 'def', 'start': 1, 'end': 4, 'ns': 1_350_000,
                         'header_ns': 100_000, 'hits': 1000}]},
            {'path': str(library), 'role': 'library',
             'lines': {'2': [500_000, 1000]},
             'blocks': [{'kind': 'def', 'start': 1, 'end': 2, 'ns': 500_000,
                         'header_ns': 0, 'hits': 0}]},
        ],
    }
    return profile, tmp_path / 'script_profile.json'


def _render(profile: dict, path: Path, top: int) -> str:
    console = Console(record=True, width=200, color_system=None)
    print_profile_report(profile, path, top, console)
    return console.export_text()


def __test_lists_the_slowest_lines_first__(tmp_path):
    """ The lines come by time, as many as asked, each with its share of the script """
    profile, path = _profile(tmp_path)
    lines = _render(profile, path, 3).splitlines()

    rows = [line for line in lines if line.strip().startswith(('script.py', 'library.py'))]
    assert len(rows) == 3
    assert 'slow = values' in rows[0] and '50.0%' in rows[0]
    assert 'library.py' in rows[1] and 'return 2' in rows[1] and '25.0%' in rows[1]
    assert 'fast = 1' in rows[2] and '10.0%' in rows[2]


def __test_marks_setup_and_keeps_code_literal__(tmp_path):
    """ A ``def`` line is per-call setup; brackets in the code are not rich markup """
    profile, path = _profile(tmp_path)
    text = _render(profile, path, 10)

    assert 'per-call setup  def main():' in text
    assert 'values["[red]x[/red]"]' in text
    assert f"Full profile: {path}" in text


def __test_lists_security_contexts__(tmp_path):
    """ Each ``request.security`` context gets a row: its data, timeframe and time """
    profile, path = _profile(tmp_path)
    profile['contexts'] = [{'sec_ids': ['sec·0'], 'timeframe': '1D', 'data': 'EURUSD_1D.ohlcv',
                            'bars': 250, 'script_ns': 3_400_000, 'files': []}]
    text = _render(profile, path, 0)

    row, = [line for line in text.splitlines() if 'EURUSD_1D.ohlcv' in line]
    assert '1D' in row and '250 bars' in row and '3.40 ms' in row


def __test_categories_show_only_measured_time__(tmp_path):
    """ The summary names the run's categories that took time, with their share """
    profile, path = _profile(tmp_path)
    text = _render(profile, path, 0)

    assert '1,000 bars' in text
    assert 'Script code' in text and '37.5%' in text
    assert 'Data feed' in text
    assert 'Strategy engine' not in text and 'request.security' not in text
    assert 'contexts' not in text
    assert 'per-call setup' not in text
