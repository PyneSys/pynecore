"""
@pyne
"""
import json
import os
import sys
from pathlib import Path

import pytest

from pynecore.core import profiler
from pynecore.core.profiler import CATEGORIES, ProfilerUnavailable
from pynecore.lib import bar_index, close, plot, script, strategy


def scaled(value: float) -> float:
    return value * 2.0


@script.strategy("Profiler", initial_capital=1000000)
def main():
    total = 0.0
    for i in range(3):
        total += close
    if bar_index % 2 == 0:
        total = scaled(total)
    j = 0
    while j < 2:
        total += scaled(
            close)
        j += 1
    for k in range(
            2):
        total += k
    if bar_index % 4 == 0:
        strategy.entry("L", strategy.long, qty=1)
    if bar_index % 4 == 2:
        strategy.close("L")
    plot(
        total,
        "total")


def __test_helper_require_profiler():
    """Skip where profiling cannot run; a wheel build sets the variable to make it fail."""
    if sys.version_info < (3, 13):
        pytest.skip("profiling needs Python 3.13+")
    if profiler._native_profiler is None:  # noqa
        if os.environ.get('PYNE_REQUIRE_NATIVE_PROFILER'):
            pytest.fail("the compiled profiler extension is missing from this build")
        pytest.skip("the compiled profiler extension is not built")


def __test_helper_line(text: str) -> int:
    """The number of the first line of this file that contains ``text``."""
    lines = Path(__file__).read_text(encoding='utf-8').splitlines()
    return next(number for number, line in enumerate(lines, 1) if text in line)


def __test_helper_profile(csv_reader, runner, path: Path) -> dict:
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        runner(cr, profile_path=path).run()
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.mark.aot_unsupported("the profiler measures CPython code objects")
def __test_profile_counts_every_execution__(csv_reader, runner, tmp_path):
    """ Each line counts the times it ran, loops and branches included """
    __test_helper_require_profiler()
    profile = __test_helper_profile(csv_reader, runner, tmp_path / 'profile.json')
    bars = profile['bars']
    assert bars > 10

    # Libraries registered by other scripts of this process run too
    file, = [file for file in profile['files'] if file['role'] == 'script']
    assert Path(file['path']) == Path(__file__).resolve()
    hits = {int(line): counts[1] for line, counts in file['lines'].items()}

    def ran(text: str) -> int:
        return hits.get(__test_helper_line(text), 0)

    even = (bars + 1) // 2
    assert ran("    total = 0.0") == bars
    assert ran("        total += close") == 3 * bars
    # The header runs once more than the body: its last check ends the loop
    assert ran("    for i in range(3):") == 4 * bars
    assert ran("    if bar_index % 2 == 0:") == bars
    assert ran("        total = scaled(total)") == even
    assert ran("    return value * 2.0") == even + 2 * bars
    # A header or a statement over several lines steps back onto its first line once
    # it is evaluated: that is not another execution, nor another check of the loop
    assert ran("    while j < 2:") == 3 * bars
    assert ran("        total += scaled(") == 2 * bars
    assert ran("    for k in range(") == 3 * bars
    assert ran("            2):") == bars
    assert ran("strategy.entry(\"L\"") == (bars + 3) // 4
    # A call over several lines steps back onto its first line when it runs: that
    # is not a second execution
    assert ran("    plot(") == bars


@pytest.mark.aot_unsupported("the profiler measures CPython code objects")
def __test_profile_times_add_up__(csv_reader, runner, tmp_path):
    """ The categories cover the run, the blocks sum their lines """
    __test_helper_require_profiler()
    profile = __test_helper_profile(csv_reader, runner, tmp_path / 'profile.json')

    categories = profile['categories']
    assert list(categories) == list(CATEGORIES)
    assert all(ns >= 0 for ns in categories.values())
    assert profile['wall_ns'] == sum(categories.values())
    for category in ('script', 'engine', 'runner'):
        assert categories[category] > 0, category
    assert categories['security'] == 0

    # Libraries registered by other scripts of this process run too
    file, = [file for file in profile['files'] if file['role'] == 'script']
    lines = {int(line): counts for line, counts in file['lines'].items()}
    header = __test_helper_line("    for i in range(3):")
    loop, = [block for block in file['blocks']
             if block['kind'] == 'loop' and block['start'] == header]
    assert (loop['start'], loop['end']) == (header, header + 1)
    assert loop['hits'] == lines[header][1]
    assert loop['header_ns'] == lines[header][0]
    assert loop['ns'] == lines[header][0] + lines[header + 1][0]

    # The script's time holds every line of main, the calls they make included
    main_start = __test_helper_line("def main():")
    main_block, = [block for block in file['blocks']
                   if block['kind'] == 'def' and block['start'] == main_start]
    assert main_block['ns'] <= profile['script_ns'] <= profile['wall_ns']
    assert sys.monitoring.get_tool(sys.monitoring.PROFILER_ID) is None


@pytest.mark.aot_unsupported("the profiler measures CPython code objects")
def __test_profile_leaves_results_unchanged__(csv_reader, runner, tmp_path):
    """ A profiled run computes the same values as a plain one """
    __test_helper_require_profiler()
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        plain = [plot_data['total'] for _candle, plot_data, _trades in runner(cr).run_iter()]
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        profiled = [plot_data['total'] for _candle, plot_data, _trades
                    in runner(cr, profile_path=tmp_path / 'profile.json').run_iter()]
    assert profiled == plain
    assert (tmp_path / 'profile.json').exists()


@pytest.mark.aot_unsupported("the profiler measures CPython code objects")
def __test_failed_run_writes_no_profile__(csv_reader, runner, tmp_path):
    """ A run that fails stops the profiler and leaves no profile behind """
    __test_helper_require_profiler()

    def failing(candles):
        for index, candle in enumerate(candles):
            if index == 5:
                raise RuntimeError("data source failed")
            yield candle

    path = tmp_path / 'profile.json'
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        with pytest.raises(RuntimeError, match="data source failed"):
            runner(failing(cr), profile_path=path).run()
    assert not path.exists()
    assert sys.monitoring.get_tool(sys.monitoring.PROFILER_ID) is None


@pytest.mark.aot_unsupported("the profiler measures CPython code objects")
def __test_profiler_refuses_where_it_cannot_run__(csv_reader, runner, tmp_path, monkeypatch):
    """ No profiling next to another profiler, without the extension, or in a live run """
    __test_helper_require_profiler()
    path = tmp_path / 'profile.json'

    sys.monitoring.use_tool_id(sys.monitoring.PROFILER_ID, "other profiler")
    try:
        with csv_reader('series_if_for.csv', subdir="data") as cr:
            with pytest.raises(ProfilerUnavailable, match="other profiler"):
                runner(cr, profile_path=path)
    finally:
        sys.monitoring.free_tool_id(sys.monitoring.PROFILER_ID)

    monkeypatch.setattr(profiler, '_native_profiler', None)
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        with pytest.raises(ProfilerUnavailable, match="compiled profiler extension"):
            runner(cr, profile_path=path)
    monkeypatch.undo()

    with csv_reader('series_if_for.csv', subdir="data") as cr:
        with pytest.raises(ValueError, match="backtests only"):
            runner(cr, profile_path=path, live=True)
    assert not path.exists()
