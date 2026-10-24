"""
Per-line profiler of a Pyne code run.

For every line of the script and of its libraries the profiler measures the time spent on
it and how many times it ran, and it splits the run's time into categories: the script's
own code, its libraries, the strategy engine, writing the outputs, reading the data,
waiting for ``request.security`` contexts, and the rest of the runner. The result is a
JSON file next to the run's other outputs (``output/<stem>_profile.json``).

The measurement runs on ``sys.monitoring`` (the ``PROFILER_ID`` tool) with the compiled
callbacks of :mod:`pynecore.core._native_profiler`. There is no pure-Python variant: its
own per-line cost would land on the measured lines and skew the result toward cheap lines
that run often. The profiler needs Python 3.13 or newer and the compiled extension.

What a number means
-------------------

- A line's time runs from its line event to the next line event of the same call, so it
  includes everything the line calls: a ``ta.sma(...)`` line carries the indicator's own
  time, a call of a script function carries the whole call.
- A function's lines add up every call of it; its ``def`` line carries the per-call setup
  PyneCore runs before the first statement (series and state bookkeeping).
- A loop or a conditional is reported as a block too: the sum of its lines, with the
  header's own share apart.
- A loop header spanning several lines whose last line also carries the body
  (``for i in range(\\n 3): x = i``) overcounts: the header's hits and the block's hits are
  too high, so the per-hit averages built on them are too low; the times are exact. A body
  on its own line is counted exactly.
- A strategy re-runs the script within a bar for some order features (``calc_on_order_fills``,
  ``calc_on_every_tick``); those runs count, so a line can run more times than there are
  bars.
- Category times are exclusive and add up to the time the runner was running. The time
  the caller spends between two bars (writing to a terminal, streaming to an IDE) is not
  part of any of them.
"""
import ast
import hashlib
import inspect
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import CodeType
from typing import Any, Generator

from . import script
from .bar_magnifier import BarMagnifier
from .csv_file import CSVReader, CSVWriter
# noinspection PyProtectedMember
from .ohlcv import ChartBarWindow, OHLCVReader, _V2OHLCVReader
from .script_timeframe import ScriptTimeframe
from .security import _wait_with_liveness
from .security_shm import RingReader
from .strategy_stats import calculate_strategy_statistics, write_strategy_statistics_csv
from .viz import VizWriter
from ..lib.strategy import SimPosition

try:
    from . import _native_profiler
except ImportError:
    _native_profiler = None

__all__ = ['CATEGORIES', 'SCHEMA', 'ProfilerUnavailable', 'RunProfiler']

#: The time categories, in the order the native module numbers them
CATEGORIES = ('script', 'libraries', 'engine', 'output', 'data', 'security', 'runner')
_SCRIPT, _LIBRARIES, _ENGINE, _OUTPUT, _DATA, _SECURITY, _RUNNER = range(len(CATEGORIES))

#: Version of the profile file format
SCHEMA = 1

#: Statements reported as blocks, by the kind written to the profile
_BLOCK_KINDS: dict[type, str] = {
    ast.FunctionDef: 'def', ast.AsyncFunctionDef: 'def',
    ast.For: 'loop', ast.AsyncFor: 'loop', ast.While: 'loop',
    ast.If: 'if', ast.Match: 'match', ast.With: 'with', ast.AsyncWith: 'with',
    ast.Try: 'try', ast.TryStar: 'try',
}


class ProfilerUnavailable(RuntimeError):
    """Profiling is not possible in this environment."""


def _loop_headers(tree: ast.Module) -> list[tuple[int, int]]:
    """The header of every loop, as its first and last line: every iteration checks
    the loop on its header, which can span several lines."""
    return [(node.lineno, max(node.lineno, node.body[0].lineno - 1))
            for node in ast.walk(tree) if isinstance(node, (ast.For, ast.AsyncFor, ast.While))]


class _ProfiledFile:
    """A source file whose lines the profiler measures."""

    def __init__(self, co_filename: str, category: int):
        self.co_filename = co_filename
        self.category = category
        self.path = Path(co_filename).resolve()
        self.source = self.path.read_text(encoding='utf-8')
        self.line_count = len(self.source.splitlines())
        self.tree = ast.parse(self.source)
        self.loop_headers = _loop_headers(self.tree)



def _check_available() -> Any:
    """The native module, once it is certain a profiler can run here.

    :return: :mod:`pynecore.core._native_profiler`.
    :raises ProfilerUnavailable: If the Python version, the installation or another
                                 profiler rules it out.
    """
    if sys.version_info < (3, 13):
        raise ProfilerUnavailable(
            f"Profiling needs Python 3.13 or newer; this is Python "
            f"{sys.version_info.major}.{sys.version_info.minor}.")
    if _native_profiler is None or not _native_profiler.CLOCK_AVAILABLE:
        raise ProfilerUnavailable(
            "Profiling needs PyneCore's compiled profiler extension, which this "
            "installation does not have. Install a PyneCore wheel built for this platform.")
    holder = sys.monitoring.get_tool(sys.monitoring.PROFILER_ID)
    if holder is not None:
        raise ProfilerUnavailable(
            f"Another profiler ({holder}) is already running in this process.")
    return _native_profiler


def _code_of(func: Any) -> CodeType:
    """The code object of a function, through any decorator wrappers."""
    return inspect.unwrap(func).__code__


def _nested_codes(func: Any, names: tuple[str, ...]) -> list[CodeType]:
    """The code objects of the functions defined inside ``func`` under ``names``."""
    codes = [const for const in _code_of(func).co_consts
             if isinstance(const, CodeType) and const.co_name in names]
    found = {code.co_name for code in codes}
    missing = [name for name in names if name not in found]
    assert not missing, f"{func.__qualname__} no longer defines {missing}"
    return codes


# noinspection PyProtectedMember
def _regions(runner: Any) -> list[tuple[CodeType, int]]:
    """The runtime functions timed as a category, with their category.

    :param runner: The :class:`ScriptRunner` being profiled.
    """
    runner_cls = type(runner)
    runner_module = sys.modules[runner_cls.__module__]
    run_iter = runner_cls._run_iter
    regions: list[tuple[CodeType, int]] = [
        (_code_of(run_iter), _RUNNER),
        (_code_of(runner_cls._run_iter_magnified), _RUNNER),
    ]
    regions += [(code, _ENGINE) for code in [
        _code_of(runner_cls._process_orders),
        _code_of(runner_cls._process_orders_magnified),
        _code_of(runner_cls._process_deferred_margin_call),
        _code_of(SimPosition.settle_immediate_closes),
        _code_of(SimPosition.process_orders_at_close),
        _code_of(SimPosition.process_orders_magnified),
        *_nested_codes(run_iter, ('_coof_loop', '_ceht_loop', '_coof_magnified_loop')),
    ]]
    regions += [(code, _OUTPUT) for code in [
        _code_of(runner_cls._write_viz_bar),
        _code_of(CSVWriter.open), _code_of(CSVWriter.write), _code_of(CSVWriter.write_ohlcv),
        _code_of(CSVWriter.close),
        _code_of(VizWriter.write_bar), _code_of(VizWriter.write_events),
        _code_of(VizWriter.write_drawings_snapshot), _code_of(VizWriter.write_end),
        _code_of(VizWriter.close),
        _code_of(calculate_strategy_statistics), _code_of(write_strategy_statistics_csv),
        *_nested_codes(run_iter, ('_write_bar_output', '_write_plot_row')),
    ]]
    regions += [(code, _DATA) for code in [
        _code_of(runner_module._set_lib_properties),
        _code_of(CSVReader.__iter__), _code_of(CSVReader.read_from),
        _code_of(_V2OHLCVReader.__iter__), _code_of(_V2OHLCVReader.read_from),
        _code_of(OHLCVReader.read_from),
        _code_of(ChartBarWindow.bars), _code_of(BarMagnifier.__iter__),
        _code_of(ScriptTimeframe.feed),
    ]]
    regions += [(code, _SECURITY) for code in [
        _code_of(_wait_with_liveness), _code_of(RingReader.wait_for_close),
        *_nested_codes(run_iter, ('_prepare_security_context', '_spawn_security_process',
                                  '_deferred_resolve')),
    ]]
    return regions


# noinspection PyProtectedMember
def _profiled_files(runner: Any) -> list[_ProfiledFile]:
    """The script's file and the files of the libraries it imported.

    :param runner: The :class:`ScriptRunner` being profiled.
    """
    script_file = _code_of(runner.script_module.main).co_filename
    files = {script_file: _ProfiledFile(script_file, _SCRIPT)}
    # The registry is process-wide: only libraries whose module is loaded now count
    for _title, library_main in script._registered_libraries:
        if sys.modules.get(getattr(library_main, '__module__', '')) is None:
            continue
        filename = _code_of(library_main).co_filename
        if filename not in files:
            files[filename] = _ProfiledFile(filename, _LIBRARIES)
    return list(files.values())


def _blocks(tree: ast.Module, lines: dict[int, tuple[int, int]]) -> list[dict[str, Any]]:
    """The compound statements of a file that ran, with their summed times.

    :param tree: The file's syntax tree.
    :param lines: The file's measured lines: line number -> (nanoseconds, executions).
    :return: One entry per block: kind, first and last line, the block's time, the
             header line's own time and the header's executions.
    """
    blocks = []
    for node in ast.walk(tree):
        kind = _BLOCK_KINDS.get(type(node))
        if kind is None or not isinstance(node, ast.stmt):
            continue
        start, end = node.lineno, node.end_lineno or node.lineno
        ran = [lines[line] for line in range(start, end + 1) if line in lines]
        if not ran:
            continue
        header_ns, header_hits = lines.get(start, (0, 0))
        blocks.append({'kind': kind, 'start': start, 'end': end,
                       'ns': sum(ns for ns, _hits in ran),
                       'header_ns': header_ns, 'hits': header_hits})
    blocks.sort(key=lambda block: (block['start'], -block['end']))
    return blocks


def _pynecore_version() -> str | None:
    try:
        return version('pynesys-pynecore')
    except PackageNotFoundError:
        return None


class RunProfiler:
    """The profiler of one :class:`ScriptRunner` run."""

    def __init__(self, runner: Any, path: Path):
        """
        :param runner: The runner to profile, its script already imported.
        :param path: The profile file to write.
        :raises ProfilerUnavailable: If profiling is not possible here.
        """
        self._native = _check_available()
        self._runner = runner
        self._path = path
        self._files = _profiled_files(runner)
        self._regions = _regions(runner)
        self._results: tuple[dict[str, dict[int, tuple[int, int]]], list[int]] | None = None

    def profile(self, results: Generator) -> Generator:
        """Measure the runner's iteration while passing its results through, then
        write the profile.

        The iteration is measured from its first statement to the end of its teardown
        (closing the outputs, the strategy statistics). A run that fails or is closed
        early writes no profile.

        :param results: The generator of the runner's iteration, not started yet.
        :return: The same results.
        :raises ProfilerUnavailable: If the profiler could not start.
        """
        self.start()
        try:
            yield from results
        finally:
            self.stop()
        self.write()

    def start(self) -> None:
        """Start measuring."""
        try:
            self._native.start(
                [(file.co_filename, file.line_count, file.category, file.loop_headers)
                 for file in self._files],
                self._regions,
                sys.monitoring.PROFILER_ID)
        except ValueError as error:
            raise ProfilerUnavailable(f"The profiler could not start: {error}") from error

    def stop(self) -> None:
        """Stop measuring and keep the results. Does nothing if measuring never started."""
        if not self._native.running():
            return
        self._native.stop()
        self._results = self._native.results()

    def to_dict(self) -> dict[str, Any]:
        """The profile in the file format (see :data:`SCHEMA`)."""
        assert self._results is not None, "the profiler has not run"
        file_lines, category_ns = self._results
        files = []
        for file in self._files:
            lines = file_lines.get(file.co_filename, {})
            files.append({
                'path': str(file.path),
                'sha256': hashlib.sha256(file.source.encode('utf-8')).hexdigest(),
                'role': 'script' if file.category == _SCRIPT else 'library',
                'lines': {str(line): [ns, hits] for line, (ns, hits) in sorted(lines.items())},
                'blocks': _blocks(file.tree, lines),
            })
        categories = dict(zip(CATEGORIES, category_ns))
        return {
            'schema': SCHEMA,
            'pynecore': _pynecore_version(),
            'python': '.'.join(str(part) for part in sys.version_info[:3]),
            'bars': self._runner.bar_index + 1,
            'wall_ns': sum(categories.values()),
            'categories': categories,
            'files': files,
        }

    def write(self) -> None:
        """Write the profile, replacing any earlier one only once it is complete."""
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.partial')
        partial.write_text(json.dumps(self.to_dict(), indent=1), encoding='utf-8')
        os.replace(partial, path)
