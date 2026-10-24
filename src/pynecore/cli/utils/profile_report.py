"""
The terminal summary of a run profile (``pyne run --profile``).

The summary is the run's time split into categories, then the lines that took the most
time. The full profile is the JSON file the run wrote (see :mod:`pynecore.core.profiler`).
"""
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

__all__ = ['print_profile_report']

#: Category names in the summary, in the profile's order
_CATEGORY_NAMES = {
    'script': "Script code",
    'libraries': "Libraries",
    'engine': "Strategy engine",
    'output': "Output",
    'data': "Data feed",
    'security': "request.security",
    'runner': "Runner",
}

#: Longest code excerpt shown for a line
_CODE_WIDTH = 64


def _duration(ns: float) -> str:
    """A duration in the unit that keeps it readable, with three significant digits."""
    for scale, unit in ((1e9, "s"), (1e6, "ms"), (1e3, "µs"), (1.0, "ns")):
        if ns >= scale or unit == "ns":
            value: float = ns / scale
            digits = 0 if value >= 100 or unit == "ns" else 1 if value >= 10 else 2
            return f"{value:.{digits}f} {unit}"
    raise AssertionError("unreachable")


def _excerpt(source_lines: list[str], line: int, setup: bool) -> Text:
    """The code of a line, shortened to fit; a ``def`` line is marked as per-call setup."""
    code = source_lines[line - 1].strip() if line <= len(source_lines) else ""
    if len(code) > _CODE_WIDTH:
        code = code[:_CODE_WIDTH - 1] + "…"
    text = Text()
    if setup:
        text.append("per-call setup  ", style="dim italic")
    text.append(code)
    return text


def print_profile_report(profile: dict[str, Any], path: Path, top: int,
                         console: Console) -> None:
    """
    Print the summary of a run profile.

    :param profile: The profile, as the run wrote it.
    :param path: The profile file, named at the end.
    :param top: The number of lines to list.
    :param console: The console to print to.
    """
    wall_ns = profile['wall_ns']
    script_ns = profile['script_ns']

    console.print()
    console.print(f"[bold]Profile:[/] {profile['bars']:,} bars, {_duration(wall_ns)}; "
                  f"{_duration(script_ns)} in the script's code (the base of % script)")
    categories = Table(show_header=False, box=None, padding=(0, 2, 0, 2))
    categories.add_column(style="cyan")
    categories.add_column(justify="right")
    categories.add_column(justify="right", style="dim")
    for key, ns in profile['categories'].items():
        if ns <= 0:
            continue
        share: float = ns / wall_ns * 100 if wall_ns else 0.0
        categories.add_row(_CATEGORY_NAMES.get(key, key), _duration(ns), f"{share:.1f}%")
    console.print(categories)

    rows = []
    sources: dict[str, list[str]] = {}
    show_file = len(profile['files']) > 1
    for file in profile['files']:
        setup_lines = {block['start'] for block in file['blocks'] if block['kind'] == 'def'}
        for line, (ns, hits) in file['lines'].items():
            rows.append((ns, hits, int(line), file, int(line) in setup_lines))
    rows.sort(key=lambda row: row[0], reverse=True)

    if rows and top > 0:
        lines = Table(header_style="bold magenta", box=None, padding=(0, 1))
        if show_file:
            lines.add_column("File", style="dim")
        lines.add_column("Line", justify="right", style="cyan")
        lines.add_column("Time", justify="right")
        lines.add_column("% script", justify="right", style="green")
        lines.add_column("Execs", justify="right")
        lines.add_column("Avg", justify="right", style="dim")
        lines.add_column("Code", no_wrap=True, overflow="ellipsis")
        for ns, hits, line, file, setup in rows[:top]:
            file_path = file['path']
            if file_path not in sources:
                try:
                    sources[file_path] = Path(file_path).read_text(encoding='utf-8').splitlines()
                except OSError:
                    sources[file_path] = []
            share: float = ns / script_ns * 100 if script_ns else 0.0
            cells: list[Any] = [
                str(line), _duration(ns), f"{share:.1f}%", f"{hits:,}",
                _duration(ns / hits) if hits else "",
                _excerpt(sources[file_path], line, setup),
            ]
            if show_file:
                cells.insert(0, Path(file_path).name)
            lines.add_row(*cells)
        console.print()
        console.print(lines)

    console.print()
    console.print(Text(f"Full profile: {path}"))
