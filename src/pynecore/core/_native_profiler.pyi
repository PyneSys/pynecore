"""Type stub of the compiled callbacks of :mod:`pynecore.core.profiler`."""
from types import CodeType
from typing import Iterable

__all__ = ['CLOCK_AVAILABLE', 'start', 'running', 'stop', 'results']

CLOCK_AVAILABLE: bool


def start(files: Iterable[tuple[str, int, int, Iterable[tuple[int, int]]]],
          regions: Iterable[tuple[CodeType, int]],
          tool: int) -> None: ...


def running() -> bool: ...


def stop() -> None: ...


def results() -> tuple[dict[str, dict[int, tuple[int, int]]], list[int], int]: ...
