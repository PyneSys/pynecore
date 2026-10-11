"""
Deferred stop requests for live broker runs.

A stop signal (``SIGTERM`` from a supervisor, ``SIGINT`` from Ctrl-C) used to
raise :class:`KeyboardInterrupt` wherever the main thread happened to be. Inside
an order sync that is the worst place: the venue request keeps running on its
worker thread and lands, while the engine never records the outcome — the run
ends holding venue exposure its own journal does not own (measured live:
capitalcom cycle 268, 2026-10-11 — the rotation stop arrived during a market
entry POST, the position opened and the journal row stayed ``submitted``).

The sync engine marks its state-changing entry points as critical sections. A
stop request that arrives inside one is deferred until the outermost section
completes, then raised at that safe point; outside a section it is raised at
once. A second request while one is already pending is honoured immediately,
so a wedged section can still be forced down.
"""
import functools
import threading
from collections.abc import Callable
from typing import ParamSpec, TypeVar

__all__ = ['critical_section', 'request_interrupt', 'interrupt_pending']

_P = ParamSpec('_P')
_T = TypeVar('_T')

_depth = 0
_pending = False


def critical_section(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Defer stop requests on the main thread until ``func`` returns.

    The deferred :class:`KeyboardInterrupt` is raised when the outermost
    section returns normally. When the section ends with an exception, that
    exception propagates unchanged and the request stays pending for the next
    section boundary, so an error is never masked by the stop.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        global _depth, _pending
        if threading.current_thread() is not threading.main_thread():
            return func(*args, **kwargs)
        _depth += 1
        try:
            result = func(*args, **kwargs)
        finally:
            _depth -= 1
        if _depth == 0 and _pending:
            _pending = False
            raise KeyboardInterrupt
        return result

    return wrapper


def request_interrupt() -> None:
    """Stop-signal entry point: interrupt now, or at the end of the running section.

    :raises KeyboardInterrupt: When no critical section is running, or when a
        stop request is already pending (a repeated signal forces the stop).
    """
    global _pending
    if _depth == 0 or _pending:
        _pending = False
        raise KeyboardInterrupt
    _pending = True


def interrupt_pending() -> bool:
    """Whether a stop request is waiting for the running critical section."""
    return _pending
