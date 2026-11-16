"""
A Ctrl-C on a live run. The chart idles in the live stream's wait for its next
update, so that is where the interrupt lands, and a repeated one (a second Ctrl-C, or a
signal sent both to the process and to its group) arrives while the first is still
unwinding: the stop must still surface as a KeyboardInterrupt with the stream torn
down. Security children leave the interrupt to the chart, which owns their shutdown.
"""
import asyncio
import signal
import sys
import threading

import pytest

from pynecore.core.live_runner import live_ohlcv_generator
from pynecore.core.script_runner import LIVE_TRANSITION
# noinspection PyProtectedMember
from pynecore.core.security_process import _leave_interrupt_to_the_chart


class _SilentVenue:
    """A live provider that connects and then never delivers an update."""

    reconnect_delay = 0.01
    max_reconnect_delay = 0.05
    feed_timeout_bars = None

    def __init__(self):
        self._connected = False

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    @staticmethod
    async def watch_ohlcv(_symbol: str, _timeframe: str):
        await asyncio.Event().wait()

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return symbol

    async def on_disconnect(self):
        pass

    async def on_reconnect(self):
        pass

    @staticmethod
    async def can_shutdown():
        return True


def __test_repeated_interrupt_at_the_live_wait_stops_the_stream__():
    """The first interrupt wakes the consumer's wait, the second lands while it unwinds

    Python handles a signal at the next call, so the repeated one is raised at the
    first call made after the first KeyboardInterrupt: that is where the trace hook
    raises it. A wait whose own lock bookkeeping that call belongs to ends in
    ``RuntimeError: release unlocked lock`` instead.
    """
    previous_trace = sys.gettrace()

    def _repeated(_frame, event, _arg):
        if event == 'call':
            sys.settrace(previous_trace)
            raise KeyboardInterrupt
        return None

    def _first(_signum, _frame):
        sys.settrace(_repeated)
        raise KeyboardInterrupt

    venue = _SilentVenue()
    previous_handler = signal.signal(signal.SIGUSR1, _first)
    stream = live_ohlcv_generator(venue, "X", "1D", shutdown_timeout=5.0)  # type: ignore[arg-type]
    # The first stop is a real signal to the main thread parked in the wait
    stopper = threading.Timer(0.3, signal.pthread_kill,
                              (threading.main_thread().ident, signal.SIGUSR1))
    try:
        assert next(stream) is LIVE_TRANSITION
        stopper.start()
        with pytest.raises(KeyboardInterrupt):
            next(stream)
    finally:
        stopper.join()
        sys.settrace(previous_trace)
        signal.signal(signal.SIGUSR1, previous_handler)
    assert not venue.is_connected
    assert not any(t.name == "live-provider" for t in threading.enumerate())


def __test_security_child_leaves_the_interrupt_to_the_chart__(
        monkeypatch: pytest.MonkeyPatch):
    """A process started through multiprocessing ignores SIGINT; any other is left alone"""
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        _leave_interrupt_to_the_chart()
        assert signal.getsignal(signal.SIGINT) is previous_handler

        monkeypatch.setattr("pynecore.core.security_process.parent_process", lambda: object())
        _leave_interrupt_to_the_chart()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous_handler)
