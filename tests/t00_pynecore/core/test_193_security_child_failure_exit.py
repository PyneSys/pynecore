"""
How a security child leaves on a failure: a coded error goes to the chart through the
failure pipe and the process exits with code 1 without a traceback of its own (the
chart reports the error in full), while any other exception is re-raised, so
multiprocessing prints its traceback, and its text report still reaches the chart.
"""
from multiprocessing import Pipe

import pytest

from pynecore.core import security_process
from pynecore.core.errors import PyneError
from pynecore.errorcodes import PROVIDER_UNAVAILABLE, ProviderUnavailableError


def _run_failing(monkeypatch: pytest.MonkeyPatch, exc: BaseException):
    """Run the child entry point on a main that raises ``exc``.

    :return: ``(raised, report)``: what left the entry point and what the chart got.
    """
    def failing_main(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(security_process, 'security_process_main', failing_main)
    reader, writer = Pipe(duplex=False)
    try:
        with pytest.raises(BaseException) as info:
            security_process.run_security_process(failure_conn=writer)
        report = reader.recv() if reader.poll(5) else None
    finally:
        reader.close()
    return info.value, report


def __test_coded_failure_exits_quietly__(monkeypatch: pytest.MonkeyPatch):
    """A coded error is handed to the chart and the child exits with code 1"""
    raised, report = _run_failing(
        monkeypatch, ProviderUnavailableError(provider="ccxt", why="invalid interval"))
    # ``SystemExit`` with an int code: multiprocessing exits with it and prints nothing
    assert type(raised) is SystemExit
    assert raised.code == 1
    assert isinstance(report, PyneError)
    assert report.code is PROVIDER_UNAVAILABLE


def __test_unexpected_failure_keeps_its_traceback__(monkeypatch: pytest.MonkeyPatch):
    """Any other exception is re-raised for multiprocessing to print, and reported"""
    raised, report = _run_failing(monkeypatch, RuntimeError("boom"))
    assert type(raised) is RuntimeError
    assert isinstance(report, str)
    assert report.startswith("RuntimeError: boom")


def __test_coded_failure_without_pipe_keeps_its_traceback__(monkeypatch: pytest.MonkeyPatch):
    """With no pipe to hand the error over, the exception leaves as it is"""
    def failing_main(*_args, **_kwargs):
        raise ProviderUnavailableError(provider="ccxt", why="invalid interval")

    monkeypatch.setattr(security_process, 'security_process_main', failing_main)
    with pytest.raises(ProviderUnavailableError):
        security_process.run_security_process(failure_conn=None)
