"""
Unit tests for CSVWriter's flush / close acknowledgement paths
(core/csv_file.py).
"""
import threading
from pathlib import Path

import pytest

from pynecore.core import csv_file
from pynecore.core.csv_file import CSVWriter
from pynecore.types.ohlcv import OHLCV


class _BlockingFile:
    """Text-file stand-in whose ``write`` blocks from the ``block_after``-th call on."""

    def __init__(self, block_after: int = 0):
        self.written: list[str] = []
        self._passthrough = block_after
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def write(self, data: str) -> int:
        if self._passthrough > 0:
            self._passthrough -= 1
        else:
            self.entered.set()
            self.release.wait(5)
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FailingFile:
    """Text-file stand-in whose ``write`` fails after ``fail_after`` calls."""

    def __init__(self, fail_after: int = 0):
        self.written: list[str] = []
        self._remaining = fail_after
        self.closed = False

    def write(self, data: str) -> int:
        if self._remaining <= 0:
            raise OSError("disk full")
        self._remaining -= 1
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def __test_flush_reaches_the_file__(tmp_path: Path):
    """flush() must not return while rows are still in the memory buffer"""
    path = tmp_path / "out.csv"
    # An idle timeout far beyond the test keeps the idle drain out of the
    # picture: only the explicit FLUSH may move the row into the file
    writer = CSVWriter(path, idle_time=30.0)
    writer.open()
    try:
        assert writer.write_dict({'a': 1, 'b': 2})
        assert writer.flush()
        assert path.read_text() == "a,b\n1,2\n"
    finally:
        writer.close()


def __test_flush_raises_on_write_failure__(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A failing write must surface on flush() instead of blocking forever"""
    failing = _FailingFile()
    monkeypatch.setattr(csv_file, 'open', lambda *_a, **_kw: failing, raising=False)

    writer = CSVWriter(tmp_path / "out.csv", idle_time=30.0)
    writer.open()
    assert writer.write_dict({'a': 1})
    with pytest.raises(RuntimeError, match="disk full"):
        writer.flush()
    with pytest.raises(RuntimeError, match="disk full"):
        writer.close()


def __test_close_raises_when_final_drain_fails__(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    """Rows written out only by the post-STOP drain must not fail silently"""
    failing = _FailingFile()
    monkeypatch.setattr(csv_file, 'open', lambda *_a, **_kw: failing, raising=False)

    writer = CSVWriter(tmp_path / "out.csv", idle_time=30.0)
    writer.open()
    # No explicit flush: the row stays buffered until the worker's final drain
    assert writer.write_dict({'a': 1})
    with pytest.raises(RuntimeError, match="disk full"):
        writer.close()
    assert failing.closed


def __test_close_timeout_with_queued_records__(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A timed close() that cannot even queue the STOP must not report success"""
    # The header write goes through, the first buffer drain blocks
    blocking = _BlockingFile(block_after=1)
    monkeypatch.setattr(csv_file, 'open', lambda *_a, **_kw: blocking, raising=False)

    # A single queue slot and a drain after every row: the worker hangs in the
    # file write while the second record is still waiting in the queue
    writer = CSVWriter(tmp_path / "out.csv", queue_size=1, buffer_size=1,
                       idle_time=30.0, headers=('a',))
    writer.open()
    try:
        assert writer.write('row1')
        assert blocking.entered.wait(5)
        assert writer.write('row2')

        with pytest.raises(TimeoutError):
            writer.close(timeout=0.01)
        # Retryable: nothing was torn down under the still-running worker
        assert writer.is_open
        assert not blocking.closed
        # No STOP was queued, so the writer still accepts records — a full queue
        # is the only reason this one cannot be taken
        assert writer.write('row3', timeout=0) is False
    finally:
        blocking.release.set()
        writer.close()

    assert ''.join(blocking.written) == "a\nrow1\nrow2\n"


def __test_close_timeout_while_worker_writes__(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A worker that outlives the join timeout must not be declared closed"""
    blocking = _BlockingFile(block_after=1)
    monkeypatch.setattr(csv_file, 'open', lambda *_a, **_kw: blocking, raising=False)

    # Here the STOP fits in the queue, but the worker is stuck in the file write
    writer = CSVWriter(tmp_path / "out.csv", buffer_size=1, idle_time=30.0, headers=('a',))
    writer.open()
    try:
        assert writer.write('row1')
        assert blocking.entered.wait(5)

        with pytest.raises(TimeoutError):
            writer.close(timeout=0.01)
        assert writer.is_open
        assert not blocking.closed
    finally:
        blocking.release.set()
        writer.close()

    assert ''.join(blocking.written) == "a\nrow1\n"
    assert blocking.closed


def __test_no_writes_accepted_after_stop_was_queued__(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch):
    """Records queued behind the STOP are dropped, so they must not be accepted"""
    blocking = _BlockingFile(block_after=1)
    monkeypatch.setattr(csv_file, 'open', lambda *_a, **_kw: blocking, raising=False)

    writer = CSVWriter(tmp_path / "out.csv", buffer_size=1, idle_time=30.0, headers=('a',))
    writer.open()
    try:
        assert writer.write('row1')
        assert blocking.entered.wait(5)

        # The STOP is queued here, only the join times out
        with pytest.raises(TimeoutError):
            writer.close(timeout=0.01)

        # The worker exits on the STOP and discards whatever follows it, so
        # neither a write nor a flush may report success
        with pytest.raises(RuntimeError, match="closing"):
            writer.write('late')
        with pytest.raises(RuntimeError, match="closing"):
            writer.write_dict({'a': 'late'})
        with pytest.raises(RuntimeError, match="closing"):
            writer.write_ohlcv(OHLCV(0, 1.0, 1.0, 1.0, 1.0, 1.0))
        assert writer.flush(timeout=0.5) is False
    finally:
        blocking.release.set()
        writer.close()

    assert ''.join(blocking.written) == "a\nrow1\n"


def __test_integral_floats_are_written_without_a_fraction__(tmp_path: Path):
    """A Pine int is a float at runtime, the file still shows it as an integer"""
    path = tmp_path / "ints.csv"
    writer = CSVWriter(path, idle_time=30.0)
    writer.open()
    try:
        assert writer.write_dict({'bar_index': 14.0, 'volume': 2451.0, 'price': 1.5, 'na': float('nan')})
        assert writer.flush()
    finally:
        writer.close()
    assert path.read_text().splitlines()[1] == "14,2451,1.5,NaN"
