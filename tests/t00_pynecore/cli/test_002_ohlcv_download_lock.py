"""
Regression tests for :func:`pynecore.cli.commands.run._ohlcv_download_lock`.

Two concurrent runs on the same ``(provider, symbol, timeframe)`` share one
``.ohlcv`` file. The warmup download ``seek(0)``-truncates and rewrites it; a
second process reading (or truncating) at the same instant would see a
half-written or empty file. The advisory ``fcntl.flock`` on a sidecar ``.lock``
serializes the download so a concurrent run waits (kernel wait, no sleep loop)
and then reads a complete file.

The lock is per open file description, so two separate ``_ohlcv_download_lock``
context managers in the SAME process contend — the test drives that
contention with two threads, fully offline.
"""
import threading
from types import SimpleNamespace

from pynecore.cli.commands.run import (
    _atomic_ohlcv_download_target,
    _ohlcv_download_lock,
)
from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
from pynecore.types.ohlcv import OHLCV


def __test_ohlcv_download_lock_serializes_concurrent_writers__(tmp_path):
    """A second holder must block until the first releases the flock."""
    ohlcv_path = tmp_path / "prov_SYM_1.ohlcv"
    ohlcv_path.write_bytes(b"")

    a_holding = threading.Event()
    release_a = threading.Event()
    b_acquired = threading.Event()

    def worker_a():
        with _ohlcv_download_lock(ohlcv_path):
            a_holding.set()
            # Hold the lock until the main thread has verified B is blocked.
            release_a.wait(timeout=5.0)

    def worker_b():
        # Only contend once A actually holds the lock.
        a_holding.wait(timeout=5.0)
        with _ohlcv_download_lock(ohlcv_path):
            b_acquired.set()

    ta = threading.Thread(target=worker_a)
    tb = threading.Thread(target=worker_b)
    ta.start()
    assert a_holding.wait(timeout=5.0), "worker A never acquired the lock"
    tb.start()

    # B must NOT get the lock while A still holds it. ``Event.wait`` with a
    # timeout is an event-driven bounded wait (no busy poll); it returns False
    # because the event stays unset for the whole window.
    assert not b_acquired.wait(timeout=0.3), (
        "second writer entered the download lock while the first still held it"
    )

    # Release A; B must then acquire promptly.
    release_a.set()
    assert b_acquired.wait(timeout=5.0), (
        "second writer never acquired the lock after the first released it"
    )

    ta.join(timeout=5.0)
    tb.join(timeout=5.0)
    assert not ta.is_alive() and not tb.is_alive()


def __test_ohlcv_download_lock_creates_sidecar_lock_file__(tmp_path):
    """The lock uses a ``.ohlcv.lock`` sidecar next to the data file."""
    ohlcv_path = tmp_path / "prov_SYM_1.ohlcv"
    ohlcv_path.write_bytes(b"")
    lock_path = ohlcv_path.with_suffix(ohlcv_path.suffix + ".lock")
    assert not lock_path.exists()
    with _ohlcv_download_lock(ohlcv_path):
        assert lock_path.exists(), "sidecar .lock file must be created"


def __test_ohlcv_download_lock_noop_for_none_path__():
    """A live/in-memory feed with no shared file is a clean no-op."""
    with _ohlcv_download_lock(None):
        pass


def __test_atomic_download_keeps_canonical_file_complete_during_rewrite__(
        tmp_path,
):
    """A reader sees the prior complete file while a sibling rewrites privately."""
    path = tmp_path / "prov_SYM_1.ohlcv"
    provider = SimpleNamespace(ohlcv_path=path, ohlcv_file=OHLCVWriter(path))
    first = OHLCV(1_700_000_000, 1.0, 2.0, 0.5, 1.5, 10.0)
    second = OHLCV(1_700_000_060, 1.5, 2.5, 1.0, 2.0, 11.0)
    with _atomic_ohlcv_download_target(provider):
        with provider.ohlcv_file as writer:
            writer.write(first)

    private_truncated = threading.Event()
    allow_publish = threading.Event()

    def rewrite():
        sibling = SimpleNamespace(
            ohlcv_path=path, ohlcv_file=OHLCVWriter(path)
        )
        with _atomic_ohlcv_download_target(sibling):
            with sibling.ohlcv_file as writer:
                writer.seek(0)
                writer.truncate()
                private_truncated.set()
                assert allow_publish.wait(timeout=5.0)
                writer.write(second)

    worker = threading.Thread(target=rewrite)
    worker.start()
    assert private_truncated.wait(timeout=5.0)
    with OHLCVReader(path) as reader:
        assert reader.end_timestamp == first.timestamp
    allow_publish.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    with OHLCVReader(path) as reader:
        assert reader.end_timestamp == second.timestamp
