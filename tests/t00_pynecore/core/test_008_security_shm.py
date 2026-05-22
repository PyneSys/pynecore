"""
@pyne
"""
import pickle
import multiprocessing as mp

import pytest

from pynecore.core.security_shm import (
    SyncBlock, ResultBlock, ResultReader, INITIAL_RESULT_SIZE,
    write_result, write_na,
)

# The concurrency stress test below uses module-local target functions, which
# are not importable in spawn-mode children (the tests/ directory isn't on
# sys.path inside the spawned interpreter). Fork mode reuses the parent's
# import state, sidestepping that. Windows has no fork start method, so the
# test is skipped there — the runtime fix itself is platform-independent and
# `security_process_main` is reachable via spawn in production because it
# lives in the installed pynecore package.
_FORK_AVAILABLE = 'fork' in mp.get_all_start_methods()


def __test_sync_block_basic__(log):
    """SyncBlock creation, read/write of slot fields"""
    sec_ids = ["sec_a", "sec_b"]
    sb = SyncBlock(sec_ids)

    try:
        # Initial values should be zero
        ts, ver, rsize, target, flags = sb.get_slot("sec_a")
        assert ts == 0
        assert ver == 0
        assert rsize == 0
        assert target == 0
        assert flags == 0

        # Set and read target_time
        sb.set_target_time("sec_a", 1000000)
        assert sb.get_target_time("sec_a") == 1000000
        assert sb.get_target_time("sec_b") == 0  # other slot unchanged

        # Set and read timestamp
        sb.set_timestamp("sec_b", 2000000)
        assert sb.get_timestamp("sec_b") == 2000000
        assert sb.get_timestamp("sec_a") == 0

        # Set and read result meta
        sb.set_result_meta("sec_a", 3, 128)
        ver, rsize = sb.get_result_meta("sec_a")
        assert ver == 3
        assert rsize == 128

        # Set and read flags
        sb.set_flags("sec_a", 0x01)
        assert sb.get_flags("sec_a") == 0x01
        assert sb.get_flags("sec_b") == 0x00
    finally:
        sb.close()
        sb.unlink()


def __test_result_block_write_read__(log):
    """ResultBlock write/read cycle with pickle data"""
    sec_ids = ["sec_test"]
    sb = SyncBlock(sec_ids)
    rb = ResultBlock("sec_test", create=True, version=0)

    try:
        # Write a float value
        data = pickle.dumps(42.5, protocol=pickle.HIGHEST_PROTOCOL)
        rb.write(data, sb)

        ver, rsize = sb.get_result_meta("sec_test")
        assert ver == 0
        assert rsize == len(data)

        raw = rb.read(rsize)
        assert pickle.loads(raw) == 42.5

        # Write a complex value
        complex_val = {"close": 100.0, "sma": [1.0, 2.0, 3.0]}
        data2 = pickle.dumps(complex_val, protocol=pickle.HIGHEST_PROTOCOL)
        rb.write(data2, sb)

        ver2, rsize2 = sb.get_result_meta("sec_test")
        assert ver2 == 0  # no reallocation needed
        raw2 = rb.read(rsize2)
        assert pickle.loads(raw2) == complex_val
    finally:
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def __test_result_block_reallocation__(log):
    """ResultBlock reallocates when data exceeds OS-allocated block size"""
    sec_ids = ["sec_realloc"]
    sb = SyncBlock(sec_ids)
    rb = ResultBlock("sec_realloc", create=True, version=0, size=64)

    try:
        # Small write fits (OS allocates at least one page, typically 4-16 KB)
        small = pickle.dumps(1.0, protocol=pickle.HIGHEST_PROTOCOL)
        rb.write(small, sb)
        assert rb.version == 0

        # Generate data larger than the OS-allocated block
        os_block_size = rb.size  # actual OS allocation (e.g., 16384 on macOS)
        large_val = list(range(100000))
        large = pickle.dumps(large_val, protocol=pickle.HIGHEST_PROTOCOL)
        assert len(large) > os_block_size, (
            f"Test data ({len(large)} bytes) must exceed OS block size ({os_block_size} bytes)"
        )

        rb.write(large, sb)
        assert rb.version == 1  # version incremented
        assert rb.size >= len(large)

        ver, rsize = sb.get_result_meta("sec_realloc")
        assert ver == 1
        raw = rb.read(rsize)
        assert pickle.loads(raw) == large_val
    finally:
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def __test_result_reader__(log):
    """ResultReader tracks version and re-attaches on reallocation"""
    sec_ids = ["sec_reader"]
    sb = SyncBlock(sec_ids)
    rb = ResultBlock("sec_reader", create=True, version=0)
    rr = ResultReader("sec_reader")

    try:
        # No data yet — returns default
        result = rr.read(sb, default="NA")
        assert result == "NA"

        # Write data
        write_result(rb, sb, 3.14)

        # Reader should see the value
        result = rr.read(sb)
        assert result == 3.14

        # Overwrite with new value
        write_result(rb, sb, 2.71)
        result = rr.read(sb)
        assert result == 2.71
    finally:
        rr.close()
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def __test_write_na__(log):
    """write_na sets result_size to 0, reader returns default"""
    sec_ids = ["sec_na"]
    sb = SyncBlock(sec_ids)
    rb = ResultBlock("sec_na", create=True, version=0)
    rr = ResultReader("sec_na")

    try:
        # Write a real value first
        write_result(rb, sb, 100.0)
        assert rr.read(sb) == 100.0

        # Now write na
        write_na(rb, sb)
        result = rr.read(sb, default="NA")
        assert result == "NA"
    finally:
        rr.close()
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def _stress_writer(sec_id, sync_block_name, lock, n_iterations, error_queue):
    """Writer subprocess: cycles through small payload, large payload, and write_na
    while holding ``lock`` for each publication. Mirrors the security child's
    write surface (write_result + write_na) under contention.
    """
    try:
        sb = SyncBlock([sec_id], create=False, name=sync_block_name)
        rb = ResultBlock(sec_id, create=False, version=0)
        small_template = ("small", 0)
        large_template = ("large", 0, list(range(200)))
        for i in range(n_iterations):
            with lock:
                phase = i % 3
                if phase == 0:
                    write_result(rb, sb, ("small", i))
                elif phase == 1:
                    write_result(rb, sb, ("large", i, list(range(200))))
                else:
                    write_na(rb, sb)
        rb.close()
        sb.close()
        # touch templates so linters don't drop them
        _ = small_template, large_template
    except Exception as exc:  # pragma: no cover - propagated to parent
        error_queue.put(("writer", repr(exc)))


def _stress_reader(sec_id, sync_block_name, lock, stop_event, result_queue, error_queue):
    """Reader subprocess: spins on cross-context-style reads under ``lock``
    until ``stop_event`` is set. Records counts of each payload kind so the
    parent can sanity-check that no torn / wrong-shape values escape.
    """
    try:
        sb = SyncBlock([sec_id], create=False, name=sync_block_name)
        rr = ResultReader(sec_id)
        n_reads = 0
        n_default = 0
        n_small = 0
        n_large = 0
        while not stop_event.is_set():
            with lock:
                val = rr.read(sb, default=None)
            n_reads += 1
            if val is None:
                n_default += 1
                continue
            if not isinstance(val, tuple) or len(val) < 2:
                error_queue.put(("reader", f"malformed value: {val!r}"))
                break
            tag = val[0]
            if tag == "small" and len(val) == 2 and isinstance(val[1], int):
                n_small += 1
            elif (tag == "large" and len(val) == 3
                    and isinstance(val[1], int) and isinstance(val[2], list)
                    and len(val[2]) == 200):
                n_large += 1
            else:
                error_queue.put(("reader", f"unexpected value: {val!r}"))
                break
        rr.close()
        sb.close()
        result_queue.put((n_reads, n_default, n_small, n_large))
    except Exception as exc:  # pragma: no cover - propagated to parent
        error_queue.put(("reader", repr(exc)))


@pytest.mark.skipif(
    not _FORK_AVAILABLE,
    reason="stress test target functions are module-local; spawn-mode children "
           "cannot import them. Fork start method required.",
)
def __test_concurrent_lock_protects_cross_context__(log):
    """Two processes hammer the same slot concurrently under a shared
    ``multiprocessing.Lock``: a writer cycles through small/large/na publications
    and a reader spins on `ResultReader.read`. Mirrors the security-child
    cross-context read path. With the per-slot lock in place, every read must
    return either ``default`` or a structurally valid payload — no
    ``UnpicklingError``, no malformed tuples, no garbage.
    """
    sec_id = "sec_stress"
    sync_block_name = "pyne_sync_stress"
    sb = SyncBlock([sec_id], create=True, name=sync_block_name)
    rb = ResultBlock(sec_id, create=True, version=0)

    # Use fork context so the test-local target functions are reachable in the
    # children without needing the tests/ directory on sys.path.
    ctx = mp.get_context("fork")
    lock = ctx.Lock()
    stop_event = ctx.Event()
    error_queue = ctx.Queue()
    result_queue = ctx.Queue()

    n_iterations = 1500
    writer = ctx.Process(
        target=_stress_writer,
        args=(sec_id, sync_block_name, lock, n_iterations, error_queue),
        daemon=True,
    )
    reader = ctx.Process(
        target=_stress_reader,
        args=(sec_id, sync_block_name, lock, stop_event, result_queue, error_queue),
        daemon=True,
    )

    try:
        reader.start()
        writer.start()
        writer.join(timeout=30)
        assert not writer.is_alive(), "writer did not finish in time"
        stop_event.set()
        reader.join(timeout=10)
        assert not reader.is_alive(), "reader did not finish in time"

        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get_nowait())
        assert not errors, f"subprocess errors: {errors}"

        assert not result_queue.empty(), "reader produced no result"
        n_reads, n_default, n_small, n_large = result_queue.get_nowait()
        log.info(
            "stress: reads=%d default=%d small=%d large=%d",
            n_reads, n_default, n_small, n_large,
        )
        # Reader must observe at least one of each non-na payload kind to prove
        # the contention path was actually exercised.
        assert n_reads > 0, "reader did not read anything"
        assert n_small > 0, "reader never observed a small payload"
        assert n_large > 0, "reader never observed a large payload"
    finally:
        if writer.is_alive():
            writer.terminate()
        if reader.is_alive():
            reader.terminate()
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def __test_developing_bar_roundtrip__(log):
    """SyncBlock developing-bar slot writes and reads back exactly."""
    from pynecore.core.security_shm import FLAG_IS_DEVELOPING

    sec_ids = ["sec_0", "sec_1"]
    sb = SyncBlock(sec_ids)
    try:
        # Defaults: all zeros
        o, h, l, c, v, t = sb.get_developing_bar("sec_0")
        assert (o, h, l, c, v, t) == (0.0, 0.0, 0.0, 0.0, 0.0, 0)

        sb.set_developing_bar(
            "sec_0",
            dev_open=1.1, dev_high=1.5, dev_low=1.0,
            dev_close=1.3, dev_volume=12345.0, dev_time=1_700_000_000_000,
        )
        sb.set_flags("sec_0", FLAG_IS_DEVELOPING)

        o, h, l, c, v, t = sb.get_developing_bar("sec_0")
        assert (o, h, l, c) == (1.1, 1.5, 1.0, 1.3)
        assert v == 12345.0
        assert t == 1_700_000_000_000
        assert sb.get_flags("sec_0") == FLAG_IS_DEVELOPING

        # Other slot is independent
        o2, h2, l2, c2, v2, t2 = sb.get_developing_bar("sec_1")
        assert (o2, h2, l2, c2, v2, t2) == (0.0, 0.0, 0.0, 0.0, 0.0, 0)
        assert sb.get_flags("sec_1") == 0
    finally:
        sb.close()
        sb.unlink()


def __test_developing_bar_does_not_alias_target_time__(log):
    """Writing developing-bar fields must not clobber target_time / result meta."""
    sec_ids = ["sec_0"]
    sb = SyncBlock(sec_ids)
    try:
        sb.set_target_time("sec_0", 9_999_999)
        sb.set_result_meta("sec_0", version=7, result_size=123)
        sb.set_developing_bar(
            "sec_0",
            dev_open=2.0, dev_high=3.0, dev_low=1.0,
            dev_close=2.5, dev_volume=100.0, dev_time=42,
        )
        assert sb.get_target_time("sec_0") == 9_999_999
        ver, sz = sb.get_result_meta("sec_0")
        assert (ver, sz) == (7, 123)
    finally:
        sb.close()
        sb.unlink()


def __test_multiple_slots_independent__(log):
    """Multiple security slots are independent in the SyncBlock"""
    sec_ids = ["sec_0", "sec_1", "sec_2"]
    sb = SyncBlock(sec_ids)
    blocks = {sid: ResultBlock(sid, create=True, version=0) for sid in sec_ids}
    readers = {sid: ResultReader(sid) for sid in sec_ids}

    try:
        # Write different values to each
        write_result(blocks["sec_0"], sb, "hello")
        write_result(blocks["sec_1"], sb, 42)
        write_result(blocks["sec_2"], sb, [1, 2, 3])

        # Read each independently
        assert readers["sec_0"].read(sb) == "hello"
        assert readers["sec_1"].read(sb) == 42
        assert readers["sec_2"].read(sb) == [1, 2, 3]

        # Update one, others unchanged
        write_result(blocks["sec_1"], sb, 99)
        assert readers["sec_0"].read(sb) == "hello"
        assert readers["sec_1"].read(sb) == 99
        assert readers["sec_2"].read(sb) == [1, 2, 3]
    finally:
        for r in readers.values():
            r.close()
        for b in blocks.values():
            b.close()
            b.unlink()
        sb.close()
        sb.unlink()
