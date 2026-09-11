"""
@pyne
"""
import multiprocessing as mp

import pytest

from pynecore.core.security_shm import (
    FRONTIER_INF, RingBlock, RingReader, RingWriter, SyncBlock, create_ring_conditions,
)

# The spawned-child test below uses a module-local target function, which is
# not importable in spawn-mode children (the tests/ directory isn't on
# sys.path inside the spawned interpreter). Fork mode reuses the parent's
# import state, sidestepping that. Windows has no fork start method, so the
# test is skipped there — the ring itself is platform-independent and the
# production children reach `security_process_main` through the installed
# pynecore package.
_FORK_AVAILABLE = 'fork' in mp.get_all_start_methods()


def _make(sec_ids: list[str]) -> tuple[SyncBlock, dict]:
    """Create a SyncBlock plus one Condition per security id."""
    return SyncBlock(sec_ids), create_ring_conditions(sec_ids)


def __test_ring_append_and_read_by_close__(log):
    """Entries are appended in close order and looked up by as-of instant."""
    sb, conds = _make(["prod", "cons"])
    w = RingWriter("prod", sb, conds["prod"], capacity=8)
    r = RingReader("prod", sb, conds["prod"])
    try:
        assert r.last_close_at_or_before(1000) is None
        assert r.count() == 0

        w.append(0, 100, "a", frontier_close=199)
        w.append(100, 200, "b", frontier_close=299)
        w.append(200, 300, ["c", 1, 2], frontier_close=399)

        assert r.count() == 3
        assert r.last_close_at_or_before(99) is None
        assert r.last_close_at_or_before(100) == (0, 100, "a")
        assert r.last_close_at_or_before(150) == (0, 100, "a")
        assert r.last_close_at_or_before(300) == (200, 300, ["c", 1, 2])
        assert r.last_close_at_or_before(10_000) == (200, 300, ["c", 1, 2])

        assert r.has_close(200) is True
        assert r.has_close(250) is False

        # start < close <= end
        assert [e[1] for e in r.range_by_close(100, 300)] == [200, 300]
        assert [e[1] for e in r.range_by_close(0, 300)] == [100, 200, 300]
        assert r.range_by_close(300, 400) == []
    finally:
        r.close()
        w.close()
        w.unlink()
        sb.close()
        sb.unlink()


def __test_ring_frontier_semantics__(log):
    """frontier_close rises monotonically; finish() publishes the +inf sentinel."""
    sb, conds = _make(["prod"])
    w = RingWriter("prod", sb, conds["prod"])
    r = RingReader("prod", sb, conds["prod"])
    try:
        assert r.frontier_close() == 0

        w.append(0, 100, 1.0, frontier_close=199)
        assert r.frontier_close() == 199

        # A lower frontier never moves it back
        w.set_frontier_close(50)
        assert r.frontier_close() == 199
        w.append(100, 200, 2.0, frontier_close=100)
        assert r.frontier_close() == 199

        w.set_frontier_close(299)
        assert r.frontier_close() == 299

        # Waiting is satisfied either by an exact close or by the frontier
        assert r.wait_for_close(200, timeout=1.0) is True
        assert r.wait_for_close(250, timeout=1.0) is True
        assert r.wait_for_close(500, timeout=0.3) is False

        w.finish()
        assert r.frontier_close() == FRONTIER_INF
        assert r.wait_for_close(10 ** 15, timeout=1.0) is True
    finally:
        r.close()
        w.close()
        w.unlink()
        sb.close()
        sb.unlink()


def __test_ring_gc_keeps_last_entry_at_or_below_watermark__(log):
    """A full ring drops consumed entries but keeps the last one <= watermark."""
    sb, conds = _make(["gcprod", "gccons"])
    prod_idx = sb.index_of("gcprod")
    cons_idx = sb.index_of("gccons")
    w = RingWriter("gcprod", sb, conds["gcprod"], capacity=4)
    r = RingReader("gcprod", sb, conds["gcprod"])
    try:
        sb.set_watermark(cons_idx, prod_idx, 300)
        assert sb.min_watermark(prod_idx, [cons_idx]) == 300

        for i in range(4):
            w.append(i * 100, (i + 1) * 100, f"v{i}", consumer_indexes=[cons_idx])
        assert r.count() == 4
        version_before = sb.get_ring_version("gcprod")

        # Ring is full -> GC runs; closes 100 and 200 are below the watermark,
        # 300 is the last entry at or below it and must survive.
        w.append(400, 500, "v4", consumer_indexes=[cons_idx])
        assert sb.get_ring_version("gcprod") == version_before, "GC must not grow the ring"
        assert [e[1] for e in r.range_by_close(0, 10 ** 9)] == [300, 400, 500]

        # The kept entry still answers an as-of above every dropped close
        assert r.last_close_at_or_before(350) == (200, 300, "v2")
        log.info("after gc: count=%d", r.count())
    finally:
        r.close()
        w.close()
        w.unlink()
        sb.close()
        sb.unlink()


def __test_ring_growth_bumps_version_and_reader_reopens__(log):
    """A ring nothing can be GC'd from doubles, bumping the version."""
    sb, conds = _make(["growprod", "growcons"])
    cons_idx = sb.index_of("growcons")
    w = RingWriter("growprod", sb, conds["growprod"], capacity=4, arena_size=256)
    r = RingReader("growprod", sb, conds["growprod"])
    try:
        # Watermark 0: nothing is consumed, so the GC can free nothing
        for i in range(4):
            w.append(i * 100, (i + 1) * 100, f"v{i}", consumer_indexes=[cons_idx])
        assert r.last_close_at_or_before(400) == (300, 400, "v3")
        assert r.version == 0
        assert sb.get_ring_version("growprod") == 0

        w.append(400, 500, "v4", consumer_indexes=[cons_idx])
        assert sb.get_ring_version("growprod") == 1
        assert w.capacity == 8

        # The reader notices the version change and re-attaches
        assert [e[1] for e in r.range_by_close(0, 10 ** 9)] == [100, 200, 300, 400, 500]
        assert r.version == 1
        assert r.last_close_at_or_before(500) == (400, 500, "v4")

        # A payload larger than the whole arena also forces growth
        big = list(range(5000))
        w.append(500, 600, big, consumer_indexes=[cons_idx])
        assert sb.get_ring_version("growprod") == 2
        assert r.last_close_at_or_before(600) == (500, 600, big)
        assert r.last_close_at_or_before(100) == (0, 100, "v0")
    finally:
        r.close()
        w.close()
        w.unlink()
        sb.close()
        sb.unlink()


def __test_watermark_matrix__(log):
    """Watermark matrix get/set by index pair and the per-producer minimum."""
    sb, _ = _make(["a", "b", "c"])
    try:
        ia, ib, ic = sb.index_of("a"), sb.index_of("b"), sb.index_of("c")

        # Every cell starts at zero and no consumer at all means +inf
        assert sb.get_watermark(ia, ib) == 0
        assert sb.min_watermark(ic, []) == FRONTIER_INF

        sb.set_watermark(ia, ic, 500)
        sb.set_watermark(ib, ic, 300)
        assert sb.get_watermark(ia, ic) == 500
        assert sb.get_watermark(ib, ic) == 300
        assert sb.min_watermark(ic, [ia, ib]) == 300
        assert sb.min_watermark(ic, [ia]) == 500

        # Rows are independent: a's watermark for b is untouched
        assert sb.get_watermark(ia, ib) == 0
        assert sb.min_watermark(ib, [ia, ic]) == 0

        # A finished consumer needs nothing any more
        sb.set_watermark(ib, ic, FRONTIER_INF)
        assert sb.min_watermark(ic, [ia, ib]) == 500

        with pytest.raises(IndexError):
            sb.get_watermark(0, 3)
    finally:
        sb.close()
        sb.unlink()


def __test_round_context_and_rounds_done__(log):
    """Round context fields and the rounds_done counter are per slot."""
    sb, _ = _make(["r0", "r1"])
    try:
        assert sb.get_round_context("r0") == (0, 0)
        assert sb.get_rounds_done("r0") == 0

        sb.set_round_context("r0", 1_700_000_000_000, 1_700_000_600_000)
        assert sb.get_round_context("r0") == (1_700_000_000_000, 1_700_000_600_000)
        assert sb.get_round_context("r1") == (0, 0)

        assert sb.increment_rounds_done("r0") == 1
        assert sb.increment_rounds_done("r0") == 2
        assert sb.get_rounds_done("r0") == 2
        assert sb.get_rounds_done("r1") == 0

        sb.set_rounds_done("r1", 7)
        assert sb.get_rounds_done("r1") == 7

        # The new fields must not alias the existing slot fields
        sb.set_target_time("r0", 12345)
        sb.set_result_meta("r0", version=3, result_size=99)
        sb.set_flags("r0", 0x21)
        sb.set_ltf_period_start("r0", 777)
        sb.set_frontier_close("r0", 888)
        sb.set_ring_state("r0", count=4, head=2, used=64)
        sb.set_ring_version("r0", 5)
        assert sb.get_slot("r0") == (3, 99, 12345, 0x21)
        assert sb.get_ltf_period_start("r0") == 777
        assert sb.get_frontier_close("r0") == 888
        assert sb.get_ring_state("r0") == (4, 2, 64)
        assert sb.get_ring_version("r0") == 5
        assert sb.get_round_context("r0") == (1_700_000_000_000, 1_700_000_600_000)
        assert sb.get_rounds_done("r0") == 2
    finally:
        sb.close()
        sb.unlink()


def _child_appender(sec_id, sync_name, sec_ids, condition, n, error_queue):
    """Child process: appends ``n`` bars, then finishes the ring."""
    try:
        sb = SyncBlock(sec_ids, create=False, name=sync_name)
        w = RingWriter(sec_id, sb, condition, capacity=4, arena_size=64)
        for i in range(n):
            w.append(i * 100, (i + 1) * 100, ("bar", i), frontier_close=(i + 1) * 100 + 99)
        sb.increment_rounds_done(sec_id)
        w.finish()
        w.close()
        sb.close()
    except Exception as exc:  # pragma: no cover - propagated to the parent
        error_queue.put(repr(exc))


@pytest.mark.skipif(
    not _FORK_AVAILABLE,
    reason="the child target function is module-local; spawn-mode children "
           "cannot import it. Fork start method required.",
)
def __test_ring_child_process_append_parent_waits__(log):
    """A real child appends while the parent blocks on the per-sid Condition."""
    sec_ids = ["childprod"]
    sync_name = "pyne_sync_ring_child"
    sb = SyncBlock(sec_ids, create=True, name=sync_name)
    conds = create_ring_conditions(sec_ids)
    cond = conds["childprod"]

    ctx = mp.get_context("fork")
    error_queue = ctx.Queue()
    n = 40
    proc = ctx.Process(
        target=_child_appender,
        args=("childprod", sync_name, sec_ids, cond, n, error_queue),
        daemon=True,
    )
    reader = RingReader("childprod", sb, cond)
    try:
        proc.start()

        # Block on the condition until the last bar's close is published —
        # no sleep-based polling anywhere.
        assert reader.wait_for_close(n * 100, timeout=30.0), "child never published"

        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get_nowait())
        assert not errors, f"child errors: {errors}"

        assert reader.last_close_at_or_before(n * 100) == ((n - 1) * 100, n * 100, ("bar", n - 1))
        # The ring grew and/or was compacted in the child; the parent follows
        assert reader.version >= 1
        assert sb.get_rounds_done("childprod") == 1

        proc.join(timeout=10)
        assert not proc.is_alive(), "child did not exit"
        assert reader.frontier_close() == FRONTIER_INF
        log.info("child ring: count=%d version=%d", reader.count(), reader.version)
    finally:
        if proc.is_alive():
            proc.terminate()
        last_version = sb.get_ring_version("childprod")
        reader.close()
        stale = RingBlock("childprod", create=False, version=last_version,
                          prefix=sb.block_prefix("childprod"))
        stale.close()
        stale.unlink()
        sb.close()
        sb.unlink()
