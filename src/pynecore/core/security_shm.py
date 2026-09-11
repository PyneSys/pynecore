"""
Shared memory layer for request.security() inter-process communication.

Three types of shared memory blocks:

1. **SyncBlock** — single fixed-size block containing metadata for all security
   slots (version, result_size, target_time, flags, ring state, round context
   per slot) plus an N×N consumer×producer watermark matrix.

2. **ResultBlock** — one per security ID, holds the pickled result value.
   Reallocated with doubled size when the pickle data outgrows the block.

3. **RingBlock** — one per security ID that has consumers, holds an append-only
   history of ``(open, close, value)`` entries ordered by close. Reallocated
   with doubled size (capacity and payload arena) when full; the version is
   embedded in the SharedMemory name so readers detect and re-attach.
"""
import pickle
import struct
from bisect import bisect_left, bisect_right
from multiprocessing import Condition
from multiprocessing.shared_memory import SharedMemory
from time import monotonic
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from multiprocessing.synchronize import Condition as ConditionType, Event as EventType

# Per-slot layout in the sync block:
#   offset 0:   int64   frontier_close  (8 bytes) — every bar of this producer
#                       whose scheduled close is <= this instant is already in
#                       the ring; FRONTIER_INF means "the producer is done"
#   offset 8:   uint32  version         (4 bytes) — result block version (incremented on realloc)
#   offset 12:  uint32  result_size     (4 bytes) — current pickle data size in bytes
#   offset 16:  int64   target_time     (8 bytes) — target time the process should advance to
#   offset 24:  uint8   flags           (1 byte)  — state flags
#   offset 25:  3 bytes pad (alignment to 28)
#   offset 28:  int32   ring_count      (4 bytes) — number of live ring entries
#   offset 32:  float64 dev_open        (8 bytes) — developing HTF bar OHLCV (lookahead_on live)
#   offset 40:  float64 dev_high        (8 bytes)
#   offset 48:  float64 dev_low         (8 bytes)
#   offset 56:  float64 dev_close       (8 bytes)
#   offset 64:  float64 dev_volume      (8 bytes)
#   offset 72:  int64   dev_time        (8 bytes) — developing HTF bar timestamp (ms);
#                       reused as period_end_exclusive (ms) by the live LTF-window
#                       path, which carries no pushed OHLCV (see FLAG_LTF_WINDOW)
#                       (value or na); cache key for the security-child
#                       cross-context read cache (version alone only tracks
#                       reallocation, not writes)
#   offset 84:  int32   ring_head       (4 bytes) — index of the first live ring entry
#   offset 88:  int64   ltf_period_start (8 bytes) — file-backed LTF rounds: the
#                       chart bar's period START (ms). The child runs every
#                       feed bar up to target_time for expression state, but
#                       only values written at/after this instant belong in
#                       the flushed intrabar array (TradingView arrays hold
#                       only the bar's OWN period).
#   offset 96:  int32   ring_version    (4 bytes) — ring block version (bumped on realloc)
#   offset 100: int32   ring_used       (4 bytes) — bytes used in the ring payload arena
#   offset 104: int64   round_tick      (8 bytes) — the chart round's fixed tick instant (ms)
#   offset 112: int64   round_sched_next_open (8 bytes) — scheduled session open closing the
#                       break that contains round_tick (0 = round_tick is inside an open
#                       session, or the symbol trades 24h)
#   offset 120: int64   rounds_done     (8 bytes) — rounds finished by the child
#
# Total per slot: 128 bytes.
#
# The slots are followed by an N×N int64 watermark matrix (row = consumer slot
# index, column = producer slot index): the oldest instant a consumer may still
# need from that producer. The ring GC drops entries below the minimum over the
# producer's consumers (keeping the last entry at or below it).
SLOT_FORMAT = '<IIqB'
SLOT_SIZE = 128
SLOT_DATA_SIZE = struct.calcsize(SLOT_FORMAT)  # 17 bytes — original fields only
_FRONTIER_OFFSET = 0
_RESULT_META_OFFSET = 8
_TARGET_TIME_OFFSET = 16
_FLAGS_OFFSET = 24
_RING_COUNT_OFFSET = 28
_RING_HEAD_OFFSET = 84
_LTF_PERIOD_START_OFFSET = 88
_RING_VERSION_OFFSET = 96
_RING_USED_OFFSET = 100
_ROUND_TICK_OFFSET = 104
_ROUND_SCHED_NEXT_OPEN_OFFSET = 112
_ROUNDS_DONE_OFFSET = 120

_WATERMARK_ITEM_SIZE = 8

# Sentinel for "+infinity" in millisecond time fields (frontier_close and
# watermarks). Large enough to be unreachable as a real timestamp, small enough
# to stay well inside int64.
FRONTIER_INF = 2 ** 62

# Offset of the developing-bar block within a slot.
_DEV_OHLCV_OFFSET = 32
_DEV_OHLCV_FORMAT = '<dddddq'  # open, high, low, close, volume, time(ms)

# Flag bits
FLAG_HAS_DATA = 0x01  # result block contains valid data
FLAG_SAME_CONTEXT = 0x02  # same symbol + TF as chart (no process needed)
FLAG_IS_DEVELOPING = 0x04  # current target_time refers to a developing (open) HTF bar
FLAG_CLOSED_OVERRIDE = 0x08  # closed-bar OHLCV is supplied via SyncBlock (live mode);
                             # subprocess must use SyncBlock OHLCV, not the .ohlcv file
FLAG_LTF_WINDOW = 0x10  # live request.security_lower_tf window round: subprocess pulls
                        # intrabars from its own LTF streamer (not pushed OHLCV);
                        # target_time = period_start, dev_time = period_end_exclusive
FLAG_LTF_CHART_DEVELOPING = 0x20  # within an LTF-window round, the chart bar is still
                                  # developing (keep the developing tail); clear means
                                  # the chart bar has closed (finalize, publish full period)
FLAG_LTF_LIVE_PHASE = 0x40  # within an LTF-window round, the chart has crossed the
                            # warmup->live transition (barstate realtime, not history)
FLAG_DEV_HISTORICAL = 0x80  # the pushed developing/closed OHLCV was aggregated on a
                            # HISTORICAL chart bar (backtest or live warmup), so the
                            # subprocess must keep history barstate instead of the
                            # realtime phase the live transport implies


def is_ltf_window(flags: int) -> bool:
    """Whether a slot is in a live LTF-window round (see :data:`FLAG_LTF_WINDOW`)."""
    return bool(flags & FLAG_LTF_WINDOW)


def is_ltf_chart_developing(flags: int) -> bool:
    """Whether the chart bar of an LTF-window round is still developing."""
    return bool(flags & FLAG_LTF_CHART_DEVELOPING)


def is_ltf_live_phase(flags: int) -> bool:
    """Whether an LTF-window round is past the warmup->live transition."""
    return bool(flags & FLAG_LTF_LIVE_PHASE)


# Initial result block size
INITIAL_RESULT_SIZE = 4096


def _result_block_name(prefix: str, version: int) -> str:
    """Generate SharedMemory name for a result block.

    :param prefix: The block's run-unique prefix (see
        :meth:`SyncBlock.block_prefix`).
    :param version: Reallocation counter, embedded so readers re-attach.
    """
    return f"{prefix}v{version}"


class SyncBlock:
    """
    Fixed-size shared memory block containing sync metadata for all security slots.

    Layout: N consecutive slots of :data:`SLOT_SIZE` bytes each, followed by an
    N×N int64 watermark matrix (row = consumer slot index, column = producer
    slot index).
    """

    def __init__(self, sec_ids: list[str], *, create: bool = True, name: str | None = None):
        self._sec_ids = list(sec_ids)
        self._index = {sid: i for i, sid in enumerate(sec_ids)}
        n = len(sec_ids)
        self._n = n
        self._watermark_offset = SLOT_SIZE * n
        total_size = max(SLOT_SIZE * n + _WATERMARK_ITEM_SIZE * n * n, 1)

        if create:
            self._shm = SharedMemory(
                name=name, create=True, size=total_size
            )
        else:
            self._shm = SharedMemory(name=name, create=False)

        buf = self._shm.buf
        assert buf is not None
        self._buf: memoryview = buf

        if create:
            self._buf[:total_size] = b'\x00' * total_size

    @property
    def name(self) -> str:
        return self._shm.name

    def block_prefix(self, sec_id: str) -> str:
        """This RUN's unique name prefix for one security's result/ring blocks.

        The SyncBlock is the one segment whose name the OS generates (and it is
        unique among every live segment on the machine), and the parent hands
        that name to every child — so it is the run identity, used whole. The
        security is named by its SLOT INDEX rather than by its compile-time
        sec_id: the id is the same string in two concurrent runs of the same
        script, and spelling it out would also blow the macOS 31-character
        POSIX shared memory name limit for a long synthetic id such as
        ``__auto_rate_EUR_USD__``. The index is as consistent across processes
        as the slot addressing itself, which already relies on it.

        :param sec_id: The security context id.
        :return: The prefix every block name of this sid is built from.
        """
        return f"{self._shm.name}_{self._index[sec_id]}"

    @property
    def sec_ids(self) -> list[str]:
        return list(self._sec_ids)

    def index_of(self, sec_id: str) -> int:
        """Return the slot index of a security id."""
        return self._index[sec_id]

    def _offset(self, sec_id: str) -> int:
        return self._index[sec_id] * SLOT_SIZE

    def _get_i64(self, sec_id: str, field_offset: int) -> int:
        return struct.unpack_from('<q', self._buf, self._offset(sec_id) + field_offset)[0]

    def _set_i64(self, sec_id: str, field_offset: int, value: int) -> None:
        struct.pack_into('<q', self._buf, self._offset(sec_id) + field_offset, value)

    def _get_i32(self, sec_id: str, field_offset: int) -> int:
        return struct.unpack_from('<i', self._buf, self._offset(sec_id) + field_offset)[0]

    def _set_i32(self, sec_id: str, field_offset: int, value: int) -> None:
        struct.pack_into('<i', self._buf, self._offset(sec_id) + field_offset, value)

    def get_slot(self, sec_id: str) -> tuple[int, int, int, int]:
        """
        Read a slot's core fields.

        :return: (version, result_size, target_time, flags)
        """
        off = self._offset(sec_id) + _RESULT_META_OFFSET
        return struct.unpack_from(SLOT_FORMAT, self._buf, off)

    def set_target_time(self, sec_id: str, target_time: int):
        """Set the target_time field for a slot."""
        self._set_i64(sec_id, _TARGET_TIME_OFFSET, target_time)

    def get_target_time(self, sec_id: str) -> int:
        """Read the target_time field for a slot."""
        return self._get_i64(sec_id, _TARGET_TIME_OFFSET)

    def set_ltf_period_end(self, sec_id: str, period_end_exclusive: int):
        """Set the LTF chart-period end (ms, exclusive) for a slot.

        Reuses the ``dev_time`` field (offset 72): the live LTF-window path
        pushes no developing OHLCV, so this slot is free to carry the chart
        bar's period end alongside ``target_time`` (the period start).
        """
        self._set_i64(sec_id, 72, period_end_exclusive)

    def get_ltf_period_end(self, sec_id: str) -> int:
        """Read the LTF chart-period end (ms, exclusive) for a slot."""
        return self._get_i64(sec_id, 72)

    def set_result_meta(self, sec_id: str, version: int, result_size: int):
        """Set version and result_size fields."""
        off = self._offset(sec_id) + _RESULT_META_OFFSET
        struct.pack_into('<II', self._buf, off, version, result_size)

    def get_result_meta(self, sec_id: str) -> tuple[int, int]:
        """
        Read version and result_size.

        :return: (version, result_size)
        """
        off = self._offset(sec_id) + _RESULT_META_OFFSET
        return struct.unpack_from('<II', self._buf, off)

    def set_ltf_period_start(self, sec_id: str, period_start_ms: int) -> None:
        """Set the file-backed LTF round's chart-bar period start (ms)."""
        self._set_i64(sec_id, _LTF_PERIOD_START_OFFSET, period_start_ms)

    def get_ltf_period_start(self, sec_id: str) -> int:
        """Read the file-backed LTF round's chart-bar period start (ms)."""
        return self._get_i64(sec_id, _LTF_PERIOD_START_OFFSET)

    def set_flags(self, sec_id: str, flags: int):
        """Set the flags byte."""
        off = self._offset(sec_id) + _FLAGS_OFFSET
        struct.pack_into('<B', self._buf, off, flags)

    def get_flags(self, sec_id: str) -> int:
        """Read the flags byte."""
        off = self._offset(sec_id) + _FLAGS_OFFSET
        return struct.unpack_from('<B', self._buf, off)[0]

    # --- ring state -------------------------------------------------------

    def set_ring_state(self, sec_id: str, *, count: int, head: int, used: int) -> None:
        """Publish the ring's live-entry count, head index and arena usage."""
        self._set_i32(sec_id, _RING_COUNT_OFFSET, count)
        self._set_i32(sec_id, _RING_HEAD_OFFSET, head)
        self._set_i32(sec_id, _RING_USED_OFFSET, used)

    def get_ring_state(self, sec_id: str) -> tuple[int, int, int]:
        """
        Read the ring's live-entry count, head index and arena usage.

        :return: (count, head, used)
        """
        return (
            self._get_i32(sec_id, _RING_COUNT_OFFSET),
            self._get_i32(sec_id, _RING_HEAD_OFFSET),
            self._get_i32(sec_id, _RING_USED_OFFSET),
        )

    def get_ring_count(self, sec_id: str) -> int:
        """Read the number of live ring entries."""
        return self._get_i32(sec_id, _RING_COUNT_OFFSET)

    def get_ring_head(self, sec_id: str) -> int:
        """Read the index of the first live ring entry."""
        return self._get_i32(sec_id, _RING_HEAD_OFFSET)

    def get_ring_used(self, sec_id: str) -> int:
        """Read the number of bytes used in the ring payload arena."""
        return self._get_i32(sec_id, _RING_USED_OFFSET)

    def set_ring_version(self, sec_id: str, version: int) -> None:
        """Publish the ring block version (bumped on reallocation)."""
        self._set_i32(sec_id, _RING_VERSION_OFFSET, version)

    def get_ring_version(self, sec_id: str) -> int:
        """Read the ring block version."""
        return self._get_i32(sec_id, _RING_VERSION_OFFSET)

    # --- frontier ---------------------------------------------------------

    def set_frontier_close(self, sec_id: str, frontier_close: int) -> None:
        """Set the producer's frontier close (ms); :data:`FRONTIER_INF` = done."""
        self._set_i64(sec_id, _FRONTIER_OFFSET, frontier_close)

    def get_frontier_close(self, sec_id: str) -> int:
        """Read the producer's frontier close (ms)."""
        return self._get_i64(sec_id, _FRONTIER_OFFSET)

    # --- round context ----------------------------------------------------

    def set_round_context(self, sec_id: str, round_tick: int, sched_next_open: int) -> None:
        """Write the chart round's fixed tick instant and scheduled next session open."""
        self._set_i64(sec_id, _ROUND_TICK_OFFSET, round_tick)
        self._set_i64(sec_id, _ROUND_SCHED_NEXT_OPEN_OFFSET, sched_next_open)

    def get_round_context(self, sec_id: str) -> tuple[int, int]:
        """
        Read the chart round context.

        :return: (round_tick, round_sched_next_open)
        """
        return (
            self._get_i64(sec_id, _ROUND_TICK_OFFSET),
            self._get_i64(sec_id, _ROUND_SCHED_NEXT_OPEN_OFFSET),
        )

    def set_rounds_done(self, sec_id: str, rounds: int) -> None:
        """Set the child's finished-round counter."""
        self._set_i64(sec_id, _ROUNDS_DONE_OFFSET, rounds)

    def get_rounds_done(self, sec_id: str) -> int:
        """Read the child's finished-round counter."""
        return self._get_i64(sec_id, _ROUNDS_DONE_OFFSET)

    def increment_rounds_done(self, sec_id: str) -> int:
        """Increment and return the child's finished-round counter."""
        value = self._get_i64(sec_id, _ROUNDS_DONE_OFFSET) + 1
        self._set_i64(sec_id, _ROUNDS_DONE_OFFSET, value)
        return value

    # --- watermark matrix -------------------------------------------------

    def _watermark_offset_of(self, consumer_index: int, producer_index: int) -> int:
        if not (0 <= consumer_index < self._n and 0 <= producer_index < self._n):
            raise IndexError(
                f"watermark index out of range: ({consumer_index}, {producer_index})"
            )
        return self._watermark_offset + (
            consumer_index * self._n + producer_index) * _WATERMARK_ITEM_SIZE

    def set_watermark(self, consumer_index: int, producer_index: int, value: int) -> None:
        """Set the consumer's watermark (ms) for a producer; :data:`FRONTIER_INF` = done."""
        struct.pack_into(
            '<q', self._buf, self._watermark_offset_of(consumer_index, producer_index), value)

    def get_watermark(self, consumer_index: int, producer_index: int) -> int:
        """Read the consumer's watermark (ms) for a producer."""
        return struct.unpack_from(
            '<q', self._buf, self._watermark_offset_of(consumer_index, producer_index))[0]

    def min_watermark(self, producer_index: int,
                      consumer_indexes: Iterable[int] | None = None) -> int:
        """
        Minimum watermark over a producer's consumers.

        :param producer_index: Slot index of the producer.
        :param consumer_indexes: Consumer slot indexes to consider; ``None``
                                 means every slot. With no consumer at all the
                                 result is :data:`FRONTIER_INF` (nothing needs
                                 keeping).
        :return: The minimum watermark in ms.
        """
        indexes = range(self._n) if consumer_indexes is None else consumer_indexes
        result = FRONTIER_INF
        for idx in indexes:
            value = self.get_watermark(idx, producer_index)
            if value < result:
                result = value
        return result

    # --- developing bar ---------------------------------------------------

    def set_developing_bar(
        self, sec_id: str,
        dev_open: float, dev_high: float, dev_low: float,
        dev_close: float, dev_volume: float, dev_time: int,
    ):
        """Write developing HTF bar OHLCV+time for live ``lookahead_on``."""
        off = self._offset(sec_id) + _DEV_OHLCV_OFFSET
        struct.pack_into(
            _DEV_OHLCV_FORMAT, self._buf, off,
            dev_open, dev_high, dev_low, dev_close, dev_volume, dev_time,
        )

    def get_developing_bar(
        self, sec_id: str,
    ) -> tuple[float, float, float, float, float, int]:
        """Read developing HTF bar OHLCV+time.

        :return: (open, high, low, close, volume, time_ms)
        """
        off = self._offset(sec_id) + _DEV_OHLCV_OFFSET
        return struct.unpack_from(_DEV_OHLCV_FORMAT, self._buf, off)

    def close(self):
        """Close the shared memory (does not unlink)."""
        self._shm.close()

    def unlink(self):
        """Unlink (destroy) the shared memory."""
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass


class ResultBlock:
    """
    Per-security-ID shared memory block holding the pickled result value.

    Supports automatic reallocation when data outgrows the block.
    The version number is embedded in the SharedMemory name to allow
    readers to detect and re-attach after reallocation.
    """

    def __init__(self, sec_id: str, *, create: bool = True,
                 version: int = 0, size: int = INITIAL_RESULT_SIZE,
                 prefix: str):
        self._sec_id = sec_id
        self._version = version
        self._prefix = prefix
        name = _result_block_name(prefix, version)

        if create:
            try:
                self._shm = SharedMemory(name=name, create=True, size=size)
            except FileExistsError:
                stale = SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
                self._shm = SharedMemory(name=name, create=True, size=size)
        else:
            self._shm = SharedMemory(name=name, create=False)

        buf = self._shm.buf
        assert buf is not None
        self._buf: memoryview = buf

    @property
    def sec_id(self) -> str:
        return self._sec_id

    @property
    def version(self) -> int:
        return self._version

    @property
    def size(self) -> int:
        return self._shm.size

    def write(self, data: bytes, sync_block: SyncBlock) -> int:
        """
        Write pickle data to the result block. Reallocates if needed.

        :param data: Pickled bytes to write
        :param sync_block: SyncBlock to update version/size metadata
        :return: New version number
        """
        if len(data) > self._shm.size:
            new_size = max(len(data) * 2, INITIAL_RESULT_SIZE)
            new_version = self._version + 1
            new_name = _result_block_name(self._prefix, new_version)

            try:
                new_shm = SharedMemory(name=new_name, create=True, size=new_size)
            except FileExistsError:
                # As in the initial-allocation path: the name carries THIS
                # run's SyncBlock name, so an existing segment can only be a
                # leak from a dead run. Drop it and recreate.
                stale = SharedMemory(name=new_name, create=False)
                stale.close()
                stale.unlink()
                new_shm = SharedMemory(name=new_name, create=True, size=new_size)
            new_buf = new_shm.buf
            assert new_buf is not None
            new_buf[:len(data)] = data

            old_shm = self._shm
            self._shm = new_shm
            self._buf = new_buf
            self._version = new_version

            old_shm.close()
            try:
                old_shm.unlink()
            except FileNotFoundError:
                pass
        else:
            self._buf[:len(data)] = data

        sync_block.set_result_meta(self._sec_id, self._version, len(data))
        return self._version

    def read(self, size: int) -> bytes:
        """
        Read raw bytes from the result block.

        :param size: Number of bytes to read
        :return: Raw bytes
        """
        return bytes(self._buf[:size])

    def close(self):
        """Close the shared memory (does not unlink)."""
        self._shm.close()

    def unlink(self):
        """Unlink (destroy) the shared memory."""
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass


class ResultReader:
    """
    Reader-side handle for a security result block.

    Tracks the current version and re-attaches when the writer reallocates.
    """

    def __init__(self, sec_id: str, prefix: str):
        self._sec_id = sec_id
        self._prefix = prefix
        self._shm: SharedMemory | None = None
        self._version: int = -1

    def read(self, sync_block: SyncBlock, default=None):
        """
        Read the latest result value from shared memory.

        :param sync_block: SyncBlock to check version/size
        :param default: Value to return if no data available
        :return: Unpickled result value, or default
        """
        version, result_size = sync_block.get_result_meta(self._sec_id)

        if result_size == 0:
            return default

        # Re-attach if version changed (writer reallocated)
        if version != self._version:
            if self._shm is not None:
                self._shm.close()
            name = _result_block_name(self._prefix, version)
            self._shm = SharedMemory(name=name, create=False)
            self._version = version

        shm = self._shm
        assert shm is not None
        buf = shm.buf
        assert buf is not None
        return pickle.loads(buf[:result_size])

    def close(self):
        """Close the reader's shared memory handle."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None


def write_result(result_block: ResultBlock, sync_block: SyncBlock, value) -> int:
    """
    Pickle a value and write it to the result block.

    :param result_block: The result block to write to
    :param sync_block: The sync block to update metadata
    :param value: The value to pickle and store
    :return: New version number
    """
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return result_block.write(data, sync_block)


def write_na(result_block: ResultBlock, sync_block: SyncBlock) -> int:
    """
    Write an empty result (na) — sets result_size to 0.

    :param result_block: The result block (not modified)
    :param sync_block: The sync block to update metadata
    :return: Current version number
    """
    sync_block.set_result_meta(
        result_block.sec_id,
        result_block.version,
        0  # zero size = no data = na
    )
    return result_block.version


# --- ring block ------------------------------------------------------------

# Ring block layout:
#   offset 0:  int32 capacity    — number of index entries the block can hold
#   offset 4:  int32 arena_size  — payload arena size in bytes
#   offset 8:  8 bytes pad (alignment to 16)
#   offset 16: index array of ``capacity`` entries, _RING_ENTRY_SIZE bytes each
#   then:      payload arena of ``arena_size`` bytes
#
# Live entries occupy indexes ``head .. head + count - 1`` (no wrap-around):
# the GC compacts to index 0 instead of wrapping, which keeps both the index
# array and the payload arena in close order and lets readers binary-search.
_RING_HEADER_FORMAT = '<ii'
_RING_HEADER_SIZE = 16
_RING_ENTRY_FORMAT = '<qqqi'  # open(ms), close(ms), payload_offset, payload_len
_RING_ENTRY_SIZE = 32  # 28 bytes of fields + 4 pad, keeping 8-byte alignment

INITIAL_RING_CAPACITY = 64
INITIAL_RING_ARENA = 8192


def _ring_block_name(prefix: str, version: int) -> str:
    """Generate SharedMemory name for a ring block.

    :param prefix: The block's run-unique prefix (see
        :meth:`SyncBlock.block_prefix`).
    :param version: Reallocation counter, embedded so readers re-attach.
    """
    return f"{prefix}r{version}"


class RingBlock:
    """
    Per-security-ID shared memory block holding the append-only bar history.

    Holds a fixed-size index array of ``(open, close, payload_offset,
    payload_len)`` entries plus a payload arena of pickled values. The version
    is embedded in the SharedMemory name so readers detect a reallocation and
    re-attach.
    """

    def __init__(self, sec_id: str, *, create: bool = True, version: int = 0,
                 capacity: int = INITIAL_RING_CAPACITY, arena_size: int = INITIAL_RING_ARENA,
                 prefix: str):
        self._sec_id = sec_id
        self._version = version
        self._prefix = prefix
        name = _ring_block_name(prefix, version)

        if create:
            size = _RING_HEADER_SIZE + capacity * _RING_ENTRY_SIZE + arena_size
            try:
                self._shm = SharedMemory(name=name, create=True, size=size)
            except FileExistsError:
                # Only a leak from a dead run can hold this name: it carries
                # THIS run's SyncBlock name, which the OS never hands out while
                # a segment of that name is alive. Drop the stale block.
                stale = SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
                self._shm = SharedMemory(name=name, create=True, size=size)
        else:
            self._shm = SharedMemory(name=name, create=False)

        buf = self._shm.buf
        assert buf is not None
        self._buf: memoryview = buf

        if create:
            struct.pack_into(_RING_HEADER_FORMAT, self._buf, 0, capacity, arena_size)
            self._capacity = capacity
            self._arena_size = arena_size
        else:
            self._capacity, self._arena_size = struct.unpack_from(
                _RING_HEADER_FORMAT, self._buf, 0)

        self._arena_offset = _RING_HEADER_SIZE + self._capacity * _RING_ENTRY_SIZE

    @property
    def sec_id(self) -> str:
        return self._sec_id

    @property
    def version(self) -> int:
        return self._version

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def arena_size(self) -> int:
        return self._arena_size

    def read_entry(self, index: int) -> tuple[int, int, int, int]:
        """
        Read one index entry.

        :param index: Absolute index in the index array.
        :return: (open_ms, close_ms, payload_offset, payload_len)
        """
        off = _RING_HEADER_SIZE + index * _RING_ENTRY_SIZE
        return struct.unpack_from(_RING_ENTRY_FORMAT, self._buf, off)

    def write_entry(self, index: int, open_ms: int, close_ms: int,
                    payload_offset: int, payload_len: int) -> None:
        """Write one index entry."""
        off = _RING_HEADER_SIZE + index * _RING_ENTRY_SIZE
        struct.pack_into(_RING_ENTRY_FORMAT, self._buf, off,
                         open_ms, close_ms, payload_offset, payload_len)

    def read_payload(self, payload_offset: int, payload_len: int) -> bytes:
        """Read raw payload bytes from the arena."""
        start = self._arena_offset + payload_offset
        return bytes(self._buf[start:start + payload_len])

    def write_payload(self, payload_offset: int, data: bytes) -> None:
        """Write raw payload bytes into the arena."""
        start = self._arena_offset + payload_offset
        self._buf[start:start + len(data)] = data

    def close(self):
        """Close the shared memory (does not unlink)."""
        self._shm.close()

    def unlink(self):
        """Unlink (destroy) the shared memory."""
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass


class RingWriter:
    """
    Producer-side handle for a security context's bar history ring.

    Appends ``(open, close, value)`` entries ordered by close, garbage-collects
    entries no consumer needs any more, and grows the block (with a version
    bump) when the GC cannot free enough room. Every publication bumps the
    producer's frontier close and wakes the waiting consumers on the per-sid
    ``multiprocessing.Condition``.

    Deadlock-freedom invariant: the writer NEVER blocks on ring space — a full
    ring grows instead of waiting — so a producer can always publish the bars
    its consumers are waiting for.
    """

    def __init__(self, sec_id: str, sync_block: SyncBlock,
                 condition: 'ConditionType', *,
                 capacity: int = INITIAL_RING_CAPACITY,
                 arena_size: int = INITIAL_RING_ARENA):
        self._sec_id = sec_id
        self._sync = sync_block
        self._cond = condition
        self._block = RingBlock(sec_id, create=True, version=0,
                                capacity=capacity, arena_size=arena_size,
                                prefix=sync_block.block_prefix(sec_id))
        with condition:
            sync_block.set_ring_version(sec_id, 0)
            sync_block.set_ring_state(sec_id, count=0, head=0, used=0)

    @property
    def sec_id(self) -> str:
        return self._sec_id

    @property
    def version(self) -> int:
        return self._block.version

    @property
    def capacity(self) -> int:
        return self._block.capacity

    def _live_entries(self) -> list[tuple[int, int, bytes]]:
        """Snapshot every live entry as ``(open, close, payload_bytes)``."""
        count, head, _ = self._sync.get_ring_state(self._sec_id)
        entries: list[tuple[int, int, bytes]] = []
        for i in range(head, head + count):
            open_ms, close_ms, p_off, p_len = self._block.read_entry(i)
            entries.append((open_ms, close_ms, self._block.read_payload(p_off, p_len)))
        return entries

    def _rebuild(self, entries: list[tuple[int, int, bytes]],
                 capacity: int, arena_size: int, *, new_version: bool) -> None:
        """Rewrite the ring from ``entries``, optionally into a fresh block."""
        if new_version:
            old = self._block
            self._block = RingBlock(self._sec_id, create=True, version=old.version + 1,
                                    capacity=capacity, arena_size=arena_size,
                                    prefix=self._sync.block_prefix(self._sec_id))
            old.close()
            old.unlink()

        used = 0
        for i, (open_ms, close_ms, payload) in enumerate(entries):
            self._block.write_payload(used, payload)
            self._block.write_entry(i, open_ms, close_ms, used, len(payload))
            used += len(payload)

        if new_version:
            self._sync.set_ring_version(self._sec_id, self._block.version)
        self._sync.set_ring_state(self._sec_id, count=len(entries), head=0, used=used)

    def _gc(self, consumer_indexes: Iterable[int] | None) -> None:
        """Drop entries below the consumers' minimum watermark.

        The last entry at or below the minimum is KEPT: a consumer asking for
        an as-of instant above every remaining close still has to receive that
        entry (the rule is "the last bar closing at or before as-of").
        """
        entries = self._live_entries()
        if not entries:
            return
        watermark = self._sync.min_watermark(
            self._sync.index_of(self._sec_id),
            None if consumer_indexes is None else list(consumer_indexes))
        closes = [e[1] for e in entries]
        keep_from = bisect_right(closes, watermark) - 1
        if keep_from <= 0:
            return
        self._rebuild(entries[keep_from:], self._block.capacity,
                      self._block.arena_size, new_version=False)

    def _grow(self, needed_payload: int) -> None:
        """Double capacity and/or arena into a fresh block (version bump)."""
        entries = self._live_entries()
        capacity = self._block.capacity
        if len(entries) + 1 > capacity:
            capacity *= 2
        used = sum(len(e[2]) for e in entries)
        arena_size = self._block.arena_size
        while used + needed_payload > arena_size:
            arena_size *= 2
        self._rebuild(entries, capacity, arena_size, new_version=True)

    def append(self, open_ms: int, close_ms: int, value: Any,
               consumer_indexes: Iterable[int] | None = None,
               frontier_close: int | None = None) -> None:
        """
        Append one bar to the ring and publish it.

        :param open_ms: Bar open time in ms.
        :param close_ms: Scheduled bar close in ms (the ring is ordered by it).
        :param value: The value to publish (pickled into the payload arena).
        :param consumer_indexes: Slot indexes of this producer's consumers,
                                 used by the GC watermark minimum. ``None``
                                 means every slot.
        :param frontier_close: New frontier close (ms); ``None`` keeps the
                               current one. Applied monotonically.
        """
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with self._cond:
            count, head, used = self._sync.get_ring_state(self._sec_id)
            if (head + count >= self._block.capacity
                    or used + len(payload) > self._block.arena_size):
                self._gc(consumer_indexes)
                count, head, used = self._sync.get_ring_state(self._sec_id)
                if (head + count >= self._block.capacity
                        or used + len(payload) > self._block.arena_size):
                    self._grow(len(payload))
                    count, head, used = self._sync.get_ring_state(self._sec_id)

            index = head + count
            self._block.write_payload(used, payload)
            self._block.write_entry(index, open_ms, close_ms, used, len(payload))
            self._sync.set_ring_state(
                self._sec_id, count=count + 1, head=head, used=used + len(payload))

            if frontier_close is not None:
                current = self._sync.get_frontier_close(self._sec_id)
                if frontier_close > current:
                    self._sync.set_frontier_close(self._sec_id, frontier_close)
            self._cond.notify_all()

    def set_frontier_close(self, frontier_close: int) -> None:
        """Raise the frontier close (monotonic) and wake the consumers."""
        with self._cond:
            current = self._sync.get_frontier_close(self._sec_id)
            if frontier_close > current:
                self._sync.set_frontier_close(self._sec_id, frontier_close)
            self._cond.notify_all()

    def finish(self) -> None:
        """Mark the producer done: frontier close = ``+inf``."""
        self.set_frontier_close(FRONTIER_INF)

    def close(self) -> None:
        """Close the ring block handle."""
        self._block.close()

    def unlink(self) -> None:
        """Unlink (destroy) the ring block."""
        self._block.unlink()


class RingReader:
    """
    Consumer-side handle for a peer security context's bar history ring.

    Re-attaches automatically when the writer reallocates (the ring version in
    the SyncBlock changes).
    """

    def __init__(self, sec_id: str, sync_block: SyncBlock, condition: 'ConditionType'):
        self._sec_id = sec_id
        self._sync = sync_block
        self._cond = condition
        self._block: RingBlock | None = None
        self._version = -1

    @property
    def sec_id(self) -> str:
        return self._sec_id

    @property
    def version(self) -> int:
        return self._version

    def _attach(self) -> RingBlock | None:
        """Open (or re-open) the current ring block; ``None`` if not created yet."""
        if self._sync.get_ring_count(self._sec_id) == 0 and self._block is None:
            return None
        version = self._sync.get_ring_version(self._sec_id)
        if version != self._version or self._block is None:
            if self._block is not None:
                self._block.close()
                self._block = None
            try:
                self._block = RingBlock(self._sec_id, create=False, version=version,
                                        prefix=self._sync.block_prefix(self._sec_id))
            except FileNotFoundError:
                return None
            self._version = version
        return self._block

    def _closes(self, block: RingBlock, head: int, count: int) -> list[int]:
        return [block.read_entry(i)[1] for i in range(head, head + count)]

    def _entry(self, block: RingBlock, index: int) -> tuple[int, int, Any]:
        open_ms, close_ms, p_off, p_len = block.read_entry(index)
        return open_ms, close_ms, pickle.loads(block.read_payload(p_off, p_len))

    def count(self) -> int:
        """Number of live entries currently in the ring."""
        return self._sync.get_ring_count(self._sec_id)

    def frontier_close(self) -> int:
        """The producer's frontier close (ms); :data:`FRONTIER_INF` = done."""
        return self._sync.get_frontier_close(self._sec_id)

    def last_close_at_or_before(self, ms: int) -> tuple[int, int, Any] | None:
        """
        The last entry whose close is at or before ``ms``.

        :param ms: As-of instant in ms.
        :return: ``(open_ms, close_ms, value)`` or ``None`` if there is none.
        """
        with self._cond:
            block = self._attach()
            if block is None:
                return None
            count, head, _ = self._sync.get_ring_state(self._sec_id)
            if count == 0:
                return None
            pos = bisect_right(self._closes(block, head, count), ms) - 1
            if pos < 0:
                return None
            return self._entry(block, head + pos)

    def has_close(self, ms: int) -> bool:
        """Whether an entry with exactly this close is in the ring."""
        with self._cond:
            block = self._attach()
            if block is None:
                return False
            count, head, _ = self._sync.get_ring_state(self._sec_id)
            if count == 0:
                return False
            closes = self._closes(block, head, count)
            pos = bisect_left(closes, ms)
            return pos < count and closes[pos] == ms

    def range_by_close(self, start_ms: int, end_ms: int) -> list[tuple[int, int, Any]]:
        """
        Entries with ``start_ms < close <= end_ms``, in close order.

        :param start_ms: Exclusive lower bound on the close.
        :param end_ms: Inclusive upper bound on the close.
        :return: List of ``(open_ms, close_ms, value)``.
        """
        with self._cond:
            block = self._attach()
            if block is None:
                return []
            count, head, _ = self._sync.get_ring_state(self._sec_id)
            if count == 0:
                return []
            closes = self._closes(block, head, count)
            lo = bisect_right(closes, start_ms)
            hi = bisect_right(closes, end_ms)
            return [self._entry(block, head + i) for i in range(lo, hi)]

    def wait_until(self, predicate: Callable[[], bool],
                   stop_event: 'EventType | None' = None,
                   timeout: float | None = None,
                   poll_interval: float = 0.1) -> bool:
        """
        Block on the producer's condition until ``predicate`` holds.

        No sleep-based polling: the wait is on the ``multiprocessing.Condition``
        the producer notifies; ``poll_interval`` only bounds how long a single
        ``wait`` call sits before ``stop_event`` and the overall ``timeout`` are
        re-checked (a producer that died notifies nobody).

        :param predicate: Called with the condition held; the wait ends when
                          it returns ``True``.
        :param stop_event: Optional shutdown event; a set event ends the wait.
        :param timeout: Overall timeout in seconds; ``None`` waits forever.
        :param poll_interval: Upper bound of one internal wait slice, seconds.
        :return: ``True`` if the predicate held, ``False`` on stop or timeout.
        """
        deadline = None if timeout is None else monotonic() + timeout
        with self._cond:
            while True:
                if predicate():
                    return True
                if stop_event is not None and stop_event.is_set():
                    return False
                slice_timeout = poll_interval
                if deadline is not None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return predicate()
                    slice_timeout = min(slice_timeout, remaining)
                self._cond.wait(slice_timeout)

    def wait_for_close(self, ms: int, stop_event: 'EventType | None' = None,
                       timeout: float | None = None) -> bool:
        """
        Wait until the producer published everything closing at or before ``ms``.

        The wait ends as soon as an entry closes exactly at ``ms`` or the
        frontier close reaches it — the producer's next unpublished bar always
        closes above the consumer's as-of instant, so this always terminates.
        """
        return self.wait_until(
            lambda: self.has_close(ms) or self.frontier_close() >= ms,
            stop_event, timeout)

    def close(self) -> None:
        """Close the reader's ring block handle."""
        if self._block is not None:
            self._block.close()
            self._block = None
            self._version = -1


def create_ring_conditions(sec_ids: Iterable[str]) -> dict[str, 'ConditionType']:
    """
    Create one ``multiprocessing.Condition`` per security id.

    The conditions are created on the default multiprocessing context, exactly
    like ``SecurityState``'s events and locks, so they pickle into spawned
    children through the ``Process`` argument tuple.

    :param sec_ids: Security ids to create conditions for.
    :return: Mapping of security id to condition.
    """
    return {sid: Condition() for sid in sec_ids}
