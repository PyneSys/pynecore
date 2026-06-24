"""
Shared memory layer for request.security() inter-process communication.

Two types of shared memory blocks:

1. **SyncBlock** — single fixed-size block containing metadata for all security slots
   (timestamp, version, result_size, flags per slot).

2. **ResultBlock** — one per security ID, holds the pickled result value.
   Reallocated with doubled size when the pickle data outgrows the current block.
"""
import pickle
import struct
from multiprocessing.shared_memory import SharedMemory

# Per-slot layout in the sync block:
#   offset 0:  int64   last_timestamp  (8 bytes) — last bar timestamp from security process
#   offset 8:  uint32  version         (4 bytes) — result block version (incremented on realloc)
#   offset 12: uint32  result_size     (4 bytes) — current pickle data size in bytes
#   offset 16: int64   target_time     (8 bytes) — target time the process should advance to
#   offset 24: uint8   flags           (1 byte)  — state flags
#   offset 25: 7 bytes pad (alignment to 32)
#   offset 32: float64 dev_open        (8 bytes) — developing HTF bar OHLCV (lookahead_on live)
#   offset 40: float64 dev_high        (8 bytes)
#   offset 48: float64 dev_low         (8 bytes)
#   offset 56: float64 dev_close       (8 bytes)
#   offset 64: float64 dev_volume      (8 bytes)
#   offset 72: int64   dev_time        (8 bytes) — developing HTF bar timestamp (ms);
#                       reused as period_end_exclusive (ms) by the live LTF-window
#                       path, which carries no pushed OHLCV (see FLAG_LTF_WINDOW)
#
# Total per slot: 80 bytes
SLOT_FORMAT = '<qIIqB'
SLOT_SIZE = 80
SLOT_DATA_SIZE = struct.calcsize(SLOT_FORMAT)  # 25 bytes — original fields only

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


def _result_block_name(sec_id: str, version: int) -> str:
    """Generate SharedMemory name for a result block."""
    safe_id = sec_id.replace('\xb7', '_')
    return f"pyne_{safe_id}_v{version}"


class SyncBlock:
    """
    Fixed-size shared memory block containing sync metadata for all security slots.

    Layout: N consecutive slots of SLOT_SIZE bytes each.
    """

    def __init__(self, sec_ids: list[str], *, create: bool = True, name: str | None = None):
        self._sec_ids = list(sec_ids)
        self._index = {sid: i for i, sid in enumerate(sec_ids)}
        total_size = max(SLOT_SIZE * len(sec_ids), 1)

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

    def _offset(self, sec_id: str) -> int:
        return self._index[sec_id] * SLOT_SIZE

    def get_slot(self, sec_id: str) -> tuple[int, int, int, int, int]:
        """
        Read a slot's fields.

        :return: (last_timestamp, version, result_size, target_time, flags)
        """
        off = self._offset(sec_id)
        return struct.unpack_from(SLOT_FORMAT, self._buf, off)

    def set_target_time(self, sec_id: str, target_time: int):
        """Set the target_time field for a slot."""
        off = self._offset(sec_id)
        struct.pack_into('<q', self._buf, off + 16, target_time)

    def get_target_time(self, sec_id: str) -> int:
        """Read the target_time field for a slot."""
        off = self._offset(sec_id)
        return struct.unpack_from('<q', self._buf, off + 16)[0]

    def set_ltf_period_end(self, sec_id: str, period_end_exclusive: int):
        """Set the LTF chart-period end (ms, exclusive) for a slot.

        Reuses the ``dev_time`` field (offset 72): the live LTF-window path
        pushes no developing OHLCV, so this slot is free to carry the chart
        bar's period end alongside ``target_time`` (the period start).
        """
        off = self._offset(sec_id)
        struct.pack_into('<q', self._buf, off + 72, period_end_exclusive)

    def get_ltf_period_end(self, sec_id: str) -> int:
        """Read the LTF chart-period end (ms, exclusive) for a slot."""
        off = self._offset(sec_id)
        return struct.unpack_from('<q', self._buf, off + 72)[0]

    def set_timestamp(self, sec_id: str, timestamp: int):
        """Set the last_timestamp field."""
        off = self._offset(sec_id)
        struct.pack_into('<q', self._buf, off, timestamp)

    def get_timestamp(self, sec_id: str) -> int:
        """Read the last_timestamp field."""
        off = self._offset(sec_id)
        return struct.unpack_from('<q', self._buf, off)[0]

    def set_result_meta(self, sec_id: str, version: int, result_size: int):
        """Set version and result_size fields."""
        off = self._offset(sec_id)
        struct.pack_into('<II', self._buf, off + 8, version, result_size)

    def get_result_meta(self, sec_id: str) -> tuple[int, int]:
        """
        Read version and result_size.

        :return: (version, result_size)
        """
        off = self._offset(sec_id)
        return struct.unpack_from('<II', self._buf, off + 8)

    def set_flags(self, sec_id: str, flags: int):
        """Set the flags byte."""
        off = self._offset(sec_id)
        struct.pack_into('<B', self._buf, off + 24, flags)

    def get_flags(self, sec_id: str) -> int:
        """Read the flags byte."""
        off = self._offset(sec_id)
        return struct.unpack_from('<B', self._buf, off + 24)[0]

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
                 version: int = 0, size: int = INITIAL_RESULT_SIZE):
        self._sec_id = sec_id
        self._version = version
        name = _result_block_name(sec_id, version)

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
            new_name = _result_block_name(self._sec_id, new_version)

            try:
                new_shm = SharedMemory(name=new_name, create=True, size=new_size)
            except FileExistsError:
                # A segment of this exact name leaked from an earlier run that
                # was killed before cleanup (the name is deterministic per
                # sec_id+version). Drop the stale block and recreate, mirroring
                # the initial-allocation path above.
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

    def __init__(self, sec_id: str):
        self._sec_id = sec_id
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
            name = _result_block_name(self._sec_id, version)
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
