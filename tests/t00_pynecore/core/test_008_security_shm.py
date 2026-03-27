"""
@pyne
"""
import pickle

from pynecore.core.security_shm import (
    SyncBlock, ResultBlock, ResultReader, INITIAL_RESULT_SIZE,
    write_result, write_na,
)


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
