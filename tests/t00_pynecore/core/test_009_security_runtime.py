"""
@pyne
"""
from pynecore.core.security import (
    SecurityState, _get_confirmed_time,
    create_chart_protocol, create_security_protocol,
    setup_security_states,
)
from pynecore.core.security_shm import (
    SyncBlock, ResultBlock, ResultReader, write_result,
)
from pynecore.core.resampler import Resampler
from multiprocessing import Event
from zoneinfo import ZoneInfo


def _make_state(
    sec_id="sec_test",
    timeframe="1D",
    same_timeframe=False,
    gaps_on=False,
    tz_str="UTC",
) -> SecurityState:
    """Helper to create a SecurityState for testing."""
    tz = ZoneInfo(tz_str)
    resampler = None if same_timeframe else Resampler.get_resampler(timeframe)
    state = SecurityState(
        sec_id=sec_id,
        timeframe=timeframe,
        gaps_on=gaps_on,
        same_timeframe=same_timeframe,
        resampler=resampler,
        tz=tz,
    )
    state.data_ready.set()
    return state


def __test_confirmed_time_same_tf__(log):
    """_get_confirmed_time returns chart_time for same timeframe"""
    state = _make_state(timeframe="5", same_timeframe=True)

    # Bar 0: same TF always returns chart_time
    result = _get_confirmed_time(state, 1000)
    assert result == 1000

    # Bar 1: still returns chart_time
    state.prev_chart_time = 1000
    result = _get_confirmed_time(state, 2000)
    assert result == 2000


def __test_confirmed_time_htf_no_new_period__(log):
    """_get_confirmed_time returns last_confirmed when HTF period hasn't changed"""
    state = _make_state(timeframe="1D")
    state.last_confirmed = 0

    # Bar 0: prev is None → returns last_confirmed
    result = _get_confirmed_time(state, 1_000_000)
    assert result == 0

    # Bars within same daily period → returns last_confirmed
    # Use timestamps within the same UTC day
    # 2024-01-15 10:00 UTC = 1705312800000 ms
    # 2024-01-15 10:05 UTC = 1705313100000 ms
    state.prev_chart_time = 1705312800000
    state.last_confirmed = 0
    result = _get_confirmed_time(state, 1705313100000)
    assert result == 0  # same day, no new period


def __test_confirmed_time_htf_new_period__(log):
    """_get_confirmed_time returns prev_period when a new HTF period starts"""
    state = _make_state(timeframe="1D", tz_str="UTC")
    state.last_confirmed = 0

    # 2024-01-15 23:55 UTC = end of Jan 15
    prev_time = 1705362900000  # 2024-01-15T23:55:00Z in ms
    # 2024-01-16 00:05 UTC = start of Jan 16
    curr_time = 1705363500000  # 2024-01-16T00:05:00Z in ms

    state.prev_chart_time = prev_time
    result = _get_confirmed_time(state, curr_time)

    # Should return the opening time of Jan 15 (the period that just closed)
    resampler = Resampler.get_resampler("1D")
    expected = resampler.get_bar_time(prev_time, ZoneInfo("UTC"))
    assert result == expected
    assert result > 0


def __test_confirmed_time_first_bar_htf__(log):
    """_get_confirmed_time on first bar with HTF returns 0 (no confirmed period)"""
    state = _make_state(timeframe="1W")
    state.last_confirmed = 0
    state.prev_chart_time = None

    result = _get_confirmed_time(state, 1705312800000)
    assert result == 0


def __test_chart_protocol_signal_read_flow__(log):
    """Chart protocol: signal sets events, read returns value after wait"""
    sec_ids = ["sec_flow"]
    sb = SyncBlock(sec_ids)
    rb = ResultBlock("sec_flow", create=True, version=0)

    state = _make_state(sec_id="sec_flow", timeframe="5", same_timeframe=True)
    states = {"sec_flow": state}

    signal_fn, write_fn, read_fn, wait_fn, cleanup = create_chart_protocol(states, sb)

    try:
        # Simulate: write a value to shared memory (as if security process did it)
        write_result(rb, sb, 42.0)

        # data_ready is initially set → read should return the value immediately
        result = read_fn("sec_flow", default=None)
        assert result == 42.0

        # Write a new value
        write_result(rb, sb, 99.0)
        result = read_fn("sec_flow", default=None)
        assert result == 99.0
    finally:
        cleanup()
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def __test_chart_protocol_gaps_on__(log):
    """Chart protocol: gaps_on returns default when no new period"""
    sec_ids = ["sec_gaps"]
    sb = SyncBlock(sec_ids)
    rb = ResultBlock("sec_gaps", create=True, version=0)

    state = _make_state(sec_id="sec_gaps", timeframe="1D", gaps_on=True)
    state.new_period = False  # no new period
    states = {"sec_gaps": state}

    signal_fn, write_fn, read_fn, wait_fn, cleanup = create_chart_protocol(states, sb)

    try:
        write_result(rb, sb, 100.0)

        # gaps_on + no new_period → should return default
        result = read_fn("sec_gaps", default="NA")
        assert result == "NA"

        # Now simulate new_period = True
        state.new_period = True
        result = read_fn("sec_gaps", default="NA")
        assert result == 100.0
    finally:
        cleanup()
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def __test_security_protocol_write_read__(log):
    """Security protocol: write and immediate read"""
    sec_ids = ["sec_a", "sec_b"]
    sb = SyncBlock(sec_ids)
    rb_a = ResultBlock("sec_a", create=True, version=0)
    rb_b = ResultBlock("sec_b", create=True, version=0)

    signal_fn, write_fn, read_fn, wait_fn, cleanup, _ = create_security_protocol(
        "sec_a", sb, rb_a, sec_ids,
    )

    try:
        # Write as security context A
        write_fn("sec_a", 55.5)

        # Read own value
        result = read_fn("sec_a", default=None)
        assert result == 55.5

        # Write a value for sec_b externally (simulate another process)
        write_result(rb_b, sb, "from_b")

        # Read cross-context (B's value)
        result = read_fn("sec_b", default=None)
        assert result == "from_b"

        # Read non-existent data → default
        from pynecore.core.security_shm import write_na
        write_na(rb_b, sb)
        result = read_fn("sec_b", default="NA")
        assert result == "NA"
    finally:
        cleanup()
        rb_a.close()
        rb_a.unlink()
        rb_b.close()
        rb_b.unlink()
        sb.close()
        sb.unlink()


def __test_setup_security_states__(log):
    """setup_security_states creates correct states from __security_contexts__"""
    from pynecore.lib import barmerge

    contexts = {
        "sec_0": {
            "symbol": "CAPITALCOM:EURUSD",
            "timeframe": "1D",
            "gaps": barmerge.gaps_off,
        },
        "sec_1": {
            "symbol": "CAPITALCOM:EURUSD",
            "timeframe": "5",  # same as chart
            "gaps": barmerge.gaps_on,
        },
    }

    states, sync_block, result_blocks = setup_security_states(
        contexts, chart_timeframe="5", tz=ZoneInfo("UTC"),
    )

    try:
        assert len(states) == 2
        assert len(result_blocks) == 2

        # sec_0: HTF (1D vs chart 5m)
        s0 = states["sec_0"]
        assert s0.timeframe == "1D"
        assert s0.gaps_on is False
        assert s0.same_timeframe is False
        assert s0.resampler is not None
        assert s0.data_ready.is_set()
        assert s0.is_ltf is False

        # sec_1: same TF as chart
        s1 = states["sec_1"]
        assert s1.timeframe == "5"
        assert s1.gaps_on is True
        assert s1.same_timeframe is True
        assert s1.resampler is None
        assert s1.data_ready.is_set()
        assert s1.is_ltf is False
    finally:
        for rb in result_blocks.values():
            rb.close()
            rb.unlink()
        sync_block.close()
        sync_block.unlink()


def __test_ltf_protocol_accumulation__(log):
    """LTF security protocol accumulates values into array via flush"""
    sec_ids = ["sec_ltf"]
    sb = SyncBlock(sec_ids)
    rb = ResultBlock("sec_ltf", create=True, version=0)

    signal_fn, write_fn, read_fn, wait_fn, cleanup, flush_fn = create_security_protocol(
        "sec_ltf", sb, rb, sec_ids, is_ltf=True,
    )

    try:
        assert flush_fn is not None

        # Simulate 3 LTF bars writing values
        write_fn("sec_ltf", 1.1050)
        write_fn("sec_ltf", 1.1052)
        write_fn("sec_ltf", 1.1048)

        # Before flush: shared memory has no data yet
        version, result_size = sb.get_result_meta("sec_ltf")
        assert result_size == 0

        # Flush writes accumulated array
        flush_fn()

        # Read the array
        result = read_fn("sec_ltf", default=[])
        assert result == [1.1050, 1.1052, 1.1048]

        # Buffer is cleared after flush — another flush writes empty list
        flush_fn()
        result = read_fn("sec_ltf", default="SHOULD_NOT_USE")
        assert result == []
    finally:
        cleanup()
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def __test_ltf_protocol_empty_flush__(log):
    """LTF flush with no writes produces empty array (not na)"""
    sec_ids = ["sec_ltf2"]
    sb = SyncBlock(sec_ids)
    rb = ResultBlock("sec_ltf2", create=True, version=0)

    signal_fn, write_fn, read_fn, wait_fn, cleanup, flush_fn = create_security_protocol(
        "sec_ltf2", sb, rb, sec_ids, is_ltf=True,
    )

    try:
        # Flush immediately — no writes
        flush_fn()

        result = read_fn("sec_ltf2", default="NA")
        assert result == []
        assert isinstance(result, list)
    finally:
        cleanup()
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def __test_ltf_htf_protocol_no_flush__(log):
    """Non-LTF protocol returns flush=None"""
    sec_ids = ["sec_htf"]
    sb = SyncBlock(sec_ids)
    rb = ResultBlock("sec_htf", create=True, version=0)

    signal_fn, write_fn, read_fn, wait_fn, cleanup, flush_fn = create_security_protocol(
        "sec_htf", sb, rb, sec_ids, is_ltf=False,
    )

    try:
        assert flush_fn is None

        # HTF write goes directly to shared memory
        write_fn("sec_htf", 42.0)
        result = read_fn("sec_htf", default=None)
        assert result == 42.0
    finally:
        cleanup()
        rb.close()
        rb.unlink()
        sb.close()
        sb.unlink()


def __test_setup_security_states_ltf__(log):
    """setup_security_states handles LTF context correctly"""
    contexts = {
        "sec_ltf": {
            "symbol": "AAPL",
            "timeframe": "1",
            "is_ltf": True,
        },
    }

    states, sync_block, result_blocks = setup_security_states(
        contexts, chart_timeframe="5", tz=ZoneInfo("UTC"),
    )

    try:
        s = states["sec_ltf"]
        assert s.is_ltf is True
        assert s.gaps_on is False
        assert s.same_timeframe is False
        assert s.resampler is None
        assert s.data_ready.is_set()
    finally:
        for rb in result_blocks.values():
            rb.close()
            rb.unlink()
        sync_block.close()
        sync_block.unlink()
