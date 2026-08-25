"""#37 S1' dual-mode feed — deterministic tick-synthesis tests.

Canned ``/trades/latest`` streams through the fake-client seam; ``time.time``
and ``asyncio.sleep`` are controlled, so slot boundaries and rollover are
exact. Covers the panel's guard set: dispatch isolation (default mode never
touches the tick endpoint), fail-fast config, cursor dedup across overlapping
polls, synthesis O/H/L/C/V, emit-ordering holdback with the OFFICIAL close
authoritative, the withheld-close SYNTH fallback (Live-L4-T03), and the
throttle transition log.
"""
import asyncio

import pytest

import pynecore.lib as lib
lib.bar_index = 0

from pynecore_dnse import broker


PERIOD = 300          # timeframe "5"
SLOT_A = 1_700_000_100 - (1_700_000_100 % PERIOD)   # a 5m slot start
SLOT_B = SLOT_A + PERIOD


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def _fast_sleep(_delay, result=None):
        return result
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)


class _Clock:
    def __init__(self, start: float):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now


def _broker(fake_client, *, feed_mode="tick", **responses):
    cfg = broker.DNSEBrokerConfig(api_key="k", api_secret="s", account_no="ACC1",
                                  feed_mode=feed_mode)
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="5", config=cfg)
    instance._client = fake_client(**responses)
    return instance


def _print(price, qty, total):
    return {"matchPrice": price, "matchQtty": qty, "totalVolumeTraded": total}


def __test_default_mode_never_touches_the_tick_endpoint__(fake_client, monkeypatch):
    """Parity isolation: feed_mode default must run today's closed-bar path —
    the tick endpoint must not receive a single request."""
    clock = _Clock(SLOT_A + 10)
    monkeypatch.setattr(broker.time, "time", clock)
    b = _broker(fake_client, feed_mode="ohlc",
                get_ohlc=(200, {"t": [SLOT_A - PERIOD], "o": [1900.0], "h": [1910.0],
                                "l": [1890.0], "c": [1905.0], "v": [50.0]}),
                get_latest_trade=(500, {"never": "called"}))
    bar = asyncio.run(b.watch_ohlcv("VN30F1M", "5"))
    assert bar.is_closed and bar.close == 1905.0
    assert b._client.count("get_latest_trade") == 0, \
        "default mode leaked into the tick endpoint"


def __test_invalid_feed_mode_fails_fast_at_init__(fake_client):
    cfg = broker.DNSEBrokerConfig(api_key="k", api_secret="s", account_no="A",
                                  feed_mode="ticks")   # typo
    with pytest.raises(ValueError, match="feed_mode"):
        broker.DNSEBroker(symbol="VN30F1M", timeframe="5", config=cfg)


def __test_forming_bar_synthesis_and_cursor_dedup__(fake_client, monkeypatch):
    """Two overlapping polls: replayed prints (total <= cursor) must not
    double-count; O=first, H/L=running, C=last, V=sum of accepted qty."""
    clock = _Clock(SLOT_A + 10)
    monkeypatch.setattr(broker.time, "time", clock)
    polls = iter([
        (200, [_print(1900.0, 2, 100), _print(1905.0, 1, 101)]),
        (200, [_print(1905.0, 1, 101), _print(1895.0, 3, 104)]),  # first is a replay
    ])
    b = _broker(fake_client, get_latest_trade=lambda *a, **k: next(polls))

    u1 = asyncio.run(b.watch_ohlcv("VN30F1M", "5"))
    assert not u1.is_closed
    assert (u1.open, u1.high, u1.low, u1.close, u1.volume) == (1900.0, 1905.0, 1900.0, 1905.0, 3.0)

    u2 = asyncio.run(b.watch_ohlcv("VN30F1M", "5"))
    assert not u2.is_closed
    assert (u2.open, u2.high, u2.low, u2.close, u2.volume) == (1900.0, 1905.0, 1895.0, 1895.0, 6.0), \
        "the replayed print must not inflate volume; the new print extends L and C"


def __test_rollover_holds_back_forming_and_official_close_wins__(fake_client, monkeypatch):
    """Emit-ordering guard: after the slot rolls over, the NEXT update must be
    slot A's CLOSED bar — with the venue's OFFICIAL values (differing from the
    synth) — never a forming update for slot B."""
    clock = _Clock(SLOT_A + 10)
    monkeypatch.setattr(broker.time, "time", clock)
    b = _broker(fake_client,
                get_latest_trade=(200, [_print(1900.0, 2, 100)]),
                get_ohlc=(200, {"t": [SLOT_A], "o": [1899.5], "h": [1906.0],
                                "l": [1898.0], "c": [1903.5], "v": [40.0]}))
    forming = asyncio.run(b.watch_ohlcv("VN30F1M", "5"))
    assert not forming.is_closed and forming.close == 1900.0

    clock.now = SLOT_B + 1              # slot rolls over
    closed = asyncio.run(b.watch_ohlcv("VN30F1M", "5"))
    assert closed.is_closed and closed.timestamp == SLOT_A * 1000
    assert (closed.open, closed.high, closed.low, closed.close, closed.volume) == \
        (1899.5, 1906.0, 1898.0, 1903.5, 40.0), \
        "the OFFICIAL row is authoritative over the synthesized bar"


def __test_withheld_official_close_falls_back_to_synth_loudly__(
        fake_client, monkeypatch, caplog):
    """Live-L4-T03: the session-final row is withheld ~+903 s — after
    tick_close_timeout the SYNTHESIZED bar closes with a warning instead of
    stalling the feed through the session close."""
    clock = _Clock(SLOT_A + 10)
    monkeypatch.setattr(broker.time, "time", clock)
    b = _broker(fake_client,
                get_latest_trade=(200, [_print(1900.0, 2, 100)]),
                get_ohlc=(200, {"t": [], "o": [], "h": [], "l": [], "c": [], "v": []}))
    forming = asyncio.run(b.watch_ohlcv("VN30F1M", "5"))
    assert not forming.is_closed

    clock.now = SLOT_B + 1
    caplog.clear()

    async def _advance_past_deadline(_delay, result=None):
        clock.now += b._tick_close_timeout + 1
        return result
    monkeypatch.setattr(asyncio, "sleep", _advance_past_deadline)

    closed = asyncio.run(b.watch_ohlcv("VN30F1M", "5"))
    assert closed.is_closed and closed.timestamp == SLOT_A * 1000
    assert closed.close == 1900.0, "synth values close the bar when the venue withholds"
    assert any("SYNTH" in r.message for r in caplog.records), \
        "a synth-closed bar must be LOUD (L4 grades who-closed)"


def __test_throttle_transition_logged_once__(fake_client, monkeypatch, caplog):
    """429 on the tick bucket is currently silent repo-wide — tick mode must
    announce the degradation exactly once, and announce recovery."""
    clock = _Clock(SLOT_A + 10)
    monkeypatch.setattr(broker.time, "time", clock)
    replies = iter([(429, {}), (429, {}), (200, [_print(1900.0, 1, 100)])])
    b = _broker(fake_client, get_latest_trade=lambda *a, **k: next(replies))

    caplog.clear()
    update = asyncio.run(b.watch_ohlcv("VN30F1M", "5"))
    assert not update.is_closed and update.close == 1900.0
    throttled = [r for r in caplog.records if "throttled" in r.message]
    cleared = [r for r in caplog.records if "throttle cleared" in r.message]
    assert len(throttled) == 1, "the 429 transition must be logged exactly once"
    assert len(cleared) == 1
