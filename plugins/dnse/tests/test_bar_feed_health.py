"""#81 — the bar feed must never go blind silently.

Measured live 2026-09-07 11:02-11:10: the engine consumed no bar for 8+ min
while a direct probe showed the venue serving fresh bars. Root findings
(card #81 panel, leader-verified):

- The out-of-loop staleness watchdog ALREADY EXISTS in core
  (``feed_stale_after = feed_timeout_bars × tf_seconds``, clock paused
  outside ``opening_hours``, stale → forced reconnect). It was silent only
  because DNSE declared ``feed_timeout_bars = 40`` (2400 s at 1m). Armed at
  16 — the smallest value that clears the in-session 14:30-14:45 ATC bar
  gap (15 bars @1m); anything under 16 false-reconnects daily.
- The live runner re-enters ``watch_ohlcv`` under a ≤2 s ``wait_for``, so
  each coroutine instance sees ~one poll: failure counters must live on
  ``self`` or they can NEVER fire in production (the panel proved the naive
  in-loop counter was untestable-in-good-faith). The host-faithful test
  below drives the same cancel/re-enter cycle the runner does.
"""
import asyncio
import logging

import pytest
import pynecore.lib as lib

lib.bar_index = 0

from pynecore_dnse import broker


def _broker(fake_client, tmp_path, **client_responses):
    responses = {}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="t",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="1", config=config)
    instance._client = fake_client(**responses)
    instance._bar_poll_interval = 0.001
    return instance


def _loud_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def __test_persistent_bar_poll_failure_warns_host_faithfully__(
        fake_client, tmp_path, caplog):
    """Host-faithful (#81 panel P1): the runner cancels and re-enters
    ``watch_ohlcv`` every ≤2 s, so the failure count must SURVIVE coroutine
    teardown. Drive 30 wait_for(0.02) cancel/re-enter cycles against a
    permanently failing OHLC endpoint — a WARNING must still appear.
    Wrong implementation caught: a coroutine-local counter (resets per
    entry, warns never — the shape the first baseline could not
    discriminate)."""
    b = _broker(fake_client, tmp_path, get_ohlc=(500, {"message": "boom"}))
    b._bar_poll_warn_after = 10          # reachable within the drive budget
    caplog.clear()

    async def _run():
        for _ in range(30):
            try:
                await asyncio.wait_for(b.watch_ohlcv("VN30F1M", "1"),
                                       timeout=0.02)
            except (asyncio.TimeoutError, TimeoutError):
                continue                 # the runner's normal re-enter path
    with caplog.at_level(logging.DEBUG):
        asyncio.run(_run())

    assert b._client.count("get_ohlc") >= 10, "polls must actually run"
    assert b._bar_poll_failures >= 10, (
        "failure accounting reset across re-entry — counters must live on "
        "the instance, not the coroutine (#81)")
    assert _loud_records(caplog), (
        "a persistent failed-poll streak produced no WARNING+ across "
        "host-faithful re-entry cycles (#81)")


def __test_poll_success_resets_the_failure_streak__(fake_client, tmp_path):
    """A healthy poll resets the ladder — transient blips never warm-start
    the next episode's warning."""
    b = _broker(fake_client, tmp_path,
                get_ohlc=(200, {"t": [], "o": [], "h": [], "l": [],
                                "c": [], "v": []}))
    b._bar_poll_failures = 19            # one below the default threshold
    async def _run():
        try:
            await asyncio.wait_for(b.watch_ohlcv("VN30F1M", "1"), timeout=0.05)
        except (asyncio.TimeoutError, TimeoutError):
            pass
    asyncio.run(_run())
    assert b._bar_poll_failures == 0


def __test_feed_staleness_threshold_is_minutes_and_atc_safe__():
    """#81 threshold control (panel P3 — discriminating in BOTH directions):
    the core watchdog's effective staleness bound for DNSE at 1m must be
    (a) MINUTES, not hours — feed_timeout_bars=40 gave 2400 s and the
    measured 8-minute wedge sailed under it silently; and (b) at least the
    15-minute in-session ATC bar gap plus one bar — anything tighter
    false-reconnects every trading day at ~14:30+threshold. Wrong
    implementations caught: re-loosening to 'hours' (a), and over-eager
    tightening (b)."""
    from pynecore_dnse.provider import DNSEProvider

    bars = DNSEProvider.feed_timeout_bars
    assert bars is not None, "the watchdog must be ARMED (None disables it)"
    stale_1m = bars * 60
    assert stale_1m <= 20 * 60, (
        f"effective staleness at 1m is {stale_1m}s — the watchdog is "
        f"disarmed in practice again (#81: 8-min wedge, silent at 2400s)")
    assert stale_1m >= 16 * 60, (
        f"effective staleness at 1m is {stale_1m}s — under the 15-min ATC "
        f"bar gap: daily false reconnect at ~14:30 (#81 G1)")
