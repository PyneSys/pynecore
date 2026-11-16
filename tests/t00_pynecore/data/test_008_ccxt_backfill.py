"""
The CCXT provider's closed-bar gap fill (``backfill_closed_bars``), offline against a
fake async client: it walks the window in chunks of the exchange's per-request cap,
returns only closed candles strictly after the requested open, ascending, and leaves
the candle still forming to the stream.
"""
import asyncio
import time

import pytest

from pynecore.types.ohlcv import OHLCV

_MINUTE_MS = 60_000


def _provider():
    pytest.importorskip("ccxt")
    from pynecore.providers.ccxt import CCXTProvider
    return CCXTProvider(symbol="BINANCE:BTC/USDT", timeframe="1", ohlcv_dir=None)


class _FakeClient:
    """``fetch_ohlcv`` over a fixed candle list, with CCXT's ``since``/``limit`` semantics.

    Answers each chunk in a scrambled order, so the provider has to sort it.
    """

    def __init__(self, candles: list[list[float]]):
        self.candles = candles
        self.calls: list[tuple[str, str, int, int]] = []

    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int):
        self.calls.append((symbol, timeframe, since, limit))
        chunk = [c for c in self.candles if c[0] >= since][:limit]
        return list(reversed(chunk))


def _candles(first_open: int, count: int) -> list[list[float]]:
    return [[first_open + i * _MINUTE_MS, 10.0 + i, 11.0 + i, 9.0 + i, 10.5 + i, 1.0 + i]
            for i in range(count)]


def _run(provider, since_ms: int) -> list[OHLCV]:
    return asyncio.run(provider.backfill_closed_bars("BTC/USDT", "1", since_ms))


def __test_closed_bars_after_since_in_chunks__():
    """Ten closed minutes after ``since`` come back in chunks of three, without the forming one"""
    provider = _provider()
    forming = (int(time.time() * 1000) // _MINUTE_MS) * _MINUTE_MS
    since = forming - 11 * _MINUTE_MS
    # ``since`` itself, ten closed minutes after it, and the forming one
    client = _FakeClient(_candles(since, 12))
    provider._async_client = client
    provider._chunk_limit = lambda _tf: 3

    bars = _run(provider, since)

    assert [bar.timestamp for bar in bars] == [since + i * _MINUTE_MS for i in range(1, 11)]
    assert all(bar.is_closed for bar in bars)
    assert bars[0].close == 11.5 and bars[0].volume == 2.0
    # Each chunk starts right after the newest candle of the previous one
    assert [call[2] for call in client.calls] == [
        since + 1, since + 3 * _MINUTE_MS + 1, since + 6 * _MINUTE_MS + 1,
        since + 9 * _MINUTE_MS + 1]
    assert all(call[:2] == ("BTC/USDT", "1m") and call[3] == 3 for call in client.calls)


def __test_only_the_forming_bar_left__():
    """When the forming candle is the only one after ``since``, nothing comes back"""
    provider = _provider()
    forming = (int(time.time() * 1000) // _MINUTE_MS) * _MINUTE_MS
    client = _FakeClient(_candles(forming - _MINUTE_MS, 2))
    provider._async_client = client

    assert _run(provider, forming - _MINUTE_MS) == []


def __test_exchange_over_answering_is_filtered__():
    """Candles at or before ``since`` that an exchange returns anyway are dropped"""
    provider = _provider()
    forming = (int(time.time() * 1000) // _MINUTE_MS) * _MINUTE_MS
    held = forming - 3 * _MINUTE_MS

    class _IgnoresSince(_FakeClient):
        async def fetch_ohlcv(self, symbol, timeframe, since, limit):
            self.calls.append((symbol, timeframe, since, limit))
            return list(self.candles)

    client = _IgnoresSince(_candles(held - 2 * _MINUTE_MS, 6))
    provider._async_client = client

    bars = _run(provider, held)

    assert [bar.timestamp for bar in bars] == [held + _MINUTE_MS, held + 2 * _MINUTE_MS]
    assert len(client.calls) == 1


def __test_without_a_live_client_nothing_is_queried__():
    """Before ``connect()`` (or after ``disconnect()``) the fill returns nothing"""
    provider = _provider()
    assert provider._async_client is None
    assert _run(provider, 0) == []


def __test_idle_bars_in_the_history_depend_on_the_exchange__():
    """An exchange whose history skips periods without trades says so to the stream"""
    pytest.importorskip("ccxt")
    from pynecore.providers.ccxt import CCXTProvider
    assert _provider().history_has_idle_bars is True
    coinbase = CCXTProvider(symbol="COINBASE:BTC/USD", timeframe="1", ohlcv_dir=None)
    assert coinbase.history_has_idle_bars is False


def __test_rollover_closes_with_the_final_record_of_the_batch__():
    """The batch that opens the next candle also updates the previous one: that is its close

    Without the previous candle in the batch, the last update seen of it closes it.
    """
    provider = _provider()
    t0 = 1_700_000_040_000
    t1 = t0 + _MINUTE_MS

    class _Socket:
        def __init__(self, batches: list[list[list[float]]]):
            self.batches = batches

        async def watch_ohlcv(self, _symbol, _timeframe):
            return self.batches.pop(0)

    provider._async_client = _Socket([
        [[t0, 100.0, 101.0, 99.0, 100.0, 1.0]],
        [[t0, 100.0, 110.0, 99.0, 109.0, 5.0], [t1, 109.0, 109.0, 109.0, 109.0, 0.5]],
        [[t1, 109.0, 112.0, 108.0, 111.0, 3.0]],
        [[t1 + _MINUTE_MS, 111.0, 111.0, 111.0, 111.0, 0.1]],
    ])

    async def _updates() -> list[OHLCV]:
        return [await provider.watch_ohlcv("BTC/USDT", "1") for _ in range(4)]

    forming, closed, forming_next, closed_next = asyncio.run(_updates())

    assert not forming.is_closed and forming.close == 100.0
    assert closed.is_closed and closed.timestamp == t0
    assert (closed.high, closed.close, closed.volume) == (110.0, 109.0, 5.0)
    assert not forming_next.is_closed and forming_next.timestamp == t1
    assert closed_next.is_closed and closed_next.timestamp == t1
    assert (closed_next.close, closed_next.volume) == (111.0, 3.0)


def __test_monthly_close_follows_the_calendar__():
    """A monthly candle closes at the start of the next month, not after 30.4 days"""
    pytest.importorskip("ccxt")
    from datetime import datetime, UTC
    # noinspection PyProtectedMember
    from pynecore.providers.ccxt import _scheduled_close_ms

    february = int(datetime(2026, 2, 1, tzinfo=UTC).timestamp() * 1000)
    assert _scheduled_close_ms(february, "1M") == int(
        datetime(2026, 3, 1, tzinfo=UTC).timestamp() * 1000)
    december = int(datetime(2025, 12, 1, tzinfo=UTC).timestamp() * 1000)
    assert _scheduled_close_ms(december, "3M") == int(
        datetime(2026, 3, 1, tzinfo=UTC).timestamp() * 1000)
    assert _scheduled_close_ms(february, "60") == february + 3_600_000
