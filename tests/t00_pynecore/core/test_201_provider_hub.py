"""
A provider hub holds the real providers and every other process reaches them through
remote provider classes. A history download must give through the hub what it gives
on the provider itself, and several clients of one symbol must share one real
connection.
"""
import asyncio
import contextlib
import json
import pickle
import threading
import time
from pathlib import Path

import pytest

from pynecore.core.plugin import ProviderError
from pynecore.core.plugin.live_provider import LiveProviderPlugin
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV


class __test_helper_VenueDown(ProviderError):
    """A plugin's own transient failure: its type cannot travel to a client."""
    retryable = True


class __test_helper_FakeProvider(LiveProviderPlugin):
    """A live provider the tests drive by hand; counts what the hub does with it."""

    reconnect_delay = 0.25
    history_publish_lag = 7.0
    connects = 0
    disconnects = 0
    downloads_stopped_at = -1
    symbol_info_failure: BaseException | None = None
    feed: 'asyncio.Queue | None' = None

    @classmethod
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        return timeframe.rstrip('m')

    @classmethod
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        if not timeframe.isdigit():
            raise ValueError(f"bad timeframe {timeframe}")
        return f"{timeframe}m"

    def __init__(self, *, symbol=None, timeframe=None, ohlcv_dir=None, config=None):
        super().__init__(symbol=symbol, timeframe=timeframe, ohlcv_dir=ohlcv_dir, config=config)
        # As a multi-exchange plugin does: the instance reshapes its symbol
        self.symbol = (symbol or "").lower()
        self.connected = False

    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        return ["AAA", "BBB"]

    def update_symbol_info(self) -> SymInfo:
        failure = type(self).symbol_info_failure
        if failure is not None:
            raise failure
        return SymInfo(prefix="FAKE", description="Fake", ticker=self.symbol.upper(),
                       currency="USD", period="1", type="crypto", mintick=0.01,
                       pricescale=100, pointvalue=1.0, mincontract=1.0, opening_hours=[],
                       session_starts=[], session_ends=[])

    def download_ohlcv(self, time_from, time_to, on_progress=None, limit=None,
                       with_extra=False):
        for i in range(1000):
            try:
                if on_progress:
                    on_progress(time_from)
            except BaseException:
                type(self).downloads_stopped_at = i
                raise
            self.save_ohlcv_data(OHLCV(60_000 * (i + 1), 1.0, 2.0, 0.5, 1.5, 10.0))
            time.sleep(0.001)

    async def connect(self) -> None:
        cls = type(self)
        cls.connects += 1
        cls.feed = asyncio.Queue()
        self.connected = True

    async def disconnect(self) -> None:
        type(self).disconnects += 1
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        feed = type(self).feed
        assert feed is not None
        update = await feed.get()
        if isinstance(update, BaseException):
            raise update
        return update

    async def backfill_closed_bars(self, symbol, timeframe, since_ms):
        return [OHLCV(since_ms + 60_000, 1.0, 1.0, 1.0, 1.0, 1.0)]


@contextlib.contextmanager
def __test_helper_hub(monkeypatch, config_dir=None, load=None):
    """A hub on a thread of its own, and this process set up as its client."""
    from pynecore.core.plugin.hub import HUB_ENV, TOKEN_ENV, remote
    from pynecore.core.plugin.hub.server import ProviderHub

    loop = asyncio.new_event_loop()
    hub = ProviderHub("secret", config_dir=config_dir) if load is None \
        else ProviderHub("secret", config_dir=config_dir, load=load)
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    port = asyncio.run_coroutine_threadsafe(hub.start(), loop).result(10)
    monkeypatch.setenv(HUB_ENV, f"127.0.0.1:{port}")
    monkeypatch.setenv(TOKEN_ENV, "secret")
    remote._remote_classes.clear()
    remote._class_channel = None
    try:
        yield hub, loop
    finally:
        monkeypatch.delenv(HUB_ENV)
        remote._remote_classes.clear()
        remote._class_channel = None
        asyncio.run_coroutine_threadsafe(hub.close(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()


def __test_helper_fake_load(name: str) -> type:
    if name == "fake":
        return __test_helper_FakeProvider
    return dict


def __test_helper_reset_fake() -> None:
    fake = __test_helper_FakeProvider
    fake.connects = 0
    fake.disconnects = 0
    fake.downloads_stopped_at = -1
    fake.symbol_info_failure = None
    fake.feed = None


def __test_helper_feed(loop, update) -> None:
    """Hand the fake's stream one update, on the hub's loop."""
    feed = __test_helper_FakeProvider.feed
    assert feed is not None
    loop.call_soon_threadsafe(feed.put_nowait, update)


def __test_helper_write_fixture(tmp_path: Path, last_ts: int) -> tuple[Path, Path]:
    """A replay fixture of 10 closed warmup bars and one live bar, plus its config."""
    warmup = [[last_ts - (9 - i) * 60_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i, 1.0]
              for i in range(10)]
    live_ts = last_ts + 60_000
    live = [
        [live_ts, 110.0, 110.5, 110.0, 110.5, 1.0, False],
        [live_ts, 110.0, 111.0, 110.0, 111.0, 1.0, True],
    ]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"warmup": warmup, "live": live}), encoding="utf-8")
    config_dir = tmp_path / "config"
    (config_dir / "plugins").mkdir(parents=True)
    (config_dir / "plugins" / "replay.toml").write_text(
        f'fixture_path = "{fixture.as_posix()}"\n', encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir, config_dir


def __test_file_download_through_the_hub_matches_the_provider__(tmp_path, monkeypatch):
    """The OHLCV file a client writes from the hub's bars is the provider's own file"""
    from pynecore.core.ohlcv import OHLCVReader
    from pynecore.core.provider_download import download_provider_history

    last_ts = (int(time.time()) // 60 - 2) * 60_000
    data_dir, config_dir = __test_helper_write_fixture(tmp_path, last_ts)

    direct = download_provider_history("replay:TEST@1", "-5", data_dir=data_dir,
                                       config_dir=config_dir)
    assert direct.ohlcv_path is not None
    with OHLCVReader(direct.ohlcv_path) as reader:
        direct_bars = list(reader.read_from(0))
    direct.ohlcv_path.unlink()

    with __test_helper_hub(monkeypatch, config_dir):
        hubbed = download_provider_history("replay:TEST@1", "-5", data_dir=data_dir,
                                           config_dir=config_dir)
    assert hubbed.ohlcv_path == direct.ohlcv_path
    with OHLCVReader(hubbed.ohlcv_path) as reader:
        assert list(reader.read_from(0)) == direct_bars
    assert hubbed.time_from_ts == direct.time_from_ts
    assert hubbed.syminfo == direct.syminfo


def __test_remote_instance_carries_the_real_one__(monkeypatch):
    """A remote instance has the real instance's symbol and live contract values, and
    its class and instance calls answer as the plugin does"""
    from pynecore.core.plugin import load_plugin
    from pynecore.core.plugin.hub.remote import remote_plugin

    __test_helper_reset_fake()
    with __test_helper_hub(monkeypatch, load=__test_helper_fake_load):
        cls = load_plugin("fake")
        assert cls.__name__ == "__test_helper_FakeProvider"
        assert cls.to_exchange_timeframe("5") == "5m"
        with pytest.raises(ValueError, match="bad timeframe"):
            cls.to_exchange_timeframe("x")
        # Not a data provider: the plugin itself is loaded
        assert remote_plugin("cli") is None

        provider = cls(symbol="AAA", timeframe="1")
        assert provider.symbol == "aaa"
        assert provider.provider_symbol == "AAA"
        assert provider.xchg_timeframe == "1m"
        assert provider.reconnect_delay == 0.25
        assert provider.history_publish_lag == 7.0
        assert provider.get_list_of_symbols() == ["AAA", "BBB"]
        assert provider.update_symbol_info().ticker == "AAA"
        provider.close()


def __test_remote_failures_classify_as_the_real_ones__(monkeypatch):
    """A failure in the hub reaches the client with its type when that can travel, and
    always with its retry classification"""
    from pynecore.core.plugin import is_retryable_provider_error, load_plugin
    from pynecore.core.plugin.hub.protocol import RemoteNetworkError, RemoteProviderError

    __test_helper_reset_fake()
    fake = __test_helper_FakeProvider
    with __test_helper_hub(monkeypatch, load=__test_helper_fake_load):
        provider = load_plugin("fake")(symbol="AAA", timeframe="1")

        fake.symbol_info_failure = __test_helper_VenueDown("maintenance")
        with pytest.raises(RemoteProviderError, match="maintenance") as caught:
            provider.update_symbol_info()
        assert is_retryable_provider_error(caught.value)
        assert not isinstance(caught.value, OSError)

        fake.symbol_info_failure = ProviderError("unknown symbol")
        with pytest.raises(ProviderError, match="unknown symbol") as caught:
            provider.update_symbol_info()
        assert type(caught.value) is ProviderError
        assert not is_retryable_provider_error(caught.value)

        failure = __test_helper_VenueDown("wrapped")
        failure.__cause__ = ConnectionResetError("reset by peer")
        fake.symbol_info_failure = failure
        with pytest.raises(RemoteNetworkError) as caught:
            provider.update_symbol_info()
        assert isinstance(caught.value, OSError)
        provider.close()


def __test_download_stopped_by_the_client_stops_in_the_hub__(monkeypatch):
    """A progress callback that raises ends the hub's download, and the provider's
    call connection serves the next request"""
    from datetime import datetime
    from pynecore.core.plugin import load_plugin

    __test_helper_reset_fake()
    with __test_helper_hub(monkeypatch, load=__test_helper_fake_load):
        provider = load_plugin("fake")(symbol="AAA", timeframe="1")
        saved: list[OHLCV] = []
        provider.save_ohlcv_data = saved.extend  # type: ignore[method-assign]
        calls = []

        def stop_at_third(_current) -> None:
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            provider.download_ohlcv(datetime(2026, 1, 1), datetime(2026, 1, 2),
                                    on_progress=stop_at_third)
        assert 0 < __test_helper_FakeProvider.downloads_stopped_at < 1000
        assert provider.update_symbol_info().ticker == "AAA"
        provider.close()


def __test_clients_of_one_symbol_share_one_connection__(monkeypatch):
    """The real provider connects once for any number of clients, each gets every
    update, and the connection closes when the last client left"""
    from pynecore.core.plugin import load_plugin

    __test_helper_reset_fake()
    fake = __test_helper_FakeProvider
    with __test_helper_hub(monkeypatch, load=__test_helper_fake_load) as (hub, loop):
        cls = load_plugin("fake")
        first = cls(symbol="AAA", timeframe="1")
        second = cls(symbol="AAA", timeframe="1")
        assert len(hub._instances) == 1

        async def run() -> None:
            await first.connect()
            assert first.is_connected
            first_bar = asyncio.ensure_future(first.watch_ohlcv("aaa", "1"))
            await asyncio.sleep(0.05)
            # A later client joins the connection: the real connect() is not run again
            await second.connect()
            second_bar = asyncio.ensure_future(second.watch_ohlcv("aaa", "1"))
            await asyncio.sleep(0.05)
            assert fake.connects == 1

            bar = OHLCV(60_000, 1.0, 2.0, 0.5, 1.5, 10.0, None, False)
            __test_helper_feed(loop, bar)
            assert await asyncio.wait_for(first_bar, 5) == bar
            assert await asyncio.wait_for(second_bar, 5) == bar
            assert await first.backfill_closed_bars("aaa", "1", 60_000) == [
                OHLCV(120_000, 1.0, 1.0, 1.0, 1.0, 1.0)]

            await first.disconnect()
            await asyncio.sleep(0.05)
            assert fake.disconnects == 0
            closed = OHLCV(60_000, 1.0, 2.0, 0.5, 1.8, 12.0)
            __test_helper_feed(loop, closed)
            assert await asyncio.wait_for(second.watch_ohlcv("aaa", "1"), 5) == closed

            await second.disconnect()
            await asyncio.sleep(0.05)
            assert fake.disconnects == 1

        asyncio.run(run())
        first.close()
        second.close()


def __test_a_failed_stream_reconnects_once_for_all_clients__(monkeypatch):
    """Every client gets the stream's failure; the first one connecting again makes
    the hub reconnect, the others join the new connection"""
    from pynecore.core.plugin import is_retryable_provider_error, load_plugin

    __test_helper_reset_fake()
    fake = __test_helper_FakeProvider
    with __test_helper_hub(monkeypatch, load=__test_helper_fake_load) as (hub, loop):
        cls = load_plugin("fake")
        first = cls(symbol="AAA", timeframe="1")
        second = cls(symbol="AAA", timeframe="1")

        async def run() -> None:
            await first.connect()
            await second.connect()
            first_bar = asyncio.ensure_future(first.watch_ohlcv("aaa", "1"))
            second_bar = asyncio.ensure_future(second.watch_ohlcv("aaa", "1"))
            await asyncio.sleep(0.05)
            __test_helper_feed(loop, __test_helper_VenueDown("socket lost"))
            for waiting in (first_bar, second_bar):
                with pytest.raises(ProviderError, match="socket lost") as caught:
                    await asyncio.wait_for(waiting, 5)
                assert is_retryable_provider_error(caught.value)

            # The live runner's reconnect, in each client
            await first.disconnect()
            await first.connect()
            assert fake.connects == 2
            await second.disconnect()
            await second.connect()
            assert fake.connects == 2

            bar = OHLCV(60_000, 1.0, 2.0, 0.5, 1.5, 10.0)
            first_bar = asyncio.ensure_future(first.watch_ohlcv("aaa", "1"))
            second_bar = asyncio.ensure_future(second.watch_ohlcv("aaa", "1"))
            await asyncio.sleep(0.05)
            __test_helper_feed(loop, bar)
            assert await asyncio.wait_for(first_bar, 5) == bar
            assert await asyncio.wait_for(second_bar, 5) == bar
            await first.disconnect()
            await second.disconnect()

        asyncio.run(run())
        first.close()
        second.close()


def __test_a_stale_feed_reconnect_keeps_the_other_clients_watching__(monkeypatch):
    """A client that reconnects a silent feed gets a new real connection, and a
    client that never noticed goes on receiving from it"""
    from pynecore.core.plugin import load_plugin

    __test_helper_reset_fake()
    fake = __test_helper_FakeProvider
    with __test_helper_hub(monkeypatch, load=__test_helper_fake_load) as (hub, loop):
        cls = load_plugin("fake")
        first = cls(symbol="AAA", timeframe="1")
        second = cls(symbol="AAA", timeframe="1")

        async def run() -> None:
            await first.connect()
            await second.connect()
            waiting = asyncio.ensure_future(second.watch_ohlcv("aaa", "1"))
            await asyncio.sleep(0.05)

            await first.disconnect()
            await first.connect()
            assert fake.connects == 2

            bar = OHLCV(60_000, 1.0, 2.0, 0.5, 1.5, 10.0)
            await asyncio.sleep(0.05)
            __test_helper_feed(loop, bar)
            assert await asyncio.wait_for(waiting, 5) == bar
            await first.disconnect()
            await second.disconnect()

        asyncio.run(run())
        first.close()
        second.close()


def __test_a_frame_holds_only_core_and_value_types__():
    """A frame naming a function or a foreign class is refused before anything runs"""
    from pynecore.core.plugin.hub import protocol

    bar = OHLCV(60_000, 1.0, 2.0, 0.5, 1.5, 10.0)
    assert protocol.loads(protocol.dumps({"bars": [bar], "path": Path("/tmp/x")})) == {
        "bars": [bar], "path": Path("/tmp/x")}
    for refused in (print, contextlib.ExitStack, threading.Thread):
        with pytest.raises(pickle.UnpicklingError):
            protocol.loads(pickle.dumps(refused))
    # A function of the core is no value either
    from pynecore.core.plugin import load_plugin
    with pytest.raises(pickle.UnpicklingError):
        protocol.loads(pickle.dumps(load_plugin))


def __test_a_wrong_token_is_dropped__(monkeypatch):
    """A connection that does not start with the hub's token gets no answer"""
    from pynecore.core.plugin.hub import TOKEN_ENV
    from pynecore.core.plugin.hub.remote import remote_plugin

    with __test_helper_hub(monkeypatch, load=__test_helper_fake_load):
        monkeypatch.setenv(TOKEN_ENV, "wrong")
        with pytest.raises(ConnectionError):
            remote_plugin("fake")


def __test_a_client_that_stops_reading_is_dropped__(monkeypatch):
    """A client that falls too far behind is cut off: the hub keeps no unbounded
    backlog for it, and it sees no message out of order"""
    from pynecore.core.plugin.hub import server

    monkeypatch.setattr(server, "MAX_UNSENT_BYTES", 4096)

    async def run() -> None:
        accepted: asyncio.Future = asyncio.get_running_loop().create_future()

        async def accept(reader, writer):
            accepted.set_result(server._Connection(reader, writer))

        listener = await asyncio.start_server(accept, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        connection = await accepted
        # Posted with no chance for the writer to run, as a burst of updates is
        for i in range(100):
            connection.post({"ev": "bar", "n": i, "pad": "x" * 100})
        assert connection.closed
        assert connection._unsent == 0
        assert connection._outbox.qsize() <= 1
        await connection.close()
        with contextlib.suppress(ConnectionError):
            assert await reader.read() == b""
        writer.close()
        listener.close()
        await listener.wait_closed()

    asyncio.run(run())


def __test_a_leaving_client_that_stops_reading_is_cut_off__(monkeypatch):
    """Closing a connection whose client keeps its socket open without reading ends
    after the close timeout instead of waiting for the client forever"""
    from pynecore.core.plugin.hub import server

    monkeypatch.setattr(server, "CLOSE_TIMEOUT", 0.2)

    async def run() -> None:
        accepted: asyncio.Future = asyncio.get_running_loop().create_future()

        async def accept(reader, writer):
            accepted.set_result(server._Connection(reader, writer))

        listener = await asyncio.start_server(accept, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        connection = await accepted
        # More than the socket buffers hold, far less than the unsent limit
        for i in range(16):
            connection.post({"ev": "bar", "n": i, "pad": "x" * (1024 * 1024)})
        await asyncio.wait_for(connection.close(), 5)
        assert connection.closed
        writer.close()
        listener.close()
        await listener.wait_closed()

    asyncio.run(run())


def __test_hub_shutdown_does_not_wait_for_a_client_that_stops_reading__():
    """The hub stops at once, even with a client that holds its socket open and
    reads nothing of what is queued for it"""
    from pynecore.core.plugin.hub import protocol
    from pynecore.core.plugin.hub.server import ProviderHub

    async def run() -> None:
        hub = ProviderHub("secret", load=__test_helper_fake_load)
        port = await hub.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(protocol.frame(b"secret"))
        await writer.drain()
        while not hub._connections:
            await asyncio.sleep(0)
        connection = next(iter(hub._connections))
        for i in range(16):
            connection.post({"ev": "bar", "n": i, "pad": "x" * (1024 * 1024)})
        await asyncio.wait_for(hub.close(), 2)
        assert connection.closed
        writer.close()

    asyncio.run(run())
