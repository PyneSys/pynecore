"""
The provider hub's server: the process that holds the real providers.

One real provider instance exists per plugin, symbol, timeframe and data directory,
however many clients opened it. Its blocking calls (symbol info, a history
download) run on worker threads, one at a time per instance; its connection and its
``watch_ohlcv`` loops run on the hub's event loop and are shared:

- the connection is opened by the first client that connects and closed when the
  last one left. A client that connects later joins it: the provider's own
  ``connect()`` is not run again, as that resets what the provider tracks of the
  stream.
- every client names the connection it saw (its generation) when it connects. A
  client that saw the current one and connects again reports it dead or stale: the
  hub reconnects once. A client that reports an older one joins the new
  connection.
- one ``watch_ohlcv`` loop runs per watched symbol and timeframe and hands every
  update to each client watching it. Its exception goes to each of them too, so
  every client's live runner reacts as it does on a provider of its own.

Nothing a provider answered is cached: an empty ``backfill_closed_bars`` answer
means "not yet" until the venue's publish lag is over, so the next client's
question must reach the venue again.
"""
import asyncio
import hmac
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from ...config import ensure_config
from .. import load_installed_plugin
from ..live_provider import LiveProviderPlugin
from ..provider import ProviderPlugin
from ...symbol_map import SymbolMap
from ....types.ohlcv import OHLCV
from . import protocol

logger = logging.getLogger(__name__)

#: Class-level calls of a provider plugin a client may make.
CLASS_METHODS = frozenset({
    "to_tradingview_timeframe", "to_exchange_timeframe", "get_ohlcv_path",
    "get_list_of_brokers", "construct_pair_symbol",
})

#: Blocking calls of a provider instance a client may make.
INSTANCE_METHODS = frozenset({
    "update_symbol_info", "get_list_of_symbols", "normalize_symbol",
})

#: Attributes of a real provider instance its remote twin takes over.
INSTANCE_ATTRIBUTES = ("symbol", "timezone", "xchg_timeframe", "fetch_all_by_default",
                       "extra_data")

#: Bars of a download sent to the client in one event.
DOWNLOAD_CHUNK_BARS = 500

#: Bytes a connection may hold unsent: a client that falls this far behind is dropped,
#: as it would otherwise grow the hub without bound. It connects again as after any
#: lost connection.
MAX_UNSENT_BYTES = 4 * protocol.MAX_FRAME

#: Seconds a leaving client gets to take what is still queued for it before its
#: connection is cut.
CLOSE_TIMEOUT = 5.0


def live_contract_attributes() -> tuple[str, ...]:
    """The plain-value attributes :class:`LiveProviderPlugin` declares.

    They are the stream's contract with the live runner (reconnect delays, the feed
    watchdog, how the venue's history behaves), so a remote provider carries the
    values of the real instance. Read off the class, a contract attribute the core
    gains later is carried without a change here.

    :return: The attribute names.
    """
    return tuple(name for name, value in vars(LiveProviderPlugin).items()
                 if not name.startswith("_")
                 and (value is None or isinstance(value, (bool, int, float))))


class _DownloadCancelled(Exception):
    """The client gave up a download: raised inside the provider's own callbacks."""


class _Connection:
    """One client connection. Everything sent goes through one writer task, so
    frames never interleave and a slow client never holds the event loop."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self._writer = writer
        self._outbox: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._unsent = 0
        """Bytes queued and not yet handed to the transport."""
        self._pump = asyncio.ensure_future(self._write())
        self.closed = False
        self.instance: '_Instance | None' = None
        """The instance this connection opened, on a call connection."""
        self.handle: int | None = None
        """What a stream connection names that instance by."""
        self.attached: '_Instance | None' = None
        """The instance whose shared connection this one takes part in."""
        self.seen_generation: int | None = None

    def post(self, message: dict[str, Any]) -> None:
        """Queue a message for the client; dropped once the connection closed.

        A client with more than :data:`MAX_UNSENT_BYTES` unsent is cut off: what is
        queued for it is discarded and its connection aborted, so it never sees a
        message out of order.
        """
        if self.closed:
            return
        data = protocol.frame(protocol.dumps(message))
        self._unsent += len(data)
        if self._unsent > MAX_UNSENT_BYTES:
            logger.warning("provider hub: dropping a client that stopped reading")
            self.abort()
            return
        self._outbox.put_nowait(data)

    def abort(self) -> None:
        """Discard what is queued and cut the connection at once."""
        self.closed = True
        while not self._outbox.empty():
            self._outbox.get_nowait()
        self._unsent = 0
        self._outbox.put_nowait(None)
        self._writer.transport.abort()

    def answer(self, request: dict[str, Any], value: Any) -> None:
        self.post({"ok": True, "value": value, "id": request.get("id")})

    def refuse(self, request: dict[str, Any], exc: BaseException) -> None:
        self.post({"ok": False, "error": protocol.encode_error(exc), "id": request.get("id")})

    async def _write(self) -> None:
        try:
            while True:
                data = await self._outbox.get()
                if data is None:
                    break
                self._unsent -= len(data)
                self._writer.write(data)
                await self._writer.drain()
        except (OSError, RuntimeError):
            pass
        finally:
            self.closed = True
            self._writer.close()

    async def close(self) -> None:
        """Send what is queued, then close the connection.

        A client that does not take it within :data:`CLOSE_TIMEOUT` is cut off.
        """
        self._outbox.put_nowait(None)
        done, _ = await asyncio.wait({self._pump}, timeout=CLOSE_TIMEOUT)
        if not done:
            logger.warning("provider hub: cutting off a leaving client that stopped reading")
            self.abort()
            await asyncio.wait({self._pump})


class _Watch:
    """One ``watch_ohlcv`` loop and the clients it feeds."""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.subscribers: set[_Connection] = set()
        self.task: asyncio.Task[None] | None = None


class _Instance:
    """One real provider and what the hub shares of it."""

    def __init__(self, key: tuple[Any, ...], real: ProviderPlugin):
        self.key = key
        self.real = real
        self.openers: set[_Connection] = set()
        self.attached: set[_Connection] = set()
        self.call_lock = asyncio.Lock()
        """Blocking calls run one at a time: the instance carries per-call state."""
        self.connection_lock = asyncio.Lock()
        self.connected = False
        self.generation = 0
        """Counts the connections opened: clients tell them apart by it."""
        self.watches: dict[tuple[str, str], _Watch] = {}

    @property
    def live(self) -> LiveProviderPlugin:
        assert isinstance(self.real, LiveProviderPlugin)
        return self.real


class ProviderHub:
    """Serves the providers of one working directory to any number of clients.

    :param token: What every connection must send first.
    :param config_dir: The directory holding the plugin configs (``plugins/<name>.toml``)
        and the global symbol map; ``None`` runs every provider without a config.
    :param load: Resolves a plugin name to its class.
    """

    def __init__(self, token: str, *, config_dir: Path | None = None,
                 load: Callable[[str], type] = load_installed_plugin):
        self._token = token.encode()
        self._config_dir = config_dir
        self._load = load
        self._symbol_map = SymbolMap.load(config_dir)
        self._instances: dict[tuple[Any, ...], _Instance] = {}
        self._handles: dict[int, _Instance] = {}
        self._next_handle = 1
        self._server: asyncio.Server | None = None
        self._clients: set[asyncio.Task[None]] = set()
        self._connections: set[_Connection] = set()
        # The cleanups of the clients that left: they outlive a cancelled client task
        self._leaving: set[asyncio.Future[None]] = set()

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Start listening.

        :param host: The address to listen on; the loopback one by default.
        :param port: The port, 0 for a free one.
        :return: The port the hub listens on.
        """
        self._server = await asyncio.start_server(self._accept, host, port)
        return self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        """Stop listening, drop every client and disconnect every provider."""
        if self._server is not None:
            self._server.close()
        # Nothing queued for a client matters once the hub stops; a client that
        # stopped reading would otherwise hold the shutdown
        for connection in list(self._connections):
            connection.abort()
        for task in list(self._clients):
            task.cancel()
        if self._clients:
            await asyncio.wait(self._clients)
        if self._leaving:
            await asyncio.wait(self._leaving)
        if self._server is not None:
            await self._server.wait_closed()

    async def _accept(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._clients.add(task)
        connection = _Connection(reader, writer)
        self._connections.add(connection)
        try:
            token = await protocol.read_frame(reader)
            if hmac.compare_digest(token, self._token):
                await self._serve(connection)
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception:  # noqa: BLE001 - one client's fault must not end the hub
            logger.exception("provider hub: a client connection failed")
        finally:
            self._clients.discard(task)
            leaving = asyncio.ensure_future(self._leave(connection))
            self._leaving.add(leaving)
            leaving.add_done_callback(self._leaving.discard)
            leaving.add_done_callback(lambda _: self._connections.discard(connection))
            await asyncio.shield(leaving)

    async def _serve(self, connection: _Connection) -> None:
        pending: 'asyncio.Task[bytes] | None' = None
        while True:
            reading = pending or asyncio.ensure_future(protocol.read_frame(connection.reader))
            pending = None
            try:
                request: dict[str, Any] = protocol.loads(await reading)
            except asyncio.CancelledError:
                reading.cancel()
                raise
            op = request.get("op")
            if op == "cancel":
                # Of a download that had ended by the time this arrived
                continue
            try:
                if op == "download":
                    pending = await self._download(connection, request)
                elif op == "backfill":
                    # Answered out of turn: the stream's updates must keep flowing
                    task = asyncio.ensure_future(self._backfill(connection, request))
                    self._clients.add(task)
                    task.add_done_callback(self._clients.discard)
                else:
                    connection.answer(request, await self._dispatch(connection, request))
            except asyncio.CancelledError:
                if pending is not None:
                    pending.cancel()
                raise
            except Exception as exc:  # noqa: BLE001 - the client gets every failure
                connection.refuse(request, exc)

    async def _dispatch(self, connection: _Connection, request: dict[str, Any]) -> Any:
        op = request.get("op")
        if op == "describe":
            return self._describe(request["plugin"])
        if op == "class_call":
            return await self._class_call(request)
        if op == "open":
            return await self._open(connection, request)
        if op == "call":
            return await self._call(connection, request)
        if op == "resolve_symbol":
            return await self._resolve_symbol(connection, request)
        if op == "connect":
            return await self._connect(connection, request)
        if op == "watch":
            return self._watch(connection, request)
        raise ValueError(f"unknown provider hub request: {op!r}")

    # --- Plugin classes ---

    def _provider_class(self, name: str) -> type[ProviderPlugin]:
        cls = self._load(name)
        if not (isinstance(cls, type) and issubclass(cls, ProviderPlugin)):
            raise TypeError(f"plugin {name!r} is not a data provider")
        return cls

    def _describe(self, name: str) -> dict[str, Any] | None:
        """What a client needs to build the remote class of a plugin, ``None`` for a
        plugin that is no data provider."""
        cls = self._load(name)
        if not (isinstance(cls, type) and issubclass(cls, ProviderPlugin)):
            return None
        return {
            "class_name": cls.__name__,
            "live": issubclass(cls, LiveProviderPlugin),
            "attributes": {
                "plugin_name": cls.plugin_name,
                "timezone": cls.timezone,
                "fetch_all_by_default": cls.fetch_all_by_default,
                "multi_broker": cls.multi_broker,
                "extra_data": cls.extra_data,
            },
        }

    async def _class_call(self, request: dict[str, Any]) -> Any:
        method = request["method"]
        if method not in CLASS_METHODS:
            raise ValueError(f"not a provider class call: {method!r}")
        cls = self._provider_class(request["plugin"])
        return await asyncio.to_thread(getattr(cls, method), *request["args"])

    # --- Instances ---

    async def _open(self, connection: _Connection, request: dict[str, Any]) -> dict[str, Any]:
        if connection.instance is not None:
            raise ValueError("this connection has opened a provider already")
        name: str = request["plugin"]
        ohlcv_dir: Path | None = request["ohlcv_dir"]
        key = (name, request["symbol"], request["timeframe"], ohlcv_dir)
        instance = self._instances.get(key)
        if instance is None:
            cls = self._provider_class(name)
            config = None
            config_cls: type | None = getattr(cls, "Config", None)
            if config_cls is not None and self._config_dir is not None:
                config = await asyncio.to_thread(
                    ensure_config, config_cls, self._config_dir / "plugins" / f"{name}.toml")
            real = await asyncio.to_thread(
                lambda: cls(symbol=request["symbol"], timeframe=request["timeframe"],
                            ohlcv_dir=ohlcv_dir, config=config))
            # Another client may have opened the same one while this one was built
            instance = self._instances.setdefault(key, _Instance(key, real))
        instance.openers.add(connection)
        connection.instance = instance
        handle = self._next_handle
        self._next_handle += 1
        self._handles[handle] = instance
        connection.handle = handle

        real = instance.real
        attributes = {name: getattr(real, name) for name in INSTANCE_ATTRIBUTES}
        if isinstance(real, LiveProviderPlugin):
            attributes.update({name: getattr(real, name) for name in live_contract_attributes()})
        return {
            "handle": handle,
            "attributes": attributes,
            # The one part of the config the core reads off a provider; the rest
            # (credentials) stays in the hub
            "symbol_map": dict(getattr(real.config, "symbol_map", None) or {}),
        }

    @staticmethod
    def _opened(connection: _Connection) -> _Instance:
        if connection.instance is None:
            raise ValueError("this connection has opened no provider")
        return connection.instance

    async def _call(self, connection: _Connection, request: dict[str, Any]) -> Any:
        method = request["method"]
        if method not in INSTANCE_METHODS:
            raise ValueError(f"not a provider call: {method!r}")
        instance = self._opened(connection)
        async with instance.call_lock:
            return await asyncio.to_thread(
                getattr(instance.real, method), *request["args"], **request.get("kwargs", {}))

    async def _resolve_symbol(self, connection: _Connection, request: dict[str, Any]) -> str:
        instance = self._opened(connection)
        real = instance.real

        def resolve() -> str:
            # The client's instance decides whether the global map applies: the
            # core sets it on the chart's provider only
            real.global_symbol_map = self._symbol_map if request["global_map"] else None
            real.provider_name = request["provider_name"]
            return real.resolve_symbol(request["pine_key"])

        async with instance.call_lock:
            return await asyncio.to_thread(resolve)

    async def _download(self, connection: _Connection,
                        request: dict[str, Any]) -> 'asyncio.Task[bytes] | None':
        """Run a history download, sending its bars and progress as events.

        :return: The read of the client's next request, when the download ended
            before one came.
        """
        instance = self._opened(connection)
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        chunk: list[OHLCV] = []

        def flush() -> None:
            if chunk:
                loop.call_soon_threadsafe(connection.post, {"ev": "bars", "bars": list(chunk)})
                chunk.clear()

        def capture(data: OHLCV | list[OHLCV]) -> None:
            if cancelled.is_set():
                raise _DownloadCancelled()
            if isinstance(data, OHLCV):
                chunk.append(data)
            else:
                chunk.extend(data)
            if len(chunk) >= DOWNLOAD_CHUNK_BARS:
                flush()

        def progress(current: datetime) -> None:
            if cancelled.is_set():
                raise _DownloadCancelled()
            flush()
            loop.call_soon_threadsafe(connection.post, {"ev": "progress", "time": current})

        def work() -> None:
            real = instance.real
            original_save = real.save_ohlcv_data
            real.save_ohlcv_data = capture  # type: ignore[method-assign]
            try:
                real.download_ohlcv(request["time_from"], request["time_to"],
                                    on_progress=progress, limit=request["limit"],
                                    with_extra=request["with_extra"])
                flush()
            finally:
                real.save_ohlcv_data = original_save  # type: ignore[method-assign]

        async def run() -> None:
            async with instance.call_lock:
                await asyncio.to_thread(work)

        job = asyncio.ensure_future(run())
        reading: 'asyncio.Task[bytes] | None' = asyncio.ensure_future(
            protocol.read_frame(connection.reader))
        try:
            while not job.done():
                await asyncio.wait({job, reading}, return_when=asyncio.FIRST_COMPLETED)
                if job.done() or not reading.done():
                    continue
                # Whatever the client says or however it left, the download is over
                cancelled.set()
                failure = reading.exception()
                if failure is not None:
                    await asyncio.wait({job})
                    raise failure
                reading = asyncio.ensure_future(protocol.read_frame(connection.reader))
        except asyncio.CancelledError:
            cancelled.set()
            reading.cancel()
            raise
        try:
            job.result()
        except _DownloadCancelled:
            connection.refuse(request, ConnectionAbortedError("the download was cancelled"))
        except Exception as exc:  # noqa: BLE001 - the client gets every failure
            connection.refuse(request, exc)
        else:
            connection.answer(request, None)
        return reading

    # --- The shared connection ---

    def _streamed(self, connection: _Connection, request: dict[str, Any]) -> _Instance:
        instance = self._handles.get(request["handle"])
        if instance is None:
            raise ConnectionError("the provider was closed in the hub")
        if not isinstance(instance.real, LiveProviderPlugin):
            raise TypeError(f"plugin {instance.key[0]!r} has no live data")
        return instance

    async def _connect(self, connection: _Connection, request: dict[str, Any]) -> int:
        instance = self._streamed(connection, request)
        seen: int | None = request["seen"]
        async with instance.connection_lock:
            instance.attached.add(connection)
            connection.attached = instance
            if instance.connected and seen != instance.generation:
                return instance.generation
            reconnect = instance.connected
            if reconnect:
                await self._disconnect(instance, lost=True)
            try:
                await instance.live.connect()
                if reconnect:
                    await instance.live.on_reconnect()
            except BaseException as exc:
                self._fail_watches(instance, exc)
                raise
            instance.connected = True
            instance.generation += 1
            for watch in instance.watches.values():
                self._run_watch(instance, watch)
            return instance.generation

    async def _disconnect(self, instance: _Instance, *, lost: bool) -> None:
        """Stop the watch loops and close the real connection; their clients stay."""
        tasks = [watch.task for watch in instance.watches.values() if watch.task is not None]
        for watch in instance.watches.values():
            watch.task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks)
        instance.connected = False
        # noinspection PyBroadException
        try:
            if lost:
                await instance.live.on_disconnect()
            await instance.live.disconnect()
        except Exception:  # noqa: BLE001 - a dead connection may refuse to close
            logger.debug("provider hub: disconnect of %r raised", instance.key, exc_info=True)

    @staticmethod
    def _fail_watches(instance: _Instance, exc: BaseException) -> None:
        """Tell every watching client the stream failed; they connect again."""
        error = protocol.encode_error(exc)
        for watch in instance.watches.values():
            for subscriber in watch.subscribers:
                subscriber.post({"ev": "error", "symbol": watch.symbol,
                                 "timeframe": watch.timeframe, "error": error})
        instance.watches.clear()

    def _watch(self, connection: _Connection, request: dict[str, Any]) -> None:
        instance = connection.attached
        if instance is None or not instance.connected:
            raise ConnectionError("the provider is not connected")
        key = (request["symbol"], request["timeframe"])
        watch = instance.watches.get(key)
        if watch is None:
            watch = instance.watches[key] = _Watch(*key)
        watch.subscribers.add(connection)
        self._run_watch(instance, watch)

    def _run_watch(self, instance: _Instance, watch: _Watch) -> None:
        if watch.task is not None or not watch.subscribers:
            return

        async def loop() -> None:
            try:
                while True:
                    bar = await instance.live.watch_ohlcv(watch.symbol, watch.timeframe)
                    event = {"ev": "bar", "symbol": watch.symbol,
                             "timeframe": watch.timeframe, "bar": bar}
                    for subscriber in watch.subscribers:
                        subscriber.post(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - every watching client gets it
                error = protocol.encode_error(exc)
                for subscriber in watch.subscribers:
                    subscriber.post({"ev": "error", "symbol": watch.symbol,
                                     "timeframe": watch.timeframe, "error": error})
                # The clients watch again after they reconnected
                if instance.watches.get((watch.symbol, watch.timeframe)) is watch:
                    del instance.watches[(watch.symbol, watch.timeframe)]

        watch.task = asyncio.ensure_future(loop())

    async def _backfill(self, connection: _Connection, request: dict[str, Any]) -> None:
        try:
            instance = connection.attached
            if instance is None:
                raise ConnectionError("the provider is not connected")
            bars = await instance.live.backfill_closed_bars(
                request["symbol"], request["timeframe"], request["since_ms"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the client gets every failure
            connection.refuse(request, exc)
        else:
            connection.answer(request, bars)

    # --- A client left ---

    async def _leave(self, connection: _Connection) -> None:
        attached = connection.attached
        if attached is not None:
            async with attached.connection_lock:
                attached.attached.discard(connection)
                for key, watch in list(attached.watches.items()):
                    watch.subscribers.discard(connection)
                    if not watch.subscribers:
                        del attached.watches[key]
                        if watch.task is not None:
                            watch.task.cancel()
                            await asyncio.wait({watch.task})
                if not attached.attached and attached.connected:
                    await self._disconnect(attached, lost=False)
        opened = connection.instance
        if opened is not None:
            if connection.handle is not None:
                self._handles.pop(connection.handle, None)
            opened.openers.discard(connection)
            if not opened.openers:
                async with opened.connection_lock:
                    if not opened.openers and self._instances.get(opened.key) is opened:
                        del self._instances[opened.key]
                        if opened.connected:
                            await self._disconnect(opened, lost=False)
        await connection.close()
