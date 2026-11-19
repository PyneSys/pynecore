"""
The client side of a provider hub: provider classes that forward to the hub.

:func:`remote_plugin` builds, for a provider plugin's name, a class that is a
:class:`~pynecore.core.plugin.provider.ProviderPlugin` (or a
:class:`~pynecore.core.plugin.live_provider.LiveProviderPlugin`) to the core but
holds no SDK, no market list and no venue connection: every call goes to the real
instance in the hub. The plugin's module is not imported here.

An instance keeps one blocking connection for its calls (symbol info, the history
download) and opens a second one, on the live runner's event loop, each time it
connects: the updates of the symbols it watches arrive on that.
"""
import asyncio
import os
import socket
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .. import override
from ..live_provider import LiveProviderPlugin
from ..provider import Broker, ProviderPlugin
from ...syminfo import SymInfo
from ....types.ohlcv import OHLCV
from . import HUB_ENV, TOKEN_ENV, protocol


def hub_address() -> tuple[str, int] | None:
    """The hub this process uses, from the environment.

    :return: Host and port, or ``None`` when no hub is set.
    """
    address = os.environ.get(HUB_ENV)
    if not address:
        return None
    host, _, port = address.rpartition(":")
    return host, int(port)


def _token() -> bytes:
    return os.environ.get(TOKEN_ENV, "").encode()


def _unreachable(exc: OSError) -> protocol.RemoteNetworkError:
    """The failure of the connection to the hub itself, as a provider failure that
    a retry can cure: a front end starts a hub that ended again."""
    return protocol.RemoteNetworkError(f"the provider hub cannot be reached: {exc}",
                                       f"builtins.{type(exc).__qualname__}", True)


class _Channel:
    """A blocking request connection to the hub. One request at a time."""

    def __init__(self) -> None:
        address = hub_address()
        if address is None:
            raise ConnectionError(f"no provider hub is set ({HUB_ENV})")
        self.broken = False
        """The connection failed: nothing more can be asked on it."""
        try:
            self._sock = socket.create_connection(address)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock.sendall(protocol.frame(_token()))
        except OSError as exc:
            raise _unreachable(exc) from exc
        self._lock = threading.Lock()

    def request(self, message: dict[str, Any],
                on_event: Callable[[dict[str, Any]], None] | None = None) -> Any:
        """Send a request and return its answer's value.

        :param message: The request.
        :param on_event: Receives the events the hub sends before the answer. When it
            raises, the hub is told to stop, the rest of the events are dropped and
            the exception is raised here once the hub answered.
        :return: The answer's value.
        :raises BaseException: What the hub answered with, see
            :func:`~pynecore.core.plugin.hub.protocol.decode_error`.
        :raises RemoteNetworkError: If the connection to the hub failed; the channel
            is :attr:`broken` then.
        """
        with self._lock:
            failure: BaseException | None = None
            try:
                self._sock.sendall(protocol.frame(protocol.dumps(message)))
                while True:
                    answer: dict[str, Any] = protocol.loads(protocol.recv_frame(self._sock))
                    if "ev" not in answer:
                        break
                    if failure is None and on_event is not None:
                        try:
                            on_event(answer)
                        except BaseException as exc:
                            failure = exc
                            self._sock.sendall(protocol.frame(protocol.dumps({"op": "cancel"})))
            except OSError as exc:
                self.broken = True
                if failure is not None:
                    raise failure
                raise _unreachable(exc) from exc
            if failure is not None:
                raise failure
            if not answer["ok"]:
                raise protocol.decode_error(answer["error"])
            return answer["value"]

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


_class_channel: _Channel | None = None
_class_channel_lock = threading.Lock()


def _class_request(message: dict[str, Any]) -> Any:
    """A request that concerns a plugin class, not an instance: sent on one
    connection the process keeps for them."""
    channel = _shared_class_channel()
    try:
        return _ask_class_channel(channel, message)
    except OSError:
        # The hub was started again: asked once more on a new connection, as a class
        # request changes nothing in the hub
        if not channel.broken:
            raise
    return _ask_class_channel(_shared_class_channel(), message)


def _shared_class_channel() -> _Channel:
    global _class_channel
    with _class_channel_lock:
        if _class_channel is None:
            _class_channel = _Channel()
        return _class_channel


def _ask_class_channel(channel: _Channel, message: dict[str, Any]) -> Any:
    global _class_channel
    try:
        return channel.request(message)
    except OSError:
        # The next request connects again
        with _class_channel_lock:
            if _class_channel is channel:
                _class_channel = None
        channel.close()
        raise


class RemoteProvider(ProviderPlugin[Any]):
    """A provider whose real instance lives in the hub.

    Never used as it is: :func:`remote_plugin` derives one class per plugin name.
    The config the core hands over is not used: the hub loads the plugin's config
    itself, and only its symbol map comes back (as :attr:`config`).
    """

    hub_plugin: str = ""
    """Entry point name of the plugin this class stands for."""

    @classmethod
    def _class_call(cls, method: str, *args: Any) -> Any:
        return _class_request({"op": "class_call", "plugin": cls.hub_plugin,
                               "method": method, "args": args})

    @classmethod
    @override
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        return cls._class_call("to_tradingview_timeframe", timeframe)

    @classmethod
    @override
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        return cls._class_call("to_exchange_timeframe", timeframe)

    @classmethod
    @override
    def get_ohlcv_path(cls, symbol: str, timeframe: str, ohlcv_dir: Path,
                       provider_name: str | None = None) -> Path:
        return cls._class_call("get_ohlcv_path", symbol, timeframe, ohlcv_dir, provider_name)

    @classmethod
    @override
    def get_list_of_brokers(cls) -> list[Broker]:
        return cls._class_call("get_list_of_brokers")

    @classmethod
    @override
    def construct_pair_symbol(cls, from_cur: str, to_cur: str) -> str:
        return cls._class_call("construct_pair_symbol", from_cur, to_cur)

    @override
    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlcv_dir: Path | None = None, config: Any = None):
        super().__init__(symbol=symbol, timeframe=timeframe, ohlcv_dir=ohlcv_dir, config=None)
        self._open_request = {"op": "open", "plugin": self.hub_plugin, "symbol": symbol,
                              "timeframe": timeframe, "ohlcv_dir": ohlcv_dir}
        self._channel: _Channel | None = None
        self._handle = ""
        self._open()

    def _open(self) -> _Channel:
        """Open the instance in the hub, on a call connection of its own.

        Also how the instance comes back after the hub was started again: the new
        hub knows nothing of the old one's instances.

        :return: The call connection.
        """
        previous, self._channel = self._channel, None
        if previous is not None:
            previous.close()
        channel = _Channel()
        try:
            opened = channel.request(self._open_request)
        except BaseException:
            channel.close()
            raise
        self._channel = channel
        self._handle = opened["handle"]
        for name, value in opened["attributes"].items():
            setattr(self, name, value)
        self.config = SimpleNamespace(symbol_map=opened["symbol_map"])
        return channel

    def _request(self, message: dict[str, Any],
                 on_event: Callable[[dict[str, Any]], None] | None = None, *,
                 repeatable: bool = True) -> Any:
        """Ask the hub on the instance's call connection, opening the instance again
        when that connection is gone.

        :param message: The request.
        :param on_event: Receives the events sent before the answer.
        :param repeatable: Whether the request is asked once more when the connection
            fails under it. Not for a request whose events were already acted on.
        :return: The answer's value.
        """
        channel = self._channel
        if channel is None or channel.broken:
            channel = self._open()
        try:
            return channel.request(message, on_event)
        except protocol.RemoteNetworkError:
            if not (repeatable and channel.broken):
                raise
        return self._open().request(message, on_event)

    def close(self) -> None:
        """Give the instance up in the hub. Closing the process does the same."""
        channel, self._channel = self._channel, None
        if channel is not None:
            channel.close()

    def __del__(self) -> None:
        channel = getattr(self, "_channel", None)
        if channel is not None:
            channel.close()

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self._request({"op": "call", "method": method, "args": args, "kwargs": kwargs})

    @override
    def normalize_symbol(self, symbol: str) -> str:
        return self._call("normalize_symbol", symbol)

    @override
    def resolve_symbol(self, pine_key: str) -> str:
        return self._request({
            "op": "resolve_symbol", "pine_key": pine_key,
            "global_map": bool(self.global_symbol_map), "provider_name": self.provider_name,
        })

    @override
    def get_list_of_symbols(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._call("get_list_of_symbols", *args, **kwargs)

    @override
    def update_symbol_info(self) -> SymInfo:
        return self._call("update_symbol_info")

    @override
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None, with_extra: bool = False) -> None:
        def on_event(event: dict[str, Any]) -> None:
            if event["ev"] == "bars":
                self.save_ohlcv_data(event["bars"])
            elif event["ev"] == "progress" and on_progress is not None:
                on_progress(event["time"])

        # Not asked twice: the bars of a failed pass are saved already
        self._request({
            "op": "download", "time_from": time_from, "time_to": time_to, "limit": limit,
            "with_extra": with_extra,
        }, on_event, repeatable=False)


class _Stream:
    """A live connection to the hub: the updates of the watched symbols and the
    answers to the requests made on it."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer
        self._updates: dict[tuple[str, str], asyncio.Queue[OHLCV | BaseException]] = {}
        self._answers: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 1
        self._pump: asyncio.Task[None] | None = None
        self.closed = False

    def send(self, message: dict[str, Any]) -> None:
        self._writer.write(protocol.frame(protocol.dumps(message)))

    async def receive(self) -> dict[str, Any]:
        return protocol.loads(await protocol.read_frame(self._reader))

    def start(self) -> None:
        """Start handing the incoming frames on; called once the handshake is over."""
        self._pump = asyncio.ensure_future(self._read())

    async def _read(self) -> None:
        failure: BaseException = ConnectionError("the provider hub connection closed")
        try:
            while True:
                message = await self.receive()
                if message.get("ev") == "bar":
                    self.updates(message["symbol"], message["timeframe"]).put_nowait(
                        message["bar"])
                elif message.get("ev") == "error":
                    self.updates(message["symbol"], message["timeframe"]).put_nowait(
                        protocol.decode_error(message["error"]))
                else:
                    waiting = self._answers.pop(message.get("id") or 0, None)
                    if waiting is not None and not waiting.done():
                        if message["ok"]:
                            waiting.set_result(message["value"])
                        else:
                            waiting.set_exception(protocol.decode_error(message["error"]))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - whatever ended the stream ends its waits
            failure = exc if isinstance(exc, OSError) else ConnectionError(str(exc))
        finally:
            self.closed = True
            for queue in self._updates.values():
                queue.put_nowait(failure)
            for waiting in self._answers.values():
                if not waiting.done():
                    waiting.set_exception(failure)
            self._answers.clear()

    def updates(self, symbol: str, timeframe: str) -> 'asyncio.Queue[OHLCV | BaseException]':
        key = (symbol, timeframe)
        queue = self._updates.get(key)
        if queue is None:
            queue = self._updates[key] = asyncio.Queue()
        return queue

    def watching(self, symbol: str, timeframe: str) -> bool:
        return (symbol, timeframe) in self._updates

    async def request(self, message: dict[str, Any]) -> Any:
        if self.closed:
            raise ConnectionError("the provider hub connection closed")
        request_id = self._next_id
        self._next_id += 1
        waiting: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._answers[request_id] = waiting
        self.send({**message, "id": request_id})
        try:
            return await waiting
        finally:
            self._answers.pop(request_id, None)

    async def close(self) -> None:
        self.closed = True
        if self._pump is not None:
            self._pump.cancel()
            await asyncio.wait({self._pump})
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except OSError:
            pass


class RemoteLiveProvider(RemoteProvider, LiveProviderPlugin[Any]):
    """A live provider whose real instance, connection and ``watch_ohlcv`` loops
    live in the hub and are shared with every other client of the same symbol."""

    @override
    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlcv_dir: Path | None = None, config: Any = None):
        super().__init__(symbol=symbol, timeframe=timeframe, ohlcv_dir=ohlcv_dir, config=config)
        self._stream: _Stream | None = None
        # The hub connection this instance saw last: connecting again while the hub
        # still holds that one asks the hub to reconnect
        self._generation: int | None = None

    @override
    async def connect(self) -> None:
        await self.disconnect()
        try:
            await self._connect_stream()
        except LookupError:
            # The hub was started again and holds no instance of the old one
            await asyncio.to_thread(self._open)
            self._generation = None
            await self._connect_stream()

    async def _connect_stream(self) -> None:
        address = hub_address()
        if address is None:
            raise ConnectionError(f"no provider hub is set ({HUB_ENV})")
        reader, writer = await asyncio.open_connection(*address)
        stream = _Stream(reader, writer)
        try:
            writer.write(protocol.frame(_token()))
            stream.send({"op": "connect", "handle": self._handle, "seen": self._generation})
            answer = await stream.receive()
            if not answer["ok"]:
                raise protocol.decode_error(answer["error"])
        except BaseException:
            await stream.close()
            raise
        self._generation = answer["value"]
        stream.start()
        self._stream = stream

    @override
    async def disconnect(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            await stream.close()

    @property
    @override
    def is_connected(self) -> bool:
        return self._stream is not None and not self._stream.closed

    def _connected_stream(self) -> _Stream:
        stream = self._stream
        if stream is None or stream.closed:
            raise ConnectionError("the provider hub connection closed")
        return stream

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        stream = self._connected_stream()
        if not stream.watching(symbol, timeframe):
            stream.updates(symbol, timeframe)
            await stream.request({"op": "watch", "symbol": symbol, "timeframe": timeframe})
        update = await stream.updates(symbol, timeframe).get()
        if isinstance(update, BaseException):
            raise update
        return update

    @override
    async def backfill_closed_bars(self, symbol: str, timeframe: str,
                                   since_ms: int) -> list[OHLCV]:
        return await self._connected_stream().request({
            "op": "backfill", "symbol": symbol, "timeframe": timeframe, "since_ms": since_ms,
        })


_remote_classes: dict[str, type[RemoteProvider] | None] = {}
_remote_classes_lock = threading.Lock()


def remote_plugin(name: str) -> type[RemoteProvider] | None:
    """The class that stands for a provider plugin in a process using the hub.

    :param name: Plugin name as declared in the entry point.
    :return: The remote class, or ``None`` for a plugin that is no data provider
        (the caller then loads the plugin itself).
    :raises PluginNotFoundError: If the hub has no plugin of that name.
    """
    with _remote_classes_lock:
        if name in _remote_classes:
            return _remote_classes[name]
    described = _class_request({"op": "describe", "plugin": name})
    cls: type[RemoteProvider] | None = None
    if described is not None:
        base = RemoteLiveProvider if described["live"] else RemoteProvider
        # Named as the plugin's class, so a log or a traceback names the plugin
        cls = type(described["class_name"], (base,),
                   {"hub_plugin": name, "__module__": __name__, **described["attributes"]})
    with _remote_classes_lock:
        return _remote_classes.setdefault(name, cls)
