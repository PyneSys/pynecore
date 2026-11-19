"""
The wire format between a provider hub and its clients.

A connection is a loopback TCP stream of frames: a 4-byte big-endian length, then
that many bytes. The first frame a client sends is the hub's token, raw; every later
frame is a pickled ``dict``. Pickle carries the core's own types (bars, symbol info,
datetimes) unchanged, and :func:`loads` unpickles nothing else: a frame can name
only the classes of ``pynecore`` and a few standard value types, so no frame can
import a provider's SDK into a client or run a function.

A client's request is ``{'op': ..., ...}``. The hub answers with ``{'ok': True,
'value': ...}`` or ``{'ok': False, 'error': ...}`` (see :func:`encode_error`),
echoing the request's ``id`` when it has one; before the answer it may send events,
``{'ev': ..., ...}``.
"""
import asyncio
import builtins
import io
import pickle
import socket
import struct
from typing import Any

from .. import ProviderError, is_retryable_provider_error

#: The largest frame either side accepts, in bytes.
MAX_FRAME = 64 * 1024 * 1024

_LENGTH = struct.Struct(">I")

# Standard value types a frame may hold besides the classes of ``pynecore``
_SAFE_GLOBALS: dict[str, frozenset[str]] = {
    "builtins": frozenset({"set", "frozenset", "bytearray", "complex", "slice", "range"}),
    "collections": frozenset({"OrderedDict"}),
    "datetime": frozenset({"datetime", "date", "time", "timedelta", "timezone"}),
    "pathlib": frozenset({"Path", "PosixPath", "WindowsPath", "PurePath", "PurePosixPath",
                          "PureWindowsPath"}),
    "pathlib._local": frozenset({"Path", "PosixPath", "WindowsPath", "PurePath",
                                 "PurePosixPath", "PureWindowsPath"}),
    "zoneinfo": frozenset({"ZoneInfo"}),
    "decimal": frozenset({"Decimal"}),
}


class _Unpickler(pickle.Unpickler):
    """Unpickles the value types of a frame and refuses every other global."""

    def find_class(self, module: str, name: str) -> Any:
        if name in _SAFE_GLOBALS.get(module, ()):
            return super().find_class(module, name)
        if module == "builtins":
            found = getattr(builtins, name, None)
            if isinstance(found, type) and issubclass(found, BaseException):
                return found
        elif module == "pynecore" or module.startswith("pynecore."):
            found = super().find_class(module, name)
            # A class only: a function of the package could be called with any argument
            if isinstance(found, type):
                return found
        raise pickle.UnpicklingError(f"a hub frame may not hold {module}.{name}")


def dumps(message: Any) -> bytes:
    """Pickle one message.

    :param message: The message.
    :return: The frame's payload.
    """
    return pickle.dumps(message, protocol=5)


def loads(payload: bytes) -> Any:
    """Unpickle one message, refusing every type a frame may not hold.

    :param payload: A frame's payload.
    :return: The message.
    :raises pickle.UnpicklingError: If the payload names a type outside the allowed ones.
    """
    return _Unpickler(io.BytesIO(payload)).load()


def frame(payload: bytes) -> bytes:
    """A payload with its length in front, ready to send.

    :param payload: The payload.
    :return: The frame.
    :raises ValueError: If the payload is larger than :data:`MAX_FRAME`.
    """
    if len(payload) > MAX_FRAME:
        raise ValueError(f"a hub frame of {len(payload)} bytes is over the {MAX_FRAME} limit")
    return _LENGTH.pack(len(payload)) + payload


def _checked_length(header: bytes) -> int:
    length: int = _LENGTH.unpack(header)[0]
    if length > MAX_FRAME:
        raise ConnectionError(f"a hub frame of {length} bytes is over the {MAX_FRAME} limit")
    return length


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one frame's payload from a stream.

    :param reader: The stream.
    :return: The payload.
    :raises ConnectionError: If the stream ended, or the frame is over :data:`MAX_FRAME`.
    """
    try:
        header = await reader.readexactly(_LENGTH.size)
        return await reader.readexactly(_checked_length(header))
    except asyncio.IncompleteReadError as exc:
        raise ConnectionError("the provider hub connection closed") from exc


def recv_frame(sock: socket.socket) -> bytes:
    """Read one frame's payload from a blocking socket.

    :param sock: The socket.
    :return: The payload.
    :raises ConnectionError: If the socket closed, or the frame is over :data:`MAX_FRAME`.
    """
    return _recv_exactly(sock, _checked_length(_recv_exactly(sock, _LENGTH.size)))


def _recv_exactly(sock: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    while count:
        chunk = sock.recv(min(count, 1 << 20))
        if not chunk:
            raise ConnectionError("the provider hub connection closed")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


class RemoteProviderError(ProviderError):
    """A provider's failure in the hub, as a client sees it.

    Raised in place of an exception the client cannot hold as it is: one of a
    provider's SDK or of a plugin's own module. The message names the original type.

    :ivar remote_type: Dotted name of the exception type raised in the hub.
    """

    def __init__(self, message: str, remote_type: str, retryable: bool):
        super().__init__(message)
        self.remote_type = remote_type
        self.retryable = retryable

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.args[0], self.remote_type, self.retryable)


class RemoteNetworkError(RemoteProviderError, OSError):
    """A provider's failure in the hub with a socket-level fault behind it.

    An :class:`OSError`, as the fault in the hub was: the live runner retries a
    failed first connection on those.
    """


def _has_os_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError):
            return True
        current = current.__cause__ or current.__context__
    return False


def encode_error(exc: BaseException) -> dict[str, Any]:
    """Describe an exception raised in the hub for the client that caused it.

    An exception of ``pynecore`` or a builtin one travels as it is when it survives
    the frame's type rules; any other is described by its type name and message.
    Whether a retry can help is worked out here, where the whole cause chain is:
    the chain does not travel.

    :param exc: The exception.
    :return: The ``error`` of an answer.
    """
    kind = type(exc)
    described: dict[str, Any] = {
        "type": f"{kind.__module__}.{kind.__qualname__}",
        "message": str(exc),
        "retryable": is_retryable_provider_error(exc),
        "network": _has_os_error(exc),
        "pickled": None,
    }
    if kind.__module__ == "builtins" or kind.__module__.startswith("pynecore."):
        # noinspection PyBroadException
        try:
            pickled = dumps(exc)
            loads(pickled)
        except Exception:  # noqa: BLE001 - any exception may refuse to pickle
            pass
        else:
            described["pickled"] = pickled
    return described


def decode_error(described: dict[str, Any]) -> BaseException:
    """The exception a client raises for an ``error`` of an answer.

    :param described: The ``error``, from :func:`encode_error`.
    :return: The original exception when it travelled, else a
        :class:`RemoteProviderError`. Either classifies for a retry as the original did.
    """
    retryable = bool(described["retryable"])
    network = bool(described["network"])
    message = f"{described['message']} [{described['type']}]"
    cause_kind = RemoteNetworkError if network else RemoteProviderError
    if described["pickled"] is None:
        return cause_kind(message, described["type"], retryable)
    exc: BaseException = loads(described["pickled"])
    if (retryable and not is_retryable_provider_error(exc)) or (network and not _has_os_error(exc)):
        # The reason for a retry was further down the chain
        exc.__cause__ = cause_kind(message, described["type"], retryable)
    return exc
