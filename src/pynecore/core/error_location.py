"""Where an error of a script's run happened: the bar, the line of the script,
and the line of the Pine Script source the script was compiled from.

The run calls :func:`located` on an exception leaving the script's code. A
:class:`~pynecore.core.errors.PyneError` gets the bar and its location; a few
of Python's own exceptions get their code first (a ``ZeroDivisionError`` is
``R0001``, a failed ``assert`` of the script's own ``R0004``), and so does the
error of a built-in's body the transform copied into the script (an inlined
``array.get``). Any other exception of the script's own code, a ``KeyError``
of its dict, stays a Python error without a code.
"""

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, Callable

from .. import errorcodes
from ..transformers.call_inline import INLINE_SITES
from .datetime import TimezoneNotFoundError
from .errors import MODULE_ID, RUNTIME_ROOTS, PyneError, SiteLoc, user_traceback

__all__ = ['located', 'coded', 'internal', 'INTERNAL_ERROR_REPORT', 'site_of', 'pine_line_for', 'INFINITY_TO_INT', 'NAN_TO_INT']

#: CPython's message of ``int()`` / ``round()`` of an infinite float.
INFINITY_TO_INT = "cannot convert float infinity to integer"
#: CPython's message of ``int()`` / ``round()`` of a NaN.
NAN_TO_INT = "cannot convert float NaN to integer"

#: What a run asks for after an internal error (``I0004``)
INTERNAL_ERROR_REPORT = ("This is an error of PyneCore, not of the script. Please report it at "
                         "https://github.com/pynesys/pynecore/issues with the script, the data and "
                         "the traceback.")

_UNBOUND_RE = re.compile(r"'([^']+)'")

#: The coded error of an inlined built-in's body, by its Pine name: the Python
#: exception its body raises, and the error made of its argument values.
_INLINE_ERRORS: dict[str, tuple[type[BaseException], Callable[..., PyneError]]] = {
    'array.get': (IndexError, lambda id_, index: errorcodes.IndexOutOfRangeError(
        function='array.get', index=index, size=len(id_))),
}

#: The directory of the ``pynecore`` package: the runtime's own code, without
#: the interpreter's standard library that :data:`RUNTIME_ROOTS` holds too
_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent) + os.sep

#: The broker and plugin APIs under the package: their exceptions are the
#: contract with a plugin, which the run reports as they are
_API_ROOTS = tuple(_PACKAGE_ROOT + os.path.join('core', d) + os.sep for d in ('broker', 'plugin'))

#: Exceptions the runtime raises for the machine, the environment or the
#: script's own code, never for a failure of its own: the disk, the network,
#: the time zone database, the stack, the memory, and a type error, which
#: inside the library is almost always a wrong argument of hand-written Pyne
#: code (``math.sqrt('x')``)
_NOT_INTERNAL: tuple[type[BaseException], ...] = (
    OSError, RecursionError, MemoryError, TimezoneNotFoundError, TypeError)

#: A holder of :data:`~pynecore.transformers.call_inline.INLINE_SITES` whose value is gone
_MISSING = object()


def _held(holder: Any, frame: FrameType) -> Any:
    """The value an inlined call's argument holder names in ``frame``."""
    if holder is None:
        return _MISSING
    kind = holder[0]
    if kind == 'c':
        return holder[1]
    if kind == 'n':
        name = holder[1]
        if name in frame.f_locals:
            return frame.f_locals[name]
        return frame.f_globals.get(name, _MISSING)
    state = frame.f_locals.get(holder[1])
    return _MISSING if state is None else state[holder[2]]


def _inlined(exc: BaseException) -> PyneError | None:
    """The coded error of an exception an inlined built-in's body raised.

    The pass anchors every node of a body it copies at the call's own point,
    so the failing instruction of such a body has a zero-width position; the
    module's :data:`~pynecore.transformers.call_inline.INLINE_SITES` names the
    call at that point and where each argument's value is. The values are read
    back from the failing frame, nothing is evaluated again.

    :param exc: The exception.
    :return: Its coded error, or None when an inlined body did not raise it, or
        its argument values are gone.
    """
    tb = exc.__traceback__
    if tb is None:
        return None
    while tb.tb_next is not None:
        tb = tb.tb_next
    frame = tb.tb_frame
    sites = frame.f_globals.get(INLINE_SITES)
    if not sites or frame.f_code.co_filename.startswith(RUNTIME_ROOTS):
        return None
    line, end_line, col, end_col = _position(tb)
    if line is None or line != end_line or col != end_col:
        return None
    for site_line, site_col, function, holders in sites:
        if (site_line, site_col) != (line, col):
            continue
        entry = _INLINE_ERRORS.get(function)
        if entry is None or not isinstance(exc, entry[0]):
            return None
        values = [_held(holder, frame) for holder in holders]
        if any(value is _MISSING for value in values):
            return None
        return entry[1](*values)
    return None


def coded(exc: BaseException) -> PyneError | None:
    """The coded error a Python exception of the script's code stands for.

    :param exc: The exception.
    :return: Its coded error, or None when the exception has no code.
    """
    if isinstance(exc, PyneError):
        return exc
    inlined = _inlined(exc)
    if inlined is not None:
        return inlined
    if isinstance(exc, ZeroDivisionError):
        return errorcodes.DivisionByZeroError()
    if isinstance(exc, UnboundLocalError):
        name = getattr(exc, 'name', None)
        if name is None:
            found = _UNBOUND_RE.search(str(exc))
            name = found.group(1) if found else '?'
        return errorcodes.UnboundVariableError(name=name)
    if isinstance(exc, OverflowError) and str(exc) == INFINITY_TO_INT:
        return errorcodes.InfinityToIntegerError()
    if isinstance(exc, ValueError) and str(exc) == NAN_TO_INT:
        return errorcodes.NaToIntegerError()
    if isinstance(exc, AssertionError) and _raised_by_script(exc):
        return errorcodes.AssertionFailedError(message=str(exc))
    return None


def internal(exc: BaseException) -> PyneError | None:
    """The ``I0004`` error an exception that stopped a run stands for: one
    without a code that the runtime's own code raised, which is an error of
    PyneCore, not of the script.

    An ``AttributeError`` of a ``__getattr__`` is the answer to a name the
    script misspelled (``lib.foo``), a ``TypeError`` is a wrong argument of
    the script, and the other exceptions of :data:`_NOT_INTERNAL` are the
    machine's; none of them is internal.

    :param exc: The exception.
    :return: The internal error, or None when ``exc`` has a code, or the
        script's code, the standard library or a broker or plugin API raised it.
    """
    if not isinstance(exc, Exception) or isinstance(exc, _NOT_INTERNAL) \
            or coded(exc) is not None:
        return None
    tb = exc.__traceback__
    if tb is None:
        return None
    while tb.tb_next is not None:
        tb = tb.tb_next
    code = tb.tb_frame.f_code
    path = os.path.abspath(code.co_filename)
    if not path.startswith(_PACKAGE_ROOT) or path.startswith(_API_ROOTS):
        return None
    if isinstance(exc, AttributeError) and code.co_name == '__getattr__':
        return None
    text = str(exc)
    return errorcodes.InternalError(what=type(exc).__name__ + (f": {text}" if text else ''))


def _raised_by_script(exc: BaseException) -> bool:
    """Whether the innermost frame of ``exc``'s traceback is the script's code:
    its own ``assert`` failed, not one of the runtime's."""
    tb = exc.__traceback__
    if tb is None:
        return False
    while tb.tb_next is not None:
        tb = tb.tb_next
    return user_traceback(tb) is tb


def site_of(tb: TracebackType | None) -> SiteLoc | None:
    """The line of the script's code a traceback leaves the script at, with
    its Pine line when the script was compiled from Pine.

    :param tb: The traceback.
    :return: The location, or None when every frame is the runtime's.
    """
    entry = user_traceback(tb)
    if entry is None:
        return None
    frame = entry.tb_frame
    line, _, col, _ = _position(entry)
    line = line or entry.tb_lineno
    module = frame.f_globals.get(MODULE_ID) or frame.f_globals.get('__name__', '?')
    path = frame.f_code.co_filename
    return SiteLoc(module, path, line, None if col is None else col + 1,
                   pine_line_for(path, line))


def _position(entry: TracebackType) -> tuple[int | None, int | None, int | None, int | None]:
    """The source position of the instruction a traceback entry stopped at:
    line, end line, and the 0-based columns."""
    return list(entry.tb_frame.f_code.co_positions())[entry.tb_lasti // 2]


def located(exc: BaseException, bar_index: int | None, bar_time: int | None) -> BaseException:
    """``exc`` as the run reports it: its coded error with the bar and the
    location filled in, or ``exc`` itself when it has no code.

    A coded error keeps a bar or location it already has: an error a
    ``request.security`` context raised carries the context's.

    :param exc: The exception that left the script's code.
    :param bar_index: The bar the run is on.
    :param bar_time: That bar's open time in epoch milliseconds.
    :return: The error to raise.
    """
    error = coded(exc)
    if error is None:
        return exc
    if error.bar_index is None:
        error.bar_index, error.bar_time = bar_index, bar_time
    if error.at is None:
        error.at = site_of(exc.__traceback__)
    # A Python exception's code raises from where the exception was raised
    return error.with_traceback(exc.__traceback__)


@lru_cache(maxsize=64)
def _mappings(py_path: str, _versions: tuple[int, int, int]) -> tuple[tuple[int, int], ...] | None:
    """The ``[python_line, pine_line]`` pairs of the sourcemap beside
    ``py_path``, or None when there is no usable map; ``_versions``, the
    modification times of the file, of its map and of its Pine source, only
    key the cache."""
    py = Path(py_path)
    try:
        data = json.loads(Path(py_path + '.map').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get('version') != 1 or data.get('converted_source'):
        return None
    mappings = data.get('mappings')
    if not isinstance(mappings, list):
        return None
    stamp = data.get('py_sha256')
    if stamp is not None:
        try:
            if hashlib.sha256(py.read_bytes()).hexdigest() != stamp:
                return None
        except OSError:
            return None
    pine_stamp = data.get('pine_sha256')
    if pine_stamp is not None:
        # Read as the compile reads it, so a source on disk with CRLF line
        # endings hashes the same
        try:
            pine = py.with_suffix('.pine').read_text(encoding='utf-8')
        except (OSError, ValueError):
            return None
        if hashlib.sha256(pine.encode('utf-8')).hexdigest() != pine_stamp:
            return None
    pairs = []
    for pair in mappings:
        if not (isinstance(pair, list) and len(pair) == 2
                and all(isinstance(v, int) and v > 0 for v in pair)):
            return None
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def pine_line_for(py_path: str, line: int) -> int | None:
    """The Pine line of ``line`` of a Pyne code file compiled from Pine.

    The sourcemap is the ``<file>.py.map`` beside the file, written by the
    compile: sparse ``[python_line, pine_line]`` pairs sorted by the Python
    line, a line belonging to the last pair at or before it. The map is used
    only when the ``.pine`` source is beside it too, it maps the source as it
    is on disk (not a v4/v5 source upgraded on the way), and its ``py_sha256``
    and ``pine_sha256``, when it has them, are the file's and the ``.pine``
    source's: a stale map would name a wrong line.

    :param py_path: The Pyne code file.
    :param line: One-based line in it.
    :return: The one-based Pine line, or None when there is none.
    """
    py = Path(py_path)
    if py.suffix != '.py' or not py.with_suffix('.pine').is_file():
        return None
    try:
        versions = (py.stat().st_mtime_ns, Path(py_path + '.map').stat().st_mtime_ns,
                    py.with_suffix('.pine').stat().st_mtime_ns)
    except OSError:
        return None
    pairs = _mappings(py_path, versions)
    if not pairs:
        return None
    result = None
    for py_line, pine in pairs:
        if py_line > line:
            break
        result = pine
    return result
