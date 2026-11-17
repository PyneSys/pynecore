"""The errors PyneCore reports by code: :class:`PyneError`, the base of every
exception class :mod:`~pynecore.errorcodes` generates from ``scripts/errorcodes/registry.toml``.

An error carries its code, the values of the code's parameters, and, once the
run has located it, the bar it stopped on and the line of the script that
raised it. :meth:`PyneError.report` writes the one line a run prints for it,
in the form ``pyneaot run`` writes it too::

    bar 12 (2024-01-02 03:04 UTC): R0010 at /scripts/my_script.py:7:5: <message>
"""

import os
import sys
from datetime import datetime, UTC
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, NamedTuple, TypeVar

__all__ = ['Code', 'SiteLoc', 'SyntaxLocation', 'PyneError', 'PyneSyntaxError', 'RUNTIME_ROOTS',
           'MAIN_MODULE', 'MODULE_ID', 'format_arg', 'bar_time_text', 'user_traceback']

# Frames under these roots are the runtime's own or the interpreter's, never the
# script's: an error is located at the innermost frame outside them, and the
# security trace leaves them out of a read's ``stack``.
RUNTIME_ROOTS = tuple(sorted({
    str(Path(__file__).resolve().parent.parent) + os.sep,
    str(Path(sys.prefix).resolve()) + os.sep,
    str(Path(sys.base_prefix).resolve()) + os.sep,
}))

#: The module id of the script itself in an error's location; a library is its
#: import name.
MAIN_MODULE = 'main'
#: The module global the script's module carries its id in.
MODULE_ID = '__pyne_module_id__'

#: Where in a source a syntax rule was broken, as :class:`SyntaxError` takes it:
#: the file, the one-based line and column, and the text of that line.
SyntaxLocation = tuple[str, int, int, str | None]

#: Whole numbers below this magnitude are written without a fraction.
_WHOLE_LIMIT = 1e18


class Code(NamedTuple):
    """One code of the registry: what a reported diagnostic needs."""
    code: str
    severity: str
    title: str
    #: The message with ``{param}`` placeholders; ``{{`` and ``}}`` are literal braces.
    message: str
    #: The placeholders in order of first appearance: :meth:`render`'s argument order.
    params: tuple[str, ...]
    #: The quick fix's title and the text that replaces the diagnostic's range.
    fix_title: str | None = None
    fix: str | None = None

    def render(self, *args: str) -> str:
        """The message with ``args[i]`` in place of ``params[i]``. A message whose
        last parameter is empty (``Assertion failed. {message}`` of a bare
        ``assert``) ends without the space before it.

        :param args: One text per parameter.
        :return: The filled-in message.
        """
        return self.message.format(**dict(zip(self.params, args))).rstrip()


class SiteLoc(NamedTuple):
    """Where in the script's code an error was raised.

    The location is data, the host writes it: a local run names the Pyne code
    file it ran (:meth:`__str__`), the web editor the module's tab. The Pine
    line is for a host that ran the Pine source (:meth:`place`): one that ran
    the Pyne code names the Pyne code's line. A compiled run's location has the
    same fields.
    """
    #: The module: :data:`MAIN_MODULE` for the script, a library's import name.
    module: str
    #: The Pyne code file the module was run from.
    path: str
    #: One-based line in :attr:`path`.
    line: int
    #: One-based column in :attr:`path`, when the interpreter knows it.
    col: int | None = None
    #: The line of the Pine Script source :attr:`path` was compiled from.
    pine_line: int | None = None

    def place(self, pine: bool = False) -> str:
        """The location as a run prints it.

        :param pine: The run started from the Pine source: name its line when the
            sourcemap knows it.
        :return: ``Pine line N``, or the Pyne code's ``file:line:column``.
        """
        if pine and self.pine_line is not None:
            return f"Pine line {self.pine_line}"
        return str(self)

    def __str__(self) -> str:
        if self.col is None:
            return f"{self.path}:{self.line}"
        return f"{self.path}:{self.line}:{self.col}"


def format_arg(value: Any) -> str:
    """A parameter value as an error message shows it, the same way a compiled
    run does: ``na``, ``true`` / ``false``, a whole number without a fraction,
    any other number in its shortest exact form written out in full (``1e-07``
    is ``0.0000001``). A Pine ``int`` is a float while the script runs, so it
    shows as ``3``, never ``3.0``.

    The number cases are tested against the vectors of ``scripts/errorcodes/format_arg.toml``,
    which the Rust ``pynecore_codes::format_arg`` runs too.

    :param value: The value.
    :return: Its text.
    """
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return 'na'
        if -_WHOLE_LIMIT < value < _WHOLE_LIMIT and value == int(value):
            return str(int(value))
        if value in (float('inf'), float('-inf')):
            return 'inf' if value > 0 else '-inf'
        return format(Decimal(repr(value)), 'f')
    if isinstance(value, str):
        return value
    # An ``NA`` is told by its type slot, not by its class: this module is under
    # every other one, and importing the type package from here closes an import
    # cycle through the library (types -> lib -> errorcodes -> this module).
    if hasattr(type(value), '__pyne_na_type__'):
        return 'na'
    return str(value)


def bar_time_text(ms: int) -> str:
    """A bar's open time in epoch milliseconds as an error report names it:
    ``2024-01-02 03:04 UTC``, with seconds and milliseconds only when the time
    has them.

    :param ms: The time in epoch milliseconds.
    :return: Its text.
    """
    t = datetime.fromtimestamp(ms // 1000, UTC)
    milli = ms % 1000
    text = t.strftime('%Y-%m-%d %H:%M')
    if milli:
        text += f":{t.second:02}.{milli:03}"
    elif t.second:
        text += f":{t.second:02}"
    return text + ' UTC'


def user_traceback(tb: TracebackType | None) -> TracebackType | None:
    """The innermost entry of ``tb`` outside the runtime (:data:`RUNTIME_ROOTS`):
    the frame of the script's code that raised, also when a built-in raised it
    after the script called it.

    :param tb: A traceback, or None.
    :return: Its entry, or None when every frame is the runtime's.
    """
    found = None
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if not filename.startswith('<') \
                and not os.path.abspath(filename).startswith(RUNTIME_ROOTS):
            found = tb
        tb = tb.tb_next
    return found


#: A :class:`PyneError` subclass, for the rebuild helpers that keep it
_Error = TypeVar('_Error', bound='PyneError')


def _rebuild(cls: type[_Error], params: dict[str, Any], state: dict[str, Any]) -> _Error:
    """The error :meth:`PyneError.__reduce__` pickled. The base's constructor
    takes the parameters by their registry names, which a class's own does not
    when one is a Python keyword."""
    error = cls.__new__(cls)
    PyneError.__init__(error, **params)
    error.__dict__.update(state)
    return error


class PyneError(Exception):
    """An error with a registry code. Each code has a generated subclass in
    :mod:`~pynecore.errorcodes`, which also derives from the built-in exception the
    registry names (``IndexOutOfRangeError`` is an :class:`IndexError`), and
    takes the code's parameters as keyword arguments::

        raise IndexOutOfRangeError(function='array.get', index=index, size=size)
    """
    #: The registry row of the subclass's code.
    code: ClassVar[Code]

    def __init__(self, **params: Any) -> None:
        #: The parameter values as they were raised; the message formats them.
        self.params = params
        #: The bar the run stopped on, set where the run catches the error.
        self.bar_index: int | None = None
        #: That bar's open time in epoch milliseconds.
        self.bar_time: int | None = None
        #: The line of the script that raised it.
        self.at: SiteLoc | None = None
        #: Raised by the code of a ``request.security`` context, not the chart's.
        self.in_security = False
        super().__init__(self.code.render(*(format_arg(params[p]) for p in self.code.params)))

    @property
    def message(self) -> str:
        """The code's message with the parameters filled in."""
        return self.args[0]

    def __reduce__(self) -> tuple[Any, ...]:
        # The keyword-only constructor does not take the message the default
        # reduce hands back, and a security child sends its error through a pipe
        state = {k: v for k, v in self.__dict__.items() if k != 'params'}
        return _rebuild, (type(self), self.params, state)

    def __repr__(self) -> str:
        params = ', '.join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({params})"

    def report(self, pine: bool = False) -> str:
        """The line a run prints for the error: the bar, when it is known, the
        code, the location, the message, and a note when the code of a
        ``request.security`` context raised it.

        :param pine: The run started from the Pine source, so the location is its
            line when the sourcemap knows it (:meth:`SiteLoc.place`).
        :return: The report line.
        """
        text = self.code.code
        if self.at is not None:
            text += f" at {self.at.place(pine)}"
        text += f": {self.message}"
        if self.in_security:
            text += " (in a `request.security` context)"
        if self.bar_index is None:
            return text
        when = '' if self.bar_time is None else f" ({bar_time_text(self.bar_time)})"
        return f"bar {self.bar_index}{when}: {text}"


def _rebuild_syntax(cls: type['PyneSyntaxError'], params: dict[str, Any],
                    state: dict[str, Any], location: SyntaxLocation | None) -> 'PyneSyntaxError':
    """The error :meth:`PyneSyntaxError.__reduce__` pickled: :class:`SyntaxError`
    keeps its location in slots, which the instance dictionary does not hold."""
    error = _rebuild(cls, params, state)
    error.locate(location)
    return error


class PyneSyntaxError(PyneError, SyntaxError):
    """A coded rule of Pyne code or of the Pyne Edge profile, broken in the
    source before it runs. It is a :class:`SyntaxError`, so Python's traceback
    puts a caret under the column, and the generated class of every ``P`` and
    ``E`` code derives from it::

        raise SeriesOutsideFunctionError(location=(path, line, col, text))
    """

    def __init__(self, *, location: SyntaxLocation | None = None, **params: Any) -> None:
        super().__init__(**params)
        self.locate(location)

    def locate(self, location: SyntaxLocation | None) -> None:
        """Set where in the source the rule was broken.

        :param location: The file, line, column and line text, or None to leave it unset
        """
        if location is not None:
            self.filename, self.lineno, self.offset, self.text = location

    @property
    def location(self) -> SyntaxLocation | None:
        """The file, line, column and line text, when the error has a file."""
        if self.filename is None:
            return None
        return self.filename, self.lineno or 0, self.offset or 0, self.text

    def __reduce__(self) -> tuple[Any, ...]:
        state = {k: v for k, v in self.__dict__.items() if k != 'params'}
        return _rebuild_syntax, (type(self), self.params, state, self.location)

    def report(self, pine: bool = False) -> str:
        """The line a run prints for the error: the code, the place in the
        source, when it is known, and the message.

        :param pine: Unused: the rule is broken in the Pyne code, which is the
            place named even when the run started from the Pine source.
        :return: The report line.
        """
        text = self.code.code
        if self.filename is not None:
            text += f" at {self.filename}:{self.lineno}"
            if self.offset:
                text += f":{self.offset}"
        return f"{text}: {self.message}"
