"""
Cross-platform raw key reader for interactive CLI tools.

Provides a non-blocking ``read_key()`` returning a high-level ``Key`` enum
value, a single character (for regular typed text), or ``None`` on timeout.
The ``raw_terminal()`` context manager switches the controlling terminal
into cbreak mode on POSIX systems; on Windows it is a no-op since
``msvcrt`` reads characters directly without echo.

SIGWINCH handling is intentionally NOT part of this module — install
a separate handler if you need resize notifications.
"""
import sys
from enum import Enum, auto
from contextlib import contextmanager


class Key(Enum):
    """Special non-character keys."""
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    PAGE_UP = auto()
    PAGE_DOWN = auto()
    HOME = auto()
    END = auto()
    ENTER = auto()
    ESC = auto()
    BACKSPACE = auto()
    TAB = auto()
    SHIFT_TAB = auto()


KeyOrChar = Key | str

# Time budget for completing an ESC-prefixed escape sequence before we
# decide the user actually pressed ESC alone.
_ESC_SEQ_TIMEOUT = 0.05


if sys.platform == 'win32':
    import msvcrt  # type: ignore[import-not-found]
    import time

    _WIN_SPECIAL: dict[str, Key] = {
        'H': Key.UP, 'P': Key.DOWN, 'K': Key.LEFT, 'M': Key.RIGHT,
        'G': Key.HOME, 'O': Key.END, 'I': Key.PAGE_UP, 'Q': Key.PAGE_DOWN,
    }

    @contextmanager
    def raw_terminal():
        """No-op on Windows — msvcrt reads char-by-char without echo."""
        yield

    def read_key(timeout: float = 0.05) -> KeyOrChar | None:
        """Read one key event or return ``None`` after ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ('\xe0', '\x00'):
                    code = msvcrt.getwch()
                    return _WIN_SPECIAL.get(code)
                if ch == '\r':
                    return Key.ENTER
                if ch == '\x08':
                    return Key.BACKSPACE
                if ch == '\x1b':
                    return Key.ESC
                if ch == '\t':
                    return Key.TAB
                return ch
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.01, remaining))

else:
    import os
    import termios
    import tty
    import select

    _TILDE_SEQ: dict[str, Key] = {
        '1': Key.HOME, '7': Key.HOME,
        '4': Key.END, '8': Key.END,
        '5': Key.PAGE_UP, '6': Key.PAGE_DOWN,
    }

    @contextmanager
    def raw_terminal():
        """Switch stdin to cbreak mode; restore on exit.

        cbreak keeps signal handling (Ctrl+C → SIGINT) enabled, which we
        rely on as a hard-exit path.
        """
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _read_byte_within(budget: float) -> int | None:
        """Read one byte from stdin within ``budget`` seconds.

        Uses ``os.read`` directly on the file descriptor to bypass Python's
        text-mode buffer — otherwise ``select`` reports no data even when
        the rest of an escape sequence has already been buffered in user
        space, causing arrow keys to be misread as bare ESC.
        """
        fd = sys.stdin.fileno()
        try:
            r, _, _ = select.select([fd], [], [], budget)
        except InterruptedError:
            return None
        if not r:
            return None
        try:
            b = os.read(fd, 1)
        except (OSError, InterruptedError):
            return None
        if not b:
            return None
        return b[0]

    def read_key(timeout: float = 0.05) -> KeyOrChar | None:
        """Read one key event or return ``None`` after ``timeout`` seconds.

        ESC-prefixed sequences are parsed; a bare ESC (no follow-up within
        ``_ESC_SEQ_TIMEOUT``) is returned as ``Key.ESC``.
        """
        b = _read_byte_within(timeout)
        if b is None:
            return None
        if b == 0x1b:  # ESC
            b2 = _read_byte_within(_ESC_SEQ_TIMEOUT)
            if b2 is None or b2 != ord('['):
                return Key.ESC
            b3 = _read_byte_within(_ESC_SEQ_TIMEOUT)
            if b3 is None:
                return Key.ESC
            ch3 = chr(b3)
            if ch3 == 'A':
                return Key.UP
            if ch3 == 'B':
                return Key.DOWN
            if ch3 == 'C':
                return Key.RIGHT
            if ch3 == 'D':
                return Key.LEFT
            if ch3 == 'H':
                return Key.HOME
            if ch3 == 'F':
                return Key.END
            if ch3 == 'Z':
                return Key.SHIFT_TAB
            if ch3.isdigit():
                digits = ch3
                while True:
                    bn = _read_byte_within(_ESC_SEQ_TIMEOUT)
                    if bn is None:
                        return None
                    chn = chr(bn)
                    if chn == '~':
                        break
                    digits += chn
                return _TILDE_SEQ.get(digits)
            return None
        if b in (0x0d, 0x0a):  # \r, \n
            return Key.ENTER
        if b in (0x7f, 0x08):  # DEL, BS
            return Key.BACKSPACE
        if b == 0x09:  # \t
            return Key.TAB
        return chr(b)
