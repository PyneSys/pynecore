"""
Interactive broker / exchange picker TUI for multi-broker data providers.

Multi-broker providers (CCXT serves ~100 crypto exchanges, cTrader serves every
broker the user holds an account with) need a backend chosen before a symbol can
be browsed. This module renders a single-pane, filterable, scrollable list on the
alternate screen buffer via ``rich.live`` and returns the chosen broker id, or
``None`` if the user quits without picking.

Navigation mirrors :class:`~pynecore.cli.utils.symbol_browser.SymbolBrowser`:
arrow keys / PgUp / PgDn / Home / End move the cursor, ``/`` starts a substring
filter, ENTER selects, ``q`` / ESC quits.
"""
import shutil
import sys
import threading

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .keyreader import Key, KeyOrChar, raw_terminal, read_key

# Footer block is one line of help text inside a Panel (2 border lines).
_FOOTER_HEIGHT = 3

# Panel top + bottom border lines deducted from the list panel height.
_LIST_CHROME_LINES = 2


class BrokerPicker:
    """Single-pane filterable list of broker / exchange ids.

    :ivar selected: The chosen broker once :meth:`run` returns, else ``None``.
    """

    def __init__(self, brokers: list[str], *, provider_name: str):
        """
        :param brokers: The broker / exchange ids to choose from.
        :param provider_name: Provider name shown in the panel title.
        """
        self.brokers: list[str] = list(brokers)
        self.provider_name = provider_name

        # View state.
        self.filtered: list[str] = list(self.brokers)
        self.cursor: int = 0
        self.scroll_offset: int = 0
        self.filter_text: str = ''
        self.filter_active: bool = False

        # Result.
        self.selected: str | None = None

        # Optional error shown above the help footer. The caller sets it before
        # re-running the picker when the chosen broker could not be opened (e.g.
        # an exchange that needs API credentials), so the user can pick another
        # without the command exiting.
        self.error: str | None = None

        # Resize state (Unix uses SIGWINCH; Windows polls size).
        self.resize_event: threading.Event = threading.Event()
        self.last_size: object = shutil.get_terminal_size()
        self._old_sigwinch = None

    # ---- filter + navigation -----------------------------------------

    def _apply_filter(self) -> None:
        if self.filter_text:
            needle = self.filter_text.lower()
            self.filtered = [b for b in self.brokers if needle in b.lower()]
        else:
            self.filtered = list(self.brokers)
        if self.cursor >= len(self.filtered):
            self.cursor = max(0, len(self.filtered) - 1)
        self.scroll_offset = 0

    def _move_cursor(self, delta: int) -> None:
        if not self.filtered:
            self.cursor = 0
            return
        self.cursor = max(0, min(len(self.filtered) - 1, self.cursor + delta))

    def _ensure_cursor_visible(self, list_height: int) -> None:
        if list_height <= 0:
            return
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        elif self.cursor >= self.scroll_offset + list_height:
            self.scroll_offset = self.cursor - list_height + 1

    # ---- key handling -------------------------------------------------

    def _handle_key(self, key: KeyOrChar) -> bool:
        """Return False to request exit, True to continue."""
        prev_cursor = self.cursor
        prev_filter = self.filter_text
        if self.filter_active:
            result = self._handle_filter_key(key)
        else:
            result = self._handle_normal_key(key)
        # A broker-open error belongs to the row it was raised on; once the user
        # moves the highlight or changes the filter it is stale and misleading,
        # so clear it as soon as the view selection changes.
        if self.cursor != prev_cursor or self.filter_text != prev_filter:
            self.error = None
        return result

    def _select_current(self) -> bool:
        """Commit the highlighted broker and request exit."""
        if self.filtered:
            self.selected = self.filtered[self.cursor]
        return False

    def _handle_filter_key(self, key: KeyOrChar) -> bool:
        if key is Key.ESC:
            self.filter_active = False
            if self.filter_text:
                self.filter_text = ''
                self._apply_filter()
            return True
        if key is Key.ENTER:
            self.filter_active = False
            return self._select_current()
        if key is Key.BACKSPACE:
            if self.filter_text:
                self.filter_text = self.filter_text[:-1]
                self._apply_filter()
            else:
                self.filter_active = False
            return True
        # Navigation keys exit the filter (keeping the typed text as the active
        # list filter) and move the cursor, mirroring the symbol browser.
        if key is Key.UP:
            self.filter_active = False
            self._move_cursor(-1)
            return True
        if key is Key.DOWN:
            self.filter_active = False
            self._move_cursor(1)
            return True
        if key is Key.PAGE_UP:
            self.filter_active = False
            self._move_cursor(-10)
            return True
        if key is Key.PAGE_DOWN:
            self.filter_active = False
            self._move_cursor(10)
            return True
        if isinstance(key, str) and key.isprintable():
            self.filter_text += key
            self._apply_filter()
            return True
        return True

    def _handle_normal_key(self, key: KeyOrChar) -> bool:
        if isinstance(key, str):
            if key == 'q':
                return False
            if key == '/':
                self.filter_active = True
            return True
        if key is Key.ESC:
            return False
        if key is Key.ENTER:
            return self._select_current()
        if key is Key.UP:
            self._move_cursor(-1)
        elif key is Key.DOWN:
            self._move_cursor(1)
        elif key is Key.PAGE_UP:
            self._move_cursor(-10)
        elif key is Key.PAGE_DOWN:
            self._move_cursor(10)
        elif key is Key.HOME:
            self.cursor = 0
        elif key is Key.END:
            self.cursor = max(0, len(self.filtered) - 1)
        return True

    # ---- rendering ----------------------------------------------------

    def _render_list(self, height: int) -> Panel:
        list_height = max(1, height - _LIST_CHROME_LINES)
        self._ensure_cursor_visible(list_height)
        end = self.scroll_offset + list_height
        visible = self.filtered[self.scroll_offset:end]
        lines: list[Text] = []
        for i, broker in enumerate(visible):
            idx = self.scroll_offset + i
            if idx == self.cursor:
                lines.append(Text(f"> {broker}", style="bold reverse"))
            else:
                lines.append(Text(f"  {broker}"))
        if not lines:
            lines.append(Text("  (no matches)", style="dim"))
        title_parts = [f"{self.provider_name} brokers "
                       f"({len(self.filtered)}/{len(self.brokers)})"]
        if self.filter_active or self.filter_text:
            cursor_marker = "_" if self.filter_active else ""
            title_parts.append(f"/{self.filter_text}{cursor_marker}")
        title = "  ".join(title_parts)
        return Panel(Group(*lines), title=title, title_align="left")

    def _footer_height(self) -> int:
        """Footer grows by one row when an error line is shown."""
        return _FOOTER_HEIGHT + (1 if self.error else 0)

    def _render_footer(self) -> Panel:
        help_text = Text(
            "Up Down: navigate  -  PgUp PgDn: jump 10  -  /: search  -  "
            "Enter: select  -  q: quit",
            style="dim",
        )
        if self.error:
            return Panel(Group(Text(self.error, style="red"), help_text),
                         height=self._footer_height())
        return Panel(help_text, height=_FOOTER_HEIGHT)

    def _build_layout(self, console: Console) -> Layout:
        footer_height = self._footer_height()
        height = max(footer_height + 3, console.size.height)
        list_height = height - footer_height
        layout = Layout()
        layout.split_column(
            Layout(self._render_list(list_height), name="list"),
            Layout(self._render_footer(), name="footer", size=footer_height),
        )
        return layout

    # ---- resize handling ---------------------------------------------

    def _install_sigwinch(self) -> None:
        if sys.platform == 'win32':
            return
        import signal
        self._old_sigwinch = signal.signal(
            signal.SIGWINCH, lambda *_: self.resize_event.set()
        )

    def _restore_sigwinch(self) -> None:
        if sys.platform == 'win32' or self._old_sigwinch is None:
            return
        import signal
        signal.signal(signal.SIGWINCH, self._old_sigwinch)
        self._old_sigwinch = None

    def _check_size_change(self) -> bool:
        """Windows fallback: detect resize by polling terminal size."""
        if sys.platform != 'win32':
            return False
        current = shutil.get_terminal_size()
        if current != self.last_size:
            self.last_size = current
            return True
        return False

    # ---- run loop -----------------------------------------------------

    def run(self) -> str | None:
        """Render the picker and block until the user selects or quits.

        :return: The chosen broker id, or ``None`` if the user quit.
        """
        if not self.brokers:
            print("No brokers available.", file=sys.stderr)
            return None

        # Reset so the picker can be re-run after a failed broker attempt.
        self.selected = None

        console = Console()
        self._install_sigwinch()
        try:
            with raw_terminal():
                with Live(
                    self._build_layout(console),
                    console=console,
                    screen=True,
                    auto_refresh=False,
                ) as live:
                    self._main_loop(live, console)
        except KeyboardInterrupt:
            return None
        finally:
            self._restore_sigwinch()
        return self.selected

    def _main_loop(self, live: Live, console: Console) -> None:
        while True:
            key = read_key(timeout=0.05)
            if key is not None:
                if not self._handle_key(key):
                    return
            if self.resize_event.is_set():
                self.resize_event.clear()
            self._check_size_change()
            live.update(self._build_layout(console))
            live.refresh()
