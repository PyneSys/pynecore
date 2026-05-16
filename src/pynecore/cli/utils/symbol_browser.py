"""
Interactive symbol browser TUI for the ``pyne data download`` command.

Renders a two-pane layout (scrollable symbol list on the left, live symbol
info on the right) on the alternate screen buffer via ``rich.live``. Symbol
info is fetched on demand on a single background worker, debounced so fast
scrolling does not flood the provider with requests, and cached in memory
with an LRU eviction policy.

Pressing ENTER on a symbol opens an inline timeframe + date wizard below the
panels. Submitting the wizard starts a download on a background worker; the
progress strip replaces the wizard until completion, after which the user
returns to the browse view and can pick another symbol.
"""
import shutil
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

from rich.console import Console, Group
from rich.highlighter import ReprHighlighter
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from rich.text import Text

from ...core.ohlcv_file import OHLCVWriter
from ...core.plugin import ProviderPlugin
from ...core.syminfo import SymInfo
from ..commands.data import parse_date_or_days, validate_timeframe
from .keyreader import Key, KeyOrChar, raw_terminal, read_key

_HIGHLIGHTER = ReprHighlighter()


# Time the cursor must stay on a symbol before we trigger a fetch — protects
# against bursting the provider during fast scrolling.
_FETCH_DEBOUNCE_S = 0.15

# Footer block is one line of help text inside a Panel (2 border lines).
_FOOTER_HEIGHT = 3

# Wizard / progress strip height when no dropdown is open. Panel chrome (2)
# + field row (1) + hint row (1) + a one-row breathing space below the field
# row to mirror the dropdown panel's vertical offset.
_STRIP_HEIGHT_BASE = 5

# Progress strip stays at the legacy fixed height (panel chrome + 1 bar row +
# breathing space).
_STRIP_HEIGHT_PROGRESS = 5

# Hardcoded chrome around the list (panel borders + title row).
_LIST_CHROME_LINES = 2

# Wizard field model + dropdown option lists.

CUSTOM_LABEL = "Custom..."

TF_OPTIONS = ["1", "5", "15", "30", "60", "240", "1D", "1W", "1M", CUSTOM_LABEL]
FROM_OPTIONS_RAW = ["continue", "1", "7", "30", "90", "180", "365", CUSTOM_LABEL]
FROM_DISPLAY = {
    "continue":   "continue",
    "1":          "1 day back",
    "7":          "7 days back",
    "30":         "30 days back",
    "90":         "90 days back",
    "180":        "180 days back",
    "365":        "365 days back",
    CUSTOM_LABEL: "Custom date...",
}
TO_OPTIONS_RAW = ["now", CUSTOM_LABEL]
TO_DISPLAY = {"now": "now", CUSTOM_LABEL: "Custom date..."}
TRUNCATE_OPTIONS = ["No", "Yes"]

# Per-kind label shown in front of each field cell.
_KIND_LABELS = {
    'tf': "Timeframe",
    'from': "From",
    'to': "To",
    'truncate': "Truncate",
    'submit': "",
}

# Dropdown viewport size — max number of options visible at once. Scroll
# kicks in around the cursor when an option list grows past this; keeps the
# wizard strip from eating the symbol panel on small terminals.
_DROPDOWN_MAX_ROWS = 10


@dataclass
class WizardField:
    """One focusable element in the inline download wizard.

    ``kind`` drives the UX: ``tf|from|to`` get a dropdown + an inline
    Custom-text mode, ``truncate`` is a toggle, ``submit`` is a button
    whose ``Enter`` triggers download dispatch.
    """
    kind: str
    value: str = ""
    options: list[str] = field(default_factory=list)
    active: bool = False
    text_mode: bool = False
    dd_cursor: int = 0
    text_buffer: str = ""
    pre_active_value: str = ""


def _option_display(kind: str, opt: str) -> str:
    """Return the human-facing label for an option (kind-specific map)."""
    if kind == 'from':
        return FROM_DISPLAY.get(opt, opt)
    if kind == 'to':
        return TO_DISPLAY.get(opt, opt)
    return opt


def _resolve_default_for_dropdown(value: str,
                                  options: list[str]) -> tuple[int, str, bool]:
    """Map a raw default onto (dd_cursor, field_value, is_custom).

    Case-insensitive lookup against every option except ``Custom...``; if a
    match is found, the cursor lands on it and the value snaps to the
    option's canonical casing. Otherwise the cursor parks on ``Custom...``
    and the raw value is preserved verbatim — submit-time validation will
    catch garbage.
    """
    for i, opt in enumerate(options):
        if opt == CUSTOM_LABEL:
            continue
        if value.strip().lower() == opt.strip().lower():
            return i, opt, False
    return options.index(CUSTOM_LABEL), value, True


class SymbolBrowser:
    """Two-pane interactive browser for provider symbols with live info,
    plus an inline timeframe + date wizard that drives ``download_ohlcv``."""

    def __init__(self, provider: ProviderPlugin, symbols: list[str],
                 *,
                 ohlcv_dir: Path,
                 default_timeframe: str = "1D",
                 default_from: str = "continue",
                 default_to: str = "now",
                 default_chunk_size: int | None = None,
                 max_cache: int = 200):
        self.provider = provider
        self.symbols: list[str] = list(symbols)
        self.ohlcv_dir = ohlcv_dir
        self.default_chunk_size = default_chunk_size
        self.max_cache = max_cache

        # View state.
        self.filtered: list[str] = list(self.symbols)
        self.cursor: int = 0
        self.scroll_offset: int = 0
        self.filter_text: str = ''
        self.filter_active: bool = False

        # Fetch state.
        self.info_cache: "OrderedDict[str, SymInfo | Exception]" = OrderedDict()
        self.pending_symbol: str | None = None
        self.pending_since: float = 0.0
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.active_future: Future | None = None
        self.active_symbol: str | None = None

        # Mode + wizard + download state.
        self.mode: str = 'browse'  # 'browse' | 'wizard' | 'downloading'
        self._default_timeframe = default_timeframe
        self._default_from = default_from
        self._default_to = default_to
        self.wiz_fields: list[WizardField] = self._build_wizard_fields()
        self.wiz_focus: int = 0
        self.wiz_error: str | None = None
        self.dl_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.dl_future: Future | None = None
        self.dl_lock = threading.Lock()
        self.dl_total_seconds: int = 1
        self.dl_elapsed_seconds: int = 0
        self.dl_label: str = ""
        self.dl_status: str | None = None  # last download result message
        self.dl_status_ok: bool = True
        self.dl_status_until: float = 0.0  # monotonic time when status should clear

        # Resize state (Unix uses SIGWINCH; Windows polls size).
        self.resize_event: threading.Event = threading.Event()
        self.last_size: tuple[int, int] = shutil.get_terminal_size()
        self._old_sigwinch = None

    # ---- cache --------------------------------------------------------

    def _cache_get(self, key: str) -> "SymInfo | Exception | None":
        if key in self.info_cache:
            self.info_cache.move_to_end(key)
            return self.info_cache[key]
        return None

    def _cache_put(self, key: str, value: "SymInfo | Exception") -> None:
        self.info_cache[key] = value
        self.info_cache.move_to_end(key)
        while len(self.info_cache) > self.max_cache:
            self.info_cache.popitem(last=False)

    # ---- fetch worker -------------------------------------------------

    def _fetch_info(self, symbol: str) -> SymInfo:
        # Runs on the worker thread. update_symbol_info() reads self.symbol,
        # so we mutate it here. A single worker enforces serialization.
        self.provider.symbol = symbol
        return self.provider.update_symbol_info()

    def _maybe_start_fetch(self, now: float) -> None:
        if self.mode == 'downloading':
            return
        if self.active_future is not None:
            return
        if not self.filtered:
            return
        current = self.filtered[self.cursor]
        if current in self.info_cache:
            self.pending_symbol = None
            return
        if self.pending_symbol != current:
            self.pending_symbol = current
            self.pending_since = now
            return
        if now - self.pending_since < _FETCH_DEBOUNCE_S:
            return
        self.active_symbol = current
        self.active_future = self.executor.submit(self._fetch_info, current)

    def _maybe_collect_result(self) -> None:
        if self.active_future is None or not self.active_future.done():
            return
        sym = self.active_symbol
        try:
            result: SymInfo | Exception = self.active_future.result()
        except BaseException as exc:
            result = exc if isinstance(exc, Exception) else RuntimeError(repr(exc))
        if sym is not None:
            self._cache_put(sym, result)
        self.active_future = None
        self.active_symbol = None

    # ---- filtering / navigation --------------------------------------

    def _apply_filter(self) -> None:
        if self.filter_text:
            needle = self.filter_text.lower()
            self.filtered = [s for s in self.symbols if needle in s.lower()]
        else:
            self.filtered = list(self.symbols)
        if self.cursor >= len(self.filtered):
            self.cursor = max(0, len(self.filtered) - 1)
        self.scroll_offset = 0
        self.pending_symbol = None

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
        if self.mode == 'downloading':
            return True  # ignore everything; Ctrl+C still escapes via signal
        if self.mode == 'wizard':
            return self._handle_wizard_key(key)
        if self.filter_active:
            return self._handle_filter_key(key)
        return self._handle_normal_key(key)

    def _handle_filter_key(self, key: KeyOrChar) -> bool:
        if key is Key.ESC:
            self.filter_active = False
            if self.filter_text:
                self.filter_text = ''
                self._apply_filter()
            return True
        if key is Key.ENTER:
            self.filter_active = False
            return True
        if key is Key.BACKSPACE:
            if self.filter_text:
                self.filter_text = self.filter_text[:-1]
                self._apply_filter()
            else:
                self.filter_active = False
            return True
        if key is Key.UP:
            self._move_cursor(-1)
            return True
        if key is Key.DOWN:
            self._move_cursor(1)
            return True
        if key is Key.PAGE_UP:
            self._move_cursor(-10)
            return True
        if key is Key.PAGE_DOWN:
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
            return True
        if key is Key.ESC:
            return False
        if key is Key.ENTER:
            self._enter_wizard()
            return True
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

    # ---- wizard field model ------------------------------------------

    def _build_wizard_fields(self) -> list[WizardField]:
        """Initialise the wizard with CLI defaults snapped onto dropdown
        options where possible, else parked on ``Custom...`` verbatim."""
        tf_idx, tf_val, _ = _resolve_default_for_dropdown(
            self._default_timeframe, TF_OPTIONS)
        from_idx, from_val, _ = _resolve_default_for_dropdown(
            self._default_from, FROM_OPTIONS_RAW)
        to_idx, to_val, _ = _resolve_default_for_dropdown(
            self._default_to, TO_OPTIONS_RAW)
        return [
            WizardField(kind='tf', value=tf_val, options=list(TF_OPTIONS),
                        dd_cursor=tf_idx),
            WizardField(kind='from', value=from_val,
                        options=list(FROM_OPTIONS_RAW), dd_cursor=from_idx),
            WizardField(kind='to', value=to_val, options=list(TO_OPTIONS_RAW),
                        dd_cursor=to_idx),
            WizardField(kind='submit'),
        ]

    def _field(self, kind: str) -> WizardField | None:
        for f in self.wiz_fields:
            if f.kind == kind:
                return f
        return None

    def _target_file_exists(self) -> bool:
        """Does the target OHLCV file already exist for the current
        symbol + chosen TF? Drives both the Truncate toggle visibility
        and the smart From default ('continue' if it exists, '365' otherwise)."""
        if not self.filtered:
            return False
        tf_field = self._field('tf')
        if tf_field is None:
            return False
        tf_raw = tf_field.value.strip()
        if not tf_raw:
            return False
        try:
            tf = validate_timeframe(tf_raw)
        except ValueError:
            return False
        symbol = self.filtered[self.cursor]
        try:
            path = type(self.provider).get_ohlcv_path(
                symbol, tf, self.ohlcv_dir)
        except Exception:
            return False
        return path.exists()

    def _refresh_truncate_visibility(self) -> None:
        """Insert / remove the Truncate field based on file existence.

        Idempotent — safe to call after every TF change or wizard entry.
        Pre-existing toggle state is preserved if the field stays visible;
        a removed-and-re-added field defaults to ``No``.
        """
        exists = self._target_file_exists()
        existing = self._field('truncate')
        if exists and existing is None:
            # Insert just before the submit button.
            submit_idx = next(i for i, f in enumerate(self.wiz_fields)
                              if f.kind == 'submit')
            self.wiz_fields.insert(submit_idx, WizardField(
                kind='truncate', value="No", options=list(TRUNCATE_OPTIONS),
                dd_cursor=0,
            ))
            # Focus stays put — its index didn't move (we inserted before
            # submit, which is the last item).
        elif not exists and existing is not None:
            removed_idx = self.wiz_fields.index(existing)
            self.wiz_fields.remove(existing)
            if self.wiz_focus >= removed_idx:
                self.wiz_focus = max(0, self.wiz_focus - 1)

    def _apply_smart_from_default(self) -> None:
        """Switch the From field between 'continue' and '365' based on
        whether the target OHLCV file already exists. Only fires when the
        CLI default was 'continue' (no explicit --from override)."""
        if self._default_from != "continue":
            return
        from_field = self._field('from')
        if from_field is None:
            return
        new_value = "continue" if self._target_file_exists() else "365"
        from_field.value = new_value
        idx, _, _ = _resolve_default_for_dropdown(new_value, from_field.options)
        from_field.dd_cursor = idx

    # ---- enter / leave wizard ----------------------------------------

    def _enter_wizard(self) -> None:
        if not self.filtered:
            return
        self.mode = 'wizard'
        self.wiz_focus = 0
        self.wiz_error = None
        for fld in self.wiz_fields:
            fld.active = False
            fld.text_mode = False
            fld.text_buffer = ""
            fld.pre_active_value = ""
        # Pick a sensible From default based on whether the target file
        # already exists — only when the user didn't override --from on
        # the CLI (i.e. the original default was 'continue').
        self._apply_smart_from_default()
        self._refresh_truncate_visibility()
        # Status from a previous download stays visible until the next
        # mode transition — clear it here so the wizard hint row reads cleanly.
        self.dl_status = None

    # ---- wizard key dispatch -----------------------------------------

    def _handle_wizard_key(self, key: KeyOrChar) -> bool:
        if not self.wiz_fields:
            return True
        field_ = self.wiz_fields[self.wiz_focus]
        if field_.text_mode:
            return self._handle_text_input_key(key, field_)
        if field_.active:
            return self._handle_dropdown_key(key, field_)
        return self._handle_wizard_inactive_key(key)

    def _handle_wizard_inactive_key(self, key: KeyOrChar) -> bool:
        if key is Key.ESC:
            self.mode = 'browse'
            self.wiz_error = None
            return True
        if key is Key.LEFT or key is Key.SHIFT_TAB:
            self._focus_next(-1)
            return True
        if key is Key.RIGHT or key is Key.TAB:
            self._focus_next(+1)
            return True
        if key is Key.ENTER:
            self._activate_focused()
            return True
        # Up/Down inactive: no-op (dropdown is the only place those move).
        return True

    def _handle_dropdown_key(self, key: KeyOrChar,
                             field_: WizardField) -> bool:
        if key is Key.ESC:
            field_.active = False
            field_.value = field_.pre_active_value
            return True
        if key is Key.UP:
            field_.dd_cursor = (field_.dd_cursor - 1) % len(field_.options)
            return True
        if key is Key.DOWN:
            field_.dd_cursor = (field_.dd_cursor + 1) % len(field_.options)
            return True
        if key is Key.ENTER:
            chosen = field_.options[field_.dd_cursor]
            if chosen == CUSTOM_LABEL:
                # Drop into inline text-input. Seed buffer with the prior
                # custom value if we already had one (i.e. the field was
                # parked on Custom before); otherwise start empty.
                was_custom = field_.pre_active_value not in field_.options
                field_.text_buffer = field_.pre_active_value if was_custom else ""
                field_.text_mode = True
                field_.active = False
            else:
                field_.value = chosen
                field_.active = False
                if field_.kind == 'tf':
                    # TF change may toggle Truncate visibility.
                    self._refresh_truncate_visibility()
            self.wiz_error = None
            return True
        return True

    def _handle_text_input_key(self, key: KeyOrChar,
                               field_: WizardField) -> bool:
        if key is Key.ESC:
            field_.text_mode = False
            field_.value = field_.pre_active_value
            field_.text_buffer = ""
            return True
        if key is Key.ENTER:
            field_.value = field_.text_buffer
            field_.text_mode = False
            field_.text_buffer = ""
            if field_.kind == 'tf':
                self._refresh_truncate_visibility()
            self.wiz_error = None
            return True
        if key is Key.BACKSPACE:
            field_.text_buffer = field_.text_buffer[:-1]
            return True
        if isinstance(key, str) and key.isprintable():
            field_.text_buffer += key
            return True
        return True

    def _focus_next(self, delta: int) -> None:
        if not self.wiz_fields:
            return
        n = len(self.wiz_fields)
        self.wiz_focus = (self.wiz_focus + delta) % n

    def _activate_focused(self) -> None:
        field_ = self.wiz_fields[self.wiz_focus]
        if field_.kind in ('tf', 'from', 'to'):
            field_.pre_active_value = field_.value
            field_.active = True
            # Position the dropdown cursor on whichever option matches the
            # current value (case-insensitive); otherwise park on Custom...
            idx, _, _ = _resolve_default_for_dropdown(
                field_.value, field_.options)
            field_.dd_cursor = idx
            return
        if field_.kind == 'truncate':
            field_.value = "Yes" if field_.value == "No" else "No"
            return
        if field_.kind == 'submit':
            self._submit_wizard()

    # ---- wizard validation + dispatch --------------------------------

    def _submit_wizard(self) -> None:
        tf_field = self._field('tf')
        from_field = self._field('from')
        to_field = self._field('to')
        trunc_field = self._field('truncate')
        assert tf_field is not None and from_field is not None and to_field is not None

        try:
            tf = validate_timeframe(tf_field.value)
        except ValueError as e:
            self.wiz_error = f"Timeframe: {e}"
            self.wiz_focus = self.wiz_fields.index(tf_field)
            return
        try:
            from_value = parse_date_or_days(from_field.value)
        except ValueError as e:
            self.wiz_error = f"From: {e}"
            self.wiz_focus = self.wiz_fields.index(from_field)
            return
        try:
            to_value = parse_date_or_days(to_field.value)
        except ValueError as e:
            self.wiz_error = f"To: {e}"
            self.wiz_focus = self.wiz_fields.index(to_field)
            return
        if to_value == "continue":
            self.wiz_error = "To: 'continue' is not valid here, use a date or 'now'"
            self.wiz_focus = self.wiz_fields.index(to_field)
            return

        truncate = (trunc_field is not None and trunc_field.value == "Yes")
        self.wiz_error = None
        self.mode = 'downloading'
        symbol = self.filtered[self.cursor]
        self.dl_label = f"{symbol} {tf}"
        self.dl_elapsed_seconds = 0
        self.dl_total_seconds = 1
        self.dl_status = None
        self.dl_future = self.dl_executor.submit(
            self._run_download, symbol, tf, from_value, to_value, truncate,
        )

    # ---- download worker (runs on dl_executor thread) ----------------

    def _run_download(self, symbol: str, tf: str,
                      from_value: datetime | str, to_value: datetime,
                      truncate: bool) -> None:
        try:
            # Mutate the provider in the same way the regular CLI flow
            # would have done at construction: symbol + timeframe +
            # xchg_timeframe + ohlcv_path + ohlcv_file all need to be in
            # sync before entering the provider context manager (which
            # opens the OHLCVWriter).
            provider = self.provider
            provider.symbol = symbol
            provider.timeframe = tf
            provider.xchg_timeframe = provider.to_exchange_timeframe(tf)
            provider.ohlcv_path = provider.get_ohlcv_path(
                symbol, tf, self.ohlcv_dir,
            )
            provider.ohlcv_file = OHLCVWriter(provider.ohlcv_path)

            with provider as ohlcv_writer:
                if truncate:
                    ohlcv_writer.seek(0)
                    ohlcv_writer.truncate()
                resolved_from = self._resolve_from(from_value, ohlcv_writer)
                # Strip tz to match the existing CLI download semantics.
                resolved_from = resolved_from.replace(tzinfo=None)
                resolved_to = to_value.replace(tzinfo=None)
                now_naive = datetime.now(UTC).replace(tzinfo=None)
                if resolved_to > now_naive:
                    resolved_to = now_naive

                fetch_all = (
                    from_value == "continue"
                    and ohlcv_writer.end_timestamp == 0
                    and getattr(type(provider), 'fetch_all_by_default', False)
                )
                if not fetch_all and resolved_to < resolved_from:
                    raise ValueError(
                        "End date (to) must be greater than start date (from)"
                    )

                total = max(1, int((resolved_to - resolved_from).total_seconds()))
                with self.dl_lock:
                    self.dl_total_seconds = total
                    self.dl_elapsed_seconds = 0

                def on_progress(ts: datetime) -> None:
                    elapsed = int((ts - resolved_from).total_seconds())
                    if elapsed < 0:
                        elapsed = 0
                    if elapsed > total:
                        elapsed = total
                    with self.dl_lock:
                        self.dl_elapsed_seconds = elapsed

                provider.download_ohlcv(
                    resolved_from, resolved_to,
                    on_progress=None if fetch_all else on_progress,
                    limit=self.default_chunk_size,
                )

                # Write a fresh SymInfo TOML alongside the OHLCV — matches
                # what the CLI download path does on first run.
                if not provider.is_symbol_info_exists():
                    try:
                        sym_info = provider.update_symbol_info()
                        assert provider.ohlcv_path is not None
                        sym_info.save_toml(
                            provider.ohlcv_path.with_suffix('.toml')
                        )
                    except Exception:
                        pass  # SymInfo write is best-effort

            with self.dl_lock:
                self.dl_status = f"[OK] downloaded {self.dl_label}"
                self.dl_status_ok = True
        except BaseException as exc:
            msg = str(exc) or repr(exc)
            with self.dl_lock:
                self.dl_status = f"[ERR] {type(exc).__name__}: {msg}"
                self.dl_status_ok = False

    @staticmethod
    def _resolve_from(from_value: datetime | str,
                      ohlcv_writer: OHLCVWriter) -> datetime:
        """Mirror the CLI's ``time_from`` resolution for ``"continue"``."""
        if isinstance(from_value, datetime):
            return from_value
        if from_value != "continue":
            raise ValueError(f"Unexpected from value: {from_value!r}")
        end_ts = ohlcv_writer.end_timestamp
        interval = ohlcv_writer.interval
        if end_ts and interval:
            from datetime import timedelta
            return (datetime.fromtimestamp(end_ts, UTC)
                    + timedelta(seconds=interval))
        from datetime import timedelta
        return datetime.now(UTC) - timedelta(days=365)

    def _maybe_collect_download(self) -> None:
        if self.dl_future is None or not self.dl_future.done():
            return
        # _run_download stores its outcome on dl_status before returning;
        # the future itself never raises (all exceptions are captured).
        self.dl_future = None
        self.mode = 'browse'
        # OK lines fade after 4 s; error lines linger longer (8 s) so they
        # cannot be missed if the user was looking away when the download
        # ran. Either way an explicit ENTER / mode-change clears it sooner
        # via ``_enter_wizard``.
        linger_s = 4.0 if self.dl_status_ok else 8.0
        self.dl_status_until = time.monotonic() + linger_s
        # Symbol info on disk may have changed during the download — clear
        # the in-memory cache entry for the downloaded symbol so a fresh
        # SymInfo is fetched on next selection.
        if self.dl_label:
            sym = self.dl_label.split(' ', 1)[0]
            self.info_cache.pop(sym, None)

    def _maybe_clear_status(self, now: float) -> None:
        if self.dl_status is None:
            return
        if self.dl_status_until and now >= self.dl_status_until:
            self.dl_status = None
            self.dl_status_until = 0.0

    # ---- rendering ----------------------------------------------------

    def _render_list(self, height: int) -> Panel:
        list_height = max(1, height - _LIST_CHROME_LINES)
        self._ensure_cursor_visible(list_height)
        end = self.scroll_offset + list_height
        visible = self.filtered[self.scroll_offset:end]
        lines: list[Text] = []
        for i, sym in enumerate(visible):
            idx = self.scroll_offset + i
            if idx == self.cursor:
                lines.append(Text(f"> {sym}", style="bold reverse"))
            else:
                lines.append(Text(f"  {sym}"))
        if not lines:
            lines.append(Text("  (no matches)", style="dim"))
        title_parts = [f"Symbols ({len(self.filtered)}/{len(self.symbols)})"]
        if self.filter_active or self.filter_text:
            cursor_marker = "_" if self.filter_active else ""
            title_parts.append(f"/{self.filter_text}{cursor_marker}")
        title = "  ".join(title_parts)
        return Panel(Group(*lines), title=title, title_align="left")

    def _render_info(self) -> Panel:
        if not self.filtered:
            return Panel(Text("No symbol selected", style="dim"),
                         title="Info", title_align="left")
        sym = self.filtered[self.cursor]
        cached = self._cache_get(sym)
        if cached is None:
            body: Text | Table
            if self.active_symbol == sym:
                body = Text("Loading...", style="dim")
            else:
                body = Text("(press a moment to load)", style="dim")
            return Panel(body, title=sym, title_align="left")
        if isinstance(cached, Exception):
            msg = str(cached) or repr(cached)
            return Panel(Text(f"Error ({type(cached).__name__}): {msg}", style="red"),
                         title=sym, title_align="left")
        return Panel(self._info_table(cached),
                     title=f"{cached.prefix}:{cached.ticker}",
                     title_align="left")

    _DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    @staticmethod
    def _info_table(info: SymInfo) -> Table:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column()

        def row(label: str, value: object) -> None:
            table.add_row(label, _HIGHLIGHTER(str(value)))

        row("Description:", info.description)
        row("Type:", info.type)
        row("Currency:", info.currency)
        if info.basecurrency:
            row("Base currency:", info.basecurrency)
        row("Mintick:", info.mintick)
        row("Pricescale:", info.pricescale)
        row("Minmove:", info.minmove)
        row("Pointvalue:", info.pointvalue)
        row("Volume type:", info.volumetype)
        row("Timezone:", info.timezone)
        if info.avg_spread is not None:
            row("Avg spread:", info.avg_spread)
        if info.taker_fee is not None:
            row("Taker fee:", info.taker_fee)
        if info.maker_fee is not None:
            row("Maker fee:", info.maker_fee)
        if info.opening_hours:
            row("Opening hours:", SymbolBrowser._format_opening_hours(info.opening_hours))
        return table

    @staticmethod
    def _format_opening_hours(intervals) -> str:
        by_day: dict[int, list] = {}
        for iv in intervals:
            by_day.setdefault(iv.day, []).append(iv)
        lines = []
        for day in sorted(by_day.keys()):
            label = SymbolBrowser._DAY_NAMES.get(day, f"D{day}")
            day_ivs = sorted(by_day[day], key=lambda iv: iv.start)
            parts = [f"{iv.start.strftime('%H:%M')}-{iv.end.strftime('%H:%M')}"
                     for iv in day_ivs]
            lines.append(f"{label}  {', '.join(parts)}")
        return "\n".join(lines)

    def _wizard_strip_height(self) -> int:
        """Dynamic wizard strip height — grows when a dropdown is open."""
        active = next((f for f in self.wiz_fields if f.active), None)
        if active is None:
            return _STRIP_HEIGHT_BASE
        # Dropdown panel: top/bottom border (2) + min(len, MAX) option rows.
        rows = min(len(active.options), _DROPDOWN_MAX_ROWS)
        return _STRIP_HEIGHT_BASE + rows + 2

    def _render_field_cell(self, field_: WizardField, focused: bool) -> Text:
        label = _KIND_LABELS.get(field_.kind, "")
        line = Text()
        if field_.kind == 'submit':
            text = "[ Download ]"
            line.append(text, style="bold reverse" if focused else "bold")
            return line
        if label:
            line.append(f"{label}: ", style="bold")
        if field_.text_mode:
            # Inline text input: show the buffer with a block cursor.
            body = f"[ {field_.text_buffer}█ ]"
            line.append(body, style="reverse" if focused else "")
            return line
        # Selector view.
        if field_.kind == 'truncate':
            body = f"[ {field_.value} ]"
        else:
            body = f"[ {field_.value} ▾ ]"
        line.append(body, style="reverse" if focused else "")
        return line

    def _render_dropdown_panel(self, field_: WizardField) -> Panel:
        n = len(field_.options)
        view = min(n, _DROPDOWN_MAX_ROWS)
        # Center a window of size ``view`` on the cursor where possible.
        half = view // 2
        start = max(0, min(field_.dd_cursor - half, n - view))
        end = start + view
        lines: list[Text] = []
        max_label = max(
            (len(_option_display(field_.kind, opt)) for opt in field_.options),
            default=0,
        )
        for i in range(start, end):
            opt = field_.options[i]
            display = _option_display(field_.kind, opt)
            pad = " " * (max_label - len(display))
            line = Text()
            if i == field_.dd_cursor:
                line.append(f"> {display}{pad} <", style="reverse")
            else:
                # Bold the option that matches the field's currently
                # committed value (independent of where the cursor sits).
                style = "bold" if opt == field_.value else ""
                line.append(f"  {display}{pad}  ", style=style)
            lines.append(line)
        return Panel(Group(*lines), border_style="dim", padding=(0, 1))

    def _render_wizard_strip(self) -> Panel:
        sym = self.filtered[self.cursor] if self.filtered else "?"
        title = f"Download {sym}"
        fields_table = Table.grid(padding=(0, 2))
        for _ in self.wiz_fields:
            fields_table.add_column()
        cells = [self._render_field_cell(f, i == self.wiz_focus)
                 for i, f in enumerate(self.wiz_fields)]
        fields_table.add_row(*cells)

        active = next((f for f in self.wiz_fields if f.active), None)
        focused = self.wiz_fields[self.wiz_focus]
        if self.wiz_error:
            hint: Text = Text(f"[!] {self.wiz_error}", style="red")
        elif focused.text_mode:
            if focused.kind in ('from', 'to'):
                hint = Text(
                    "Type date: YYYY-MM-DD  or  YYYY-MM-DD HH:MM:SS  "
                    "-  Enter: confirm  -  Esc: cancel",
                    style="yellow",
                )
            else:
                hint = Text(
                    "Type custom timeframe (e.g. 3, 90, 1D)  "
                    "-  Enter: confirm  -  Esc: cancel",
                    style="dim",
                )
        elif active is not None:
            hint = Text(
                "Up/Down: select  -  Enter: confirm  -  Esc: cancel",
                style="dim",
            )
        elif focused.kind == 'truncate':
            hint = Text(
                "Enter: toggle  -  Yes will erase existing OHLCV file before download",
                style="yellow",
            )
        else:
            hint = Text(
                "Tab / Left Right: switch field  -  Enter: open / confirm  -  Esc: back",
                style="dim",
            )

        if active is not None:
            body: Group = Group(fields_table, self._render_dropdown_panel(active),
                                hint)
        else:
            body = Group(fields_table, hint)
        return Panel(body, title=title, title_align="left",
                     height=self._wizard_strip_height())

    def _render_progress_strip(self) -> Panel:
        with self.dl_lock:
            elapsed = self.dl_elapsed_seconds
            total = self.dl_total_seconds
        progress = Progress(
            TextColumn("{task.fields[label]}", style="bold"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
            expand=True,
        )
        progress.add_task("download", total=total, completed=elapsed,
                          label=self.dl_label)
        return Panel(progress, title=f"Downloading {self.dl_label}",
                     title_align="left", height=_STRIP_HEIGHT_PROGRESS)

    def _render_footer(self) -> Panel:
        if self.mode == 'wizard':
            help_text: Text = Text(
                "Edit fields  -  Tab / Up Down switch  -  Enter download  -  Esc back",
                style="dim",
            )
        elif self.mode == 'downloading':
            help_text = Text("Downloading... please wait", style="dim")
        else:
            parts = "Up Down: navigate  -  PgUp PgDn: jump 10  -  /: search  -  Enter: download  -  q: quit"
            help_text = Text(parts, style="dim")
            if self.dl_status is not None:
                style = "green" if self.dl_status_ok else "red"
                status_line = Text(self.dl_status, style=style)
                return Panel(Group(status_line, help_text), height=_FOOTER_HEIGHT)
        return Panel(help_text, height=_FOOTER_HEIGHT)

    def _build_layout(self, console: Console) -> Layout:
        height = max(_FOOTER_HEIGHT + 3, console.size.height)
        if self.mode == 'wizard':
            strip_height = self._wizard_strip_height()
        elif self.mode == 'downloading':
            strip_height = _STRIP_HEIGHT_PROGRESS
        else:
            strip_height = 0
        main_height = height - _FOOTER_HEIGHT - strip_height
        list_height = main_height
        layout = Layout()
        sections: list[Layout] = [Layout(name="main")]
        if self.mode == 'wizard':
            sections.append(Layout(self._render_wizard_strip(), name="strip",
                                   size=strip_height))
        elif self.mode == 'downloading':
            sections.append(Layout(self._render_progress_strip(), name="strip",
                                   size=strip_height))
        sections.append(Layout(self._render_footer(), name="footer",
                               size=_FOOTER_HEIGHT))
        layout.split_column(*sections)
        layout["main"].split_row(
            Layout(self._render_list(list_height), name="list", ratio=1),
            Layout(self._render_info(), name="info", ratio=2),
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

    def run(self) -> None:
        if not self.symbols:
            print("No symbols available.", file=sys.stderr)
            return

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
            pass
        finally:
            self._restore_sigwinch()
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.dl_executor.shutdown(wait=False, cancel_futures=True)

    def _main_loop(self, live: Live, console: Console) -> None:
        while True:
            key = read_key(timeout=0.05)
            if key is not None:
                if not self._handle_key(key):
                    return
            now = time.monotonic()
            self._maybe_collect_result()
            self._maybe_collect_download()
            self._maybe_clear_status(now)
            self._maybe_start_fetch(now)
            if self.resize_event.is_set():
                self.resize_event.clear()
            self._check_size_change()
            live.update(self._build_layout(console))
            live.refresh()
