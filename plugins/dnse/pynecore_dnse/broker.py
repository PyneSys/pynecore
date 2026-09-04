"""DNSE broker plugin — native conditional order execution for PyneCore (v2).

Builds on :class:`DNSEProvider` (history + metadata) and implements the
``BrokerPlugin`` abstracts using DNSE's **native conditional orders**
(``orderCategory=STOP|OCO`` on the account-scoped ``/accounts/{accountNo}/orders``
endpoints). Server-side stops fire even if the plugin is offline. v2 is
**REST-only** — no WebSocket transport; bars and fills come from REST polling.

Design:

* ``position_port = None`` — DNSE derivatives are netted per symbol.
* Pine intent -> native order:
  ``entry(stop)`` / ``entry(limit,stop)`` -> STOP; ``entry(limit)`` -> NORMAL LO;
  ``exit(limit,stop)`` -> OCO; ``exit(stop)`` -> STOP; ``exit(limit)`` -> NORMAL LO;
  ``close`` / market -> marketable LO (band edge).
* ``conditionOperator`` = ``>=`` for buy stops (trigger on the way up), ``<=`` for
  sell stops (trigger on the way down).
* Cancel/replace only while ``New`` (once ``Activated`` a conditional is a NORMAL
  order and is managed there). ``version`` is pinned to 2026-07-23 in the client.
* Trading token is read from the state file written by the OTP minter.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pynecore.core.plugin import override
from pynecore.core.plugin.broker import BrokerPlugin
from pynecore.core.broker.models import (
    CancelDispositionOutcome, CapabilityLevel, ExchangeCapabilities,
    ExchangeOrder, ExchangePosition, LegType, OrderEvent, OrderStatus,
    OrderType,
)

from .cancel_disposition import (
    aggregate as _aggregate_dispositions,
    classify_readback as _classify_readback,
)
from .feed_health import FeedHealth
from .journal_wiring import (
    iter_journal_identities, journal_child_ref, journal_disposition_unknown,
    journal_rejected, journal_server_ref, journal_submitted, journal_terminal,
)
from .transport_errors import guard as _guard_transport
from .page_completeness import (
    BOOK_READ_DEADLINE_S, POSITIONS_PAGE_SIZE,
    book_page_count, is_exposure_row, positions_complete,
)
from pynecore.types.ohlcv import OHLCV
from pynecore.lib import log
from pynecore.core.broker.exceptions import (
    AuthenticationError, BrokerManualInterventionError, ExchangeConnectionError,
    ExchangeOrderRejectedError, ExchangeRateLimitError, InsufficientMarginError,
    OrderDispositionUnknownError,
)
from pynecore.core.broker.idempotency import (
    KIND_ENTRY, KIND_EXIT_TP, KIND_EXIT_SL, KIND_CLOSE)

from .provider import DNSEConfig, DNSEProvider
from . import errors

_SIDE_TO_DNSE = {"buy": "NB", "sell": "NS"}
_DNSE_TO_SIDE = {"NB": "buy", "NS": "sell"}

#: DNSE order status -> PyneCore OrderStatus (keys uppercased, ``_``/``-`` stripped).
#: Covers NORMAL statuses + the STOP/OCO lifecycle (New -> Activated -> terminal).
_STATUS_MAP = {
    "PENDING": OrderStatus.PENDING, "PENDINGNEW": OrderStatus.PENDING,
    "NEW": OrderStatus.OPEN, "OPEN": OrderStatus.OPEN,
    "ACTIVATED": OrderStatus.OPEN,       # conditional triggered -> now a working order
    "PENDINGCANCEL": OrderStatus.OPEN,   # cancel in flight; still live
    "PARTIALLYFILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED, "CANCELED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED, "EXPIRED": OrderStatus.EXPIRED,
    "FAILED": OrderStatus.REJECTED,
}

_TERMINAL_STATUSES = frozenset({
    OrderStatus.FILLED, OrderStatus.CANCELLED,
    OrderStatus.REJECTED, OrderStatus.EXPIRED,
})

#: Restore ``LegType`` from the journal's ``leg_kind`` extras (#36).
_LEG_TYPE_BY_NAME = {member.name: member for member in LegType}

#: Read-side dispositions that mean the CREDENTIAL is refused (#54): the only
#: failure kinds that can satisfy the feed-health all-books halt condition.
_AUTH_DISPOSITIONS = (errors.Disposition.AUTH, errors.Disposition.AUTH_TOKEN)

#: TradingView timeframe -> seconds (bar-period math for the closed-bar poll).
_TF_SECONDS = {"1": 60, "3": 180, "5": 300, "15": 900, "30": 1800,
               "60": 3600, "1H": 3600, "1D": 86400}

#: Order-book categories to poll/scan for WORKING orders + fills. An OCO's real
#: working order is the spawned NORMAL LO (tracked via its externalOrderId), and a
#: STOP is its own working order — so we scan NORMAL + STOP and skip the OCO
#: umbrella book (whose records would double-count the LO / linger as zombies).
#: Do NOT "fix" #43 by adding "OCO" here (S3 on that card — rejected: it would
#: double-count get_open_orders and cost +50% Get-Orders budget forever): an
#: umbrella whose LO is unknown at place time goes into ``_pending_oco`` and is
#: drained by ``watch_orders`` instead.
_CATEGORIES = ("NORMAL", "STOP")

#: Books a CANCEL may need to probe for an id with no category record (#45) —
#: unlike the scan set above this must include OCO: an unrecorded umbrella id
#: answers 404 on both scanned books, and "not found everywhere probed" is
#: treated as already-gone.
_CANCEL_PROBE_BOOKS = ("NORMAL", "STOP", "OCO")

#: LegType.name -> idempotency KIND, for the disposition-unknown client_order_id.
_LEG_KIND = {
    "ENTRY": KIND_ENTRY, "TAKE_PROFIT": KIND_EXIT_TP,
    "STOP_LOSS": KIND_EXIT_SL, "CLOSE": KIND_CLOSE,
}


# Phase 0a (#66): DNSEBrokerConfig lives in config.py; re-exported here
# so existing imports keep working.
from .config import DNSEBrokerConfig  # noqa: F401  (re-export)

class DNSEBroker(DNSEProvider[DNSEBrokerConfig], BrokerPlugin[DNSEBrokerConfig]):
    """DNSE broker: Vietnam derivatives (native STOP/OCO) and stocks."""

    plugin_name = "DNSE Broker"
    #: Phase 0a (#66): the one config type both bases agree on — pyright's
    #: structural incompatibility came from the provider pinning DNSEConfig.
    config: DNSEBrokerConfig
    Config = DNSEBrokerConfig

    #: Netting-native venue — no hedged-leg emulation.
    position_port = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._connected = False
        self._account_no: str | None = None
        #: intent_key -> [venue order ids] (the handle a later cancel/amend uses).
        self._order_ids: dict[str, list[str]] = {}
        #: venue order id -> (pine_id, from_entry, leg_type) for event tagging.
        self._identity: dict[str, tuple] = {}
        #: venue order id -> the book it lives in ("NORMAL"/"STOP") for cancel/amend.
        self._order_category: dict[str, str] = {}
        #: venue order id -> (cumulative_fill, status) from the last poll.
        self._last_seen: dict[str, tuple] = {}
        _cfg = getattr(self, "config", None)
        self._poll_interval: float = float(
            getattr(_cfg, "order_poll_interval", None) or 0.5)
        self._bar_poll_interval: float = float(
            getattr(_cfg, "bar_poll_interval", None) or 3.0)
        # --- #37 dual-mode feed (S1' — dispatch inside watch_ohlcv) ---
        self._feed_mode: str = str(getattr(_cfg, "feed_mode", None) or "ohlc")
        if self._feed_mode not in ("ohlc", "tick"):
            raise ValueError(
                f"feed_mode must be 'ohlc' or 'tick', got {self._feed_mode!r} "
                f"(fail-fast: a typo silently falling back would hide tick mode)")
        self._tick_poll_interval: float = float(
            getattr(_cfg, "tick_poll_interval", None) or 2.0)
        self._tick_close_timeout: float = float(
            getattr(_cfg, "tick_close_timeout", None) or 20.0)
        #: Session-cumulative totalVolumeTraded of the newest accepted print,
        #: PER BOARD — measured live 2026-08-25 (probe_trades_latest.py): the
        #: endpoint returns G1 (continuous) AND T1 (put-through/block) rows
        #: interleaved, each with its OWN independent, non-comparable counter (a
        #: single global cursor silently dropped valid G1 prints as replays
        #: whenever a lower-cursor T1 row interleaved). Only G1 feeds synthesis.
        self._tick_board = "G1"
        self._tick_cursor: float = 0.0
        self._tick_slot: int = 0          # bar-start ts (s) of the forming bar
        self._tick_bar: dict | None = None
        self._tick_close_deadline: float | None = None
        self._tick_throttled: bool = False
        #: Cancel-disposition read-back poll (see ``_readback_disposition``);
        #: 4 reads / 3 sleeps = a <=3 s budget (#55 panel) — a leftover UNKNOWN
        #: resolves on the watch/poll cadence instead of blocking the loop.
        self._cancel_verify_attempts: int = 4
        self._cancel_verify_delay: float = 0.7
        #: Wall-clock deadline for one off-loop positions/book read (#62);
        #: module default sits under the engine's ~30 s execute budget.
        self._book_read_deadline_s: float = BOOK_READ_DEADLINE_S
        #: #54 feed-health thresholds, in watch CYCLES (0.5 s cadence): first
        #: warning ~10 s into a persistent failure, re-warn ~60 s, and the
        #: all-books-AUTH halt only after ~60 s (a latched halt is
        #: irreversible — an auth blip must warn, never halt). Tunable for
        #: tests like ``_cancel_verify_attempts``.
        self._feed_warn_after: int = 20
        self._feed_rewarn_every: int = 120
        self._feed_halt_after: int = 120
        #: Single-flight wait per cycle on the in-flight poll read; a hung
        #: socket (60 s timeout) counts as stuck cycles, never a stack of
        #: abandoned worker threads on the SHARED default executor.
        self._watch_read_deadline_s: float = 10.0
        #: Child-adoption retry shape for an Activated conditional whose
        #: ``externalOrderId`` is not published yet (#42-A). Counted in POLLS, not
        #: seconds, so the schedule is deterministic in tests and cannot hot-spin
        #: on a wall clock. At the 0.5 s default cadence: retry every poll for
        #: 10 s (the measured stale-replica lag is ~10 s), then once per 10 s so a
        #: permanently unresolvable shell cannot drain the Get-Order-Detail budget
        #: (10,000/h — shared with ``_readback_disposition``, so starving it would
        #: turn a fill-visibility bug into a cancel-verification bug), and give up
        #: at ~2 min by escalating for manual intervention.
        self._adopt_fast_polls: int = 20
        self._adopt_slow_every: int = 20
        self._adopt_give_up_polls: int = 240
        #: parent order id -> adoption attempts so far. ADVISORY only: losing it
        #: (restart) costs a faster retry, never correctness.
        self._adopt_attempts: dict[str, int] = {}
        #: OCO umbrella ids whose working LO was unresolved at PLACE time (#43);
        #: they live on a book ``_CATEGORIES`` never scans, so ``watch_orders``
        #: drains this set every cycle. In-memory like the rest (#36): lost on a
        #: restart, so recovery must re-derive from the venue books.
        self._pending_oco: set[str] = set()
        self._last_bar_ts: int = 0
        self._loan_id: int | None = None

    # --- account / token ---

    @property
    def account_id(self) -> str:
        if self._account_no:
            return self._account_no
        assert self.config is not None
        if self.config.account_no:
            self._account_no = self.config.account_no
            return self._account_no
        status, body = self.client.get_accounts()
        accounts = (body.get("accounts") or []) if isinstance(body, dict) else []
        if status != 200 or not accounts:
            raise RuntimeError(f"cannot resolve account: {status} {body}")
        self._account_no = accounts[0]["id"]
        return self._account_no

    def _token(self) -> str:
        """Trading token: the OTP-minter state file first, config as fallback."""
        assert self.config is not None
        path = Path(self.config.token_file)
        if path.exists():
            try:
                token = json.loads(path.read_text()).get("trading_token")
                if token:
                    return token
            except (ValueError, OSError):
                pass
        if self.config.trading_token:
            return self.config.trading_token
        raise RuntimeError(
            f"no trading_token — run the OTP minter (writes {self.config.token_file}) "
            f"or set trading_token in the plugin config"
        )

    # --- capabilities ---

    @override
    def get_capabilities(self) -> ExchangeCapabilities:
        return ExchangeCapabilities(
            # Native server-side STOP (stays New until triggered -> cleanly
            # cancellable). Used for entry-stops and the SL-only exit.
            stop_order=CapabilityLevel.NATIVE,
            # Native OCO bracket: one server-side order (a TP LO that auto-amends
            # to the SL price if the SL condition hits). The venue runs the
            # one-cancels-other; the plugin tracks the OCO's working LO via
            # externalOrderId. See execute_exit.
            tp_sl_bracket=CapabilityLevel.NATIVE,
            # SOFTWARE, not NATIVE (#33): the OCO above is the single-exit
            # bracket ONLY — no DNSE payload can link separate orders into a
            # group (Live-L1-T11: oca members are venue-strangers). Declaring
            # NATIVE suppresses the sync engine's fill-time sibling cancel
            # (_cascade_oca_cancel), stranding the far leg of an oca.cancel
            # ENTRY group as a live working order after the near leg fills.
            oca_cancel=CapabilityLevel.SOFTWARE,
            # Not natively supported by DNSE conditional orders.
            trailing_stop=CapabilityLevel.SOFTWARE,
            partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
            partial_qty_bracket_exit_pyramiding=CapabilityLevel.SOFTWARE,
            # PUT /orders/{id} amends price+quantity atomically (while New).
            amend_order=CapabilityLevel.NATIVE,
            cancel_all=CapabilityLevel.SOFTWARE,
            reduce_only=CapabilityLevel.SOFTWARE,
            # REST-poll of the order books, not a live push channel.
            watch_orders=CapabilityLevel.SOFTWARE,
            fetch_position=CapabilityLevel.NATIVE,
            # No client-supplied order id in the place payload.
            idempotency=CapabilityLevel.SOFTWARE,
            short_selling=CapabilityLevel.NATIVE,
        )

    # --- live plumbing (REST-only) ---

    @override
    async def connect(self) -> None:
        # REST-only: nothing to connect. Touch the client so the endpoint
        # banner is logged. NOTE: this validates NOTHING — a dead credential
        # surfaces on the first classified read/write (#68), not here.
        _ = self.client
        self._connected = True
        self._restore_identity_from_journal()

    def _restore_identity_from_journal(self) -> None:
        '''#36: journal-ROOTED restart adoption (Live-L1-T16).

        Restores the three in-memory maps from this run identity's live
        journal rows so a same-label relaunch re-owns its resting venue
        orders (identity, book category, cancellability). Panel rules:
        adoption starts from JOURNALLED rows only — a book row with no
        journal root is FOREIGN (the operator's) and is never touched; a
        conditional parent whose child ref is missing (crash window) gets a
        best-effort child CHASE via the parent detail's externalOrderId;
        idempotent — existing in-memory entries are never overwritten (the
        engine's own store_ctx.replay() re-points envelopes separately).
        '''
        if self.store_ctx is None:
            return
        adopted = 0
        for journal_row in iter_journal_identities(self.store_ctx):
            leg_type = _LEG_TYPE_BY_NAME.get(journal_row.leg_kind)
            primary = journal_row.venue_ids[0] if journal_row.venue_ids else None
            for venue_id in journal_row.venue_ids:
                if venue_id in self._identity:
                    continue
                self._identity[venue_id] = (journal_row.pine_id or None,
                                            journal_row.from_entry, leg_type)
                self._order_category[venue_id] = (
                    "NORMAL" if venue_id == journal_row.child_id
                    else journal_row.category)
                if journal_row.intent_key:
                    self._order_ids.setdefault(
                        journal_row.intent_key, []).append(venue_id)
                adopted += 1
            if (primary is not None and journal_row.child_id is None
                    and journal_row.category in ("STOP", "OCO")):
                # Crash-window chase: the parent may have triggered while we
                # were down — its economics live on the un-journalled child.
                try:
                    detail = self._resolve_child_detail(primary,
                                                        journal_row.category)
                except Exception:                                   # noqa: BLE001
                    detail = None        # venue unreachable: next poll retries
                child = (detail or {}).get("externalOrderId")
                if child and str(child) not in self._identity:
                    child = str(child)
                    self._identity[child] = (journal_row.pine_id or None,
                                             journal_row.from_entry, leg_type)
                    self._order_category[child] = "NORMAL"
                    if journal_row.intent_key:
                        self._order_ids.setdefault(
                            journal_row.intent_key, []).append(child)
                    journal_child_ref(self.store_ctx, parent_venue_id=primary,
                                      child_id=child)
                    adopted += 1
        if adopted:
            log.broker_info("%s", (
                f"journal restore: re-owned {adopted} venue id(s) from the "
                f"run's journal rows (#36) — foreign book rows untouched"))

    @override
    async def disconnect(self) -> None:
        self._connected = False

    @property
    @override
    def is_connected(self) -> bool:
        return self._connected

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        """The engine's one bar-feed entry point (live_runner hard-calls it):
        dispatch by ``feed_mode`` — the parity-proven closed-bar path stays
        byte-identical and isolated from the tick body (#37 S1')."""
        if self._feed_mode == "tick":
            return await self._watch_ohlcv_tick(symbol, timeframe)
        return await self._watch_ohlcv_closed(symbol, timeframe)

    async def _watch_ohlcv_closed(self, symbol: str, timeframe: str) -> OHLCV:
        """Yield the next CLOSED bar by polling REST ``/price/ohlc``."""
        resolution = self.to_exchange_timeframe(timeframe)
        period = _TF_SECONDS.get(timeframe, 300)
        while True:
            now = int(time.time())
            status, body = await asyncio.to_thread(lambda: _guard_transport(
                lambda: self.client.get_ohlc(
                    self.market_type,
                    {"symbol": self.symbol, "resolution": resolution,
                     "from": now - period * 5, "to": now})))
            if status == 200 and isinstance(body, dict) and body.get("t"):
                times = body["t"]
                idx = len(times) - 1
                while idx >= 0 and int(times[idx]) + period > now:
                    idx -= 1            # skip the still-forming bar
                if idx >= 0:
                    ts = int(times[idx])
                    if ts > self._last_bar_ts:
                        self._last_bar_ts = ts
                        return OHLCV(
                            timestamp=ts * 1000,
                            open=float(body["o"][idx]), high=float(body["h"][idx]),
                            low=float(body["l"][idx]), close=float(body["c"][idx]),
                            volume=float(body["v"][idx]), is_closed=True)
            await asyncio.sleep(self._bar_poll_interval)

    async def _watch_ohlcv_tick(self, symbol: str, timeframe: str) -> OHLCV:
        """Tick mode (#37): poll ``/trades/latest``, synthesize the developing
        bar, emit ``is_closed=False`` on change; at rollover the venue's
        official closed bar is authoritative — fetched for up to
        ``tick_close_timeout`` seconds, after which the SYNTHESIZED bar is
        closed loudly (who-closed=SYNTH; the L4 red line grades this).

        Emit-ordering guard (#37 panel): no forming update for slot N+1 is
        emitted before slot N's close — a forming bar overtaking its close
        double-increments bar_index engine-side and moves time backwards.
        """
        period = _TF_SECONDS.get(timeframe, 300)
        while True:
            now = time.time()
            slot = int(now - (now % period))
            if self._tick_bar is not None and self._tick_slot < slot:
                # rollover: close slot N before any forming N+1
                if self._tick_close_deadline is None:
                    self._tick_close_deadline = now + self._tick_close_timeout
                official = await asyncio.to_thread(
                    self._tick_fetch_official_close, period)
                if official is not None:
                    return official
                if time.time() >= self._tick_close_deadline:
                    bar, ts = self._tick_bar, self._tick_slot
                    self._tick_bar, self._tick_close_deadline = None, None
                    self._last_bar_ts = ts
                    log.broker_warning(
                        "tick mode: official close for bar %d withheld past "
                        "%.0fs — closing the SYNTHESIZED bar (who-closed=SYNTH; "
                        "expected at session close, Live-L4-T03)",
                        ts, self._tick_close_timeout)
                    return OHLCV(timestamp=ts * 1000, open=bar["o"], high=bar["h"],
                                 low=bar["l"], close=bar["c"], volume=bar["v"],
                                 is_closed=True)
                await asyncio.sleep(self._tick_poll_interval)
                continue
            update = await asyncio.to_thread(self._tick_poll_once, slot)
            if update is not None:
                return update
            await asyncio.sleep(self._tick_poll_interval)

    def _tick_fetch_official_close(self, period: int) -> "OHLCV | None":
        """One attempt to read slot ``self._tick_slot``'s OFFICIAL closed bar."""
        ts = self._tick_slot
        status, body = _guard_transport(lambda: self.client.get_ohlc(
            self.market_type,
            {"symbol": self.symbol,
             "resolution": self.to_exchange_timeframe(str(period // 60 or 1)),
             "from": ts - period, "to": ts + 2 * period}))
        if status != 200 or not isinstance(body, dict) or not body.get("t"):
            return None
        for idx, row_ts in enumerate(body["t"]):
            if int(row_ts) == ts:
                self._tick_bar, self._tick_close_deadline = None, None
                self._last_bar_ts = ts
                return OHLCV(
                    timestamp=ts * 1000,
                    open=float(body["o"][idx]), high=float(body["h"][idx]),
                    low=float(body["l"][idx]), close=float(body["c"][idx]),
                    volume=float(body["v"][idx]), is_closed=True)
        return None

    def _tick_poll_once(self, slot: int) -> "OHLCV | None":
        """One ``/trades/latest`` poll: merge new prints into the forming bar.

        Dedup cursor: ``totalVolumeTraded`` is session-cumulative and monotone
        PER BOARD (measured live 2026-08-25: G1/T1 counters are independent and
        NOT comparable to each other — a global cursor silently dropped valid
        prints as replays). Only ``self._tick_board`` rows are compared/kept;
        strictly-greater-than-cursor (not >=) so a same-volume edge case cannot
        be mistaken for a replay, and cannot double-count either.
        Board filtering: T1 (put-through/block trades) is EXCLUDED — its prices
        are off-market negotiated trades that would contaminate H/L.
        """
        status, body = self.client.get_latest_trade(self.resolve_contract())
        if status == 429:
            if not self._tick_throttled:
                self._tick_throttled = True
                log.broker_warning(
                    "tick mode: /trades/latest throttled (429) — degraded to "
                    "poll-and-hope cadence; forming updates may stall (#37)")
            return None
        if status != 200:
            return None
        if self._tick_throttled:
            self._tick_throttled = False
            log.broker_info("tick mode: /trades/latest throttle cleared")
        rows = (body if isinstance(body, list)
                else (body.get("trades") or body.get("data") or [])
                if isinstance(body, dict) else [])
        changed = False
        board_rows = [r for r in rows if str(r.get("boardId")) == self._tick_board]
        for raw in sorted(board_rows, key=lambda r: float(r.get("totalVolumeTraded") or 0)):
            total = float(raw.get("totalVolumeTraded") or 0)
            if total <= self._tick_cursor:
                continue                      # replayed print from an earlier poll
            price = float(raw.get("matchPrice") or 0)
            qty = float(raw.get("matchQtty") or 0)
            self._tick_cursor = total
            if price <= 0:
                continue
            if self._tick_bar is None or self._tick_slot != slot:
                self._tick_slot = slot
                self._tick_bar = {"o": price, "h": price, "l": price,
                                  "c": price, "v": 0.0}
                self._tick_close_deadline = None
            bar = self._tick_bar
            bar["h"] = max(bar["h"], price)
            bar["l"] = min(bar["l"], price)
            bar["c"] = price
            bar["v"] += qty
            changed = True
        if not changed or self._tick_bar is None:
            return None
        bar = self._tick_bar
        return OHLCV(timestamp=self._tick_slot * 1000, open=bar["o"],
                     high=bar["h"], low=bar["l"], close=bar["c"],
                     volume=bar["v"], is_closed=False)

    # --- order construction ---

    def _to_exchange_order(self, raw: dict) -> ExchangeOrder:
        filled = float(raw.get("fillQuantity") or 0)
        qty = float(raw.get("quantity") or 0)
        stop_price = raw.get("stopPrice")
        return ExchangeOrder(
            id=str(raw.get("id")),
            symbol=raw.get("symbol") or self.symbol or "",
            side=_DNSE_TO_SIDE.get(raw.get("side", ""), "buy"),
            order_type=OrderType.LIMIT,
            qty=qty, filled_qty=filled,
            remaining_qty=float(raw.get("leaveQuantity") or max(qty - filled, 0)),
            price=float(raw.get("price") or 0) or None,
            stop_price=float(stop_price) if stop_price else None,
            average_fill_price=float(raw.get("averagePrice") or 0) or None,
            status=_STATUS_MAP.get(
                str(raw.get("orderStatus", "")).upper().replace("_", "").replace("-", ""),
                OrderStatus.PENDING),
            timestamp=int(time.time() * 1000),
            fee=0.0, fee_currency="VND",
        )

    def _marketable_price(self, side: str) -> float:
        """Band-edge price for a market intent — ceiling to buy, floor to sell."""
        ceiling, floor = self._band()
        return ceiling if side == "buy" else floor

    def _band(self) -> tuple[float, float]:
        """(ceilingPrice, floorPrice) for the symbol — the venue's hard price limits."""
        row = self._secdef(self.symbol or "")
        ceiling = float(row.get("ceilingPrice") or 0)
        floor = float(row.get("floorPrice") or 0)
        if not ceiling or not floor:
            raise RuntimeError(
                f"cannot read the price band: secdef has no ceiling/floor for "
                f"{self.symbol!r}")
        return ceiling, floor

    def _stop_already_crossed(self, side: str, stop_price: float) -> bool:
        """Is a stop's trigger condition already TRUE at placement time?

        Pine treats a crossed stop as an IMMEDIATE entry (the backtest oracle
        fills it at the next open — measured 2026-08-18, #34). Detection reads
        the venue's last 1-minute close (one REST call, stop entries only).
        Fails OPEN: any read problem returns False, i.e. the conditional path —
        today's behaviour — so a market-data hiccup can never block an order.
        """
        try:
            now = int(time.time())
            status, body = _guard_transport(lambda: self.client.get_ohlc(
                self.market_type, {
                    "symbol": self.symbol, "resolution": "1",
                    "from": now - 600, "to": now}))
            if status != 200 or not isinstance(body, dict) or not body.get("c"):
                return False
            last = float(body["c"][-1])
        except Exception:                                          # noqa: BLE001
            return False
        return last >= stop_price if side == "buy" else last <= stop_price

    def _stop_fill_price(self, side: str, stop_price: float) -> float:
        """Limit price for the LO a triggered stop emits — trigger + 2x slippage.

        Pine's ``entry(stop=)`` / ``exit(stop=)`` mean a stop-**market**: once the
        trigger prints, you want out (or in) and accept slippage. DNSE has no
        stop-market — every order is an ``LO``, and a conditional order emits that LO
        at ``price`` when ``stopPrice`` is crossed. Posting it *at* the trigger (the
        old behaviour) makes it a stop-**limit**: if price gaps through, the LO never
        fills, so the stop silently does nothing — triggered, unfilled, still exposed.

        The LO is therefore offset **through** the trigger by ``2 x slippage`` ticks
        (the strategy's own ``strategy(slippage=)``, in ticks) so it can cross the
        spread. Doubling gives room for the book to move between trigger and arrival
        while still bounding the worst fill — unlike a band-edge order, which always
        fills but can print up to +/-7% away.

        Falls back to :attr:`stop_slippage_ticks` when the script declares no slippage
        (Pine's default is 0, which would reproduce the never-fills bug), and is
        always clamped into the venue band so the order cannot be rejected.
        """
        from pynecore import lib
        script = getattr(lib, "_script", None)
        ticks = int(getattr(script, "slippage", 0) or 0) * 2
        if ticks <= 0:
            ticks = int(getattr(self.config, "stop_slippage_ticks", 0) or 0)
        offset = ticks * self._mintick()
        price = stop_price + offset if side == "buy" else stop_price - offset
        ceiling, floor = self._band()
        return round(min(max(price, floor), ceiling), 1)

    def _mintick(self) -> float:
        """Tick size for the traded contract (VN30F1M derivatives: 0.1)."""
        try:
            return float(self.get_symbol_info().mintick) or 0.1
        except Exception:                                          # noqa: BLE001
            return 0.1

    def _gtd(self, days: int = 7) -> str:
        """RFC3339 expiry for a GTD STOP, CLAMPED to the contract's final trade date.

        A derivatives contract stops trading on its ``finalTradeDate``; DNSE rejects any
        order whose GTD reaches past it with ``CO-ORD-006 Validate Order Failed``.
        Blindly adding 7 days therefore makes every native STOP/OCO — including protective
        stop-losses — unplaceable during the last week of each contract month.

        Measured 2026-08-14: VN30F1M's finalTradeDate was 2026-08-20, +7 days gave
        2026-08-21, and every conditional place was refused. The same call succeeded on
        2026-08-13 (+7 = 2026-08-20, exactly the final trade date), which is why this
        surfaced as a sudden, whole-day failure rather than a gradual one.

        Falls back to the plain +days window when the secdef carries no usable date, so a
        missing field cannot make the plugin unable to place anything at all.
        """
        target = datetime.now(timezone.utc) + timedelta(days=days)
        try:
            final = str(self._secdef(self.symbol or "").get("finalTradeDate") or "")[:10]
            if final:
                # Cap at MIDNIGHT UTC of the final trade date, which is how DNSE itself
                # reports it. Not 23:59Z: the venue reads the date in ICT (UTC+7), so
                # 23:59Z on the final date is already 07:00 the NEXT day there and is
                # refused. Measured 2026-08-14 — GTD 2026-08-20T04:00Z was accepted,
                # 2026-08-20T23:59Z was not.
                last = datetime.strptime(final, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                target = min(target, last)
        except Exception:                                           # noqa: BLE001
            pass                        # no usable expiry -> keep the plain window
        return target.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _loan_package_id(self) -> int:
        if self._loan_id is None:
            status, body = self.client.get_loan_packages(self.account_id, self.market_type)
            if status != 200 or not isinstance(body, dict) or not body.get("loanPackages"):
                raise RuntimeError(f"cannot resolve loanPackageId: {status} {body}")
            self._loan_id = body["loanPackages"][0]["id"]
        return self._loan_id

    # --- error handling (see errors.py + docs/plan/dnse-error-handling.md) ---

    def _emit(self, classified, *, action: str, ident: str) -> None:
        """Emit the one structured ``[BROKER]`` line for a classified error."""
        fn = {"error": log.broker_error, "warning": log.broker_warning,
              "info": log.broker_info}.get(classified.level, log.broker_warning)
        fn("%s", classified.log_message(action, ident))

    def _write(self, call):
        """``call(token) -> (status, body)`` — ONE venue write, never a retry.

        The retired "token-reread (retry once)" on INVALID_TRADING_TOKEN was
        a second IDENTICAL write: ``_token()`` reads the state file fresh on
        EVERY call, so the first attempt already carried the freshest token,
        and the measured #51/#46 windows showed retrying (and re-minting)
        never reclaims a refusal — it only doubled writes into the lockout
        (#58). The refusal surfaces through the classify path, whose message
        carries the operator action. #67: a raw transport exception here
        becomes the ``(0, NO_RESPONSE)`` sentinel so the classify path parks
        the write (OrderDispositionUnknownError) instead of killing the run.
        """
        return _guard_transport(lambda: call(self._token()))

    @staticmethod
    def _ident_str(envelope, leg_type) -> str:
        intent = getattr(envelope, "intent", None)
        pine = getattr(intent, "pine_id", None) or "?"
        leg = getattr(leg_type, "value", None) or getattr(leg_type, "name", None) or "?"
        key = getattr(intent, "intent_key", None) or "?"
        return f"{pine}/{leg} intent={key}"

    @staticmethod
    def _coid(envelope, leg_type) -> str:
        kind = _LEG_KIND.get(getattr(leg_type, "name", ""), KIND_ENTRY)
        try:
            return envelope.client_order_id(kind)
        except Exception:
            return getattr(getattr(envelope, "intent", None), "intent_key", "") or ""

    def _raise_write_error(self, status, body, *, action: str, ident: str,
                           coid: str) -> None:
        """Classify a WRITE reply; on failure emit its log line and raise the
        matching ``BrokerError``. Returns quietly on a 2xx success."""
        classified = errors.classify(status, body, is_write=True)
        if classified is None:
            return
        self._emit(classified, action=action, ident=ident)
        disposition = classified.disposition
        detail = f"{classified.code} {classified.message}".strip()
        if disposition is errors.Disposition.MARGIN:
            raise InsufficientMarginError(f"DNSE margin reject on {action}: {detail}")
        if disposition is errors.Disposition.RATE_LIMIT:
            raise ExchangeRateLimitError(f"DNSE rate limit on {action}: {detail}",
                                         classified.retry_after)
        if disposition is errors.Disposition.DISPOSITION_UNKNOWN:
            raise OrderDispositionUnknownError(
                f"DNSE disposition unknown on {action}: {detail}", client_order_id=coid)
        if disposition in (errors.Disposition.AUTH, errors.Disposition.AUTH_TOKEN):
            raise AuthenticationError(f"DNSE auth on {action}: {detail}",
                                      reason=classified.code)
        if disposition is errors.Disposition.CONNECTION:
            raise ExchangeConnectionError(f"DNSE transient on {action}: {detail}")
        # REJECT / SESSION_REJECT (+ any TERMINAL/NOT_FOUND reaching a place/amend)
        raise ExchangeOrderRejectedError(f"DNSE rejected {action}: {detail}")

    def _place(self, envelope, side: str, qty: float, *, price: float,
               category: str = "NORMAL", stop_price: float | None = None,
               stop_order_price: float | None = None, leg_type=None
               ) -> list[ExchangeOrder]:
        """Place one native order (NORMAL / STOP / OCO) and record its identity."""
        payload = {
            "symbol": self.resolve_contract(),   # tradable KRX contract, not the alias
            "side": _SIDE_TO_DNSE[side],
            "orderType": "LO",
            "price": round(float(price), 1),
            "quantity": int(qty),
            "loanPackageId": self._loan_package_id(),
        }
        if category == "STOP":
            payload.update({
                "stopPrice": round(float(stop_price), 1),
                "conditionOperator": ">=" if side == "buy" else "<=",
                "durationType": "GTD",
                "durationDateTime": self._gtd(),
            })
        elif category == "OCO":
            payload.update({
                "stopPrice": round(float(stop_price), 1),
                "stopOrderPrice": round(float(stop_order_price or stop_price), 1),
                "durationType": "DAY",
            })

        ident = self._ident_str(envelope, leg_type)
        coid = self._coid(envelope, leg_type)
        intent = envelope.intent
        # #36 persist-FIRST: the row exists BEFORE the POST leaves the process
        # — a crash in ANY later window leaves an auditable restart bridge.
        journal_submitted(
            self.store_ctx, coid=coid, symbol=payload["symbol"], side=side,
            qty=qty, intent_key=getattr(intent, "intent_key", None),
            pine_id=getattr(intent, "pine_id", None),
            from_entry=getattr(intent, "from_entry", None),
            leg_kind=getattr(leg_type, "name", None),
            category=category, order_type=payload["orderType"])
        status, body = self._write(lambda tok: self.client.post_order(
            self.account_id, self.market_type, payload, tok, order_category=category))
        try:
            self._raise_write_error(status, body, action="place", ident=ident,
                                    coid=coid)
        except OrderDispositionUnknownError:
            # #67 join: the reply was lost — the order may exist. phase comes
            # from the sentinel BODY (it never escapes classify, #36 panel).
            sentinel = body if isinstance(body, dict) else {}
            journal_disposition_unknown(
                self.store_ctx, coid=coid, phase=sentinel.get("phase"),
                transport=sentinel.get("transport"))
            raise
        except Exception:
            journal_rejected(self.store_ctx, coid=coid)  # write provably not live
            raise
        if not isinstance(body, dict):
            journal_rejected(self.store_ctx, coid=coid)
            raise ExchangeOrderRejectedError(
                f"DNSE place: non-dict success body: {body!r}")

        order = self._to_exchange_order(body)
        tracked_category = category
        if category == "OCO":
            # The OCO record is an umbrella; its real working order is a spawned
            # NORMAL LO. Track THAT — fills + cancels route by it — via the OCO
            # detail's externalOrderId (which appears ~instantly on activation).
            working = self._resolve_oco_lo(order.id)
            if working is not None:
                order = working
                tracked_category = "NORMAL"  # the live order lives on the NORMAL book
            else:
                # Stale replica: the LO is not visible yet. Track the umbrella
                # (cancel/amend must target the OCO book) and queue it for the
                # watch_orders drain — merely tracking it as "OCO" would park it
                # on a book no scan ever reads and its child's fill would stay
                # invisible (#43).
                self._pending_oco.add(str(order.id))
                log.broker_warning(
                    "OCO %s placed but its working LO is unresolved — queued for "
                    "poll-loop adoption (#43)", order.id)
        journal_server_ref(
            self.store_ctx, coid=coid, venue_id=order.id,
            category=tracked_category,
            umbrella_id=body.get("id") if tracked_category == "NORMAL"
            and category == "OCO" else None)
        key = getattr(intent, "intent_key", None)
        if key:
            self._order_ids.setdefault(key, []).append(order.id)
        self._identity[order.id] = (
            getattr(intent, "pine_id", None),
            getattr(intent, "from_entry", None),
            leg_type,
        )
        # Cancel/amend must target the book the tracked order actually lives in
        # (a STOP entry -> STOP; an OCO -> its NORMAL working LO). Guessing
        # NORMAL-first let a wrong-book RESOURCE_NOT_FOUND look like a clean cancel.
        self._order_category[order.id] = tracked_category
        return [order]

    def _resolve_child_detail(self, parent_id: str, category: str) -> "dict | None":
        """One 429-aware DETAIL read of a conditional parent (or None).

        Only the DETAIL carries ``externalOrderId`` (the list view omits it).
        The single resolver core behind BOTH watch-loop feeders — the
        Activated-STOP adoption (#42-A) and the pending-OCO drain (#43) — so
        the two paths cannot drift (panel guard on #43).

        ONE read per call — unlike :meth:`_resolve_oco_lo`, which polls 6x
        because it runs at PLACE time inside a synchronous call with no outer
        retry. This runs inside the ``watch_orders`` poll loop, which already
        retries every cycle (see ``_adopt_child``); a nested retry would only
        burn a worker thread and the Detail budget.
        """
        status, body = self.client.get_order_detail(
            self.account_id, parent_id, self.market_type, order_category=category)
        if status == 429:
            # A 429 body is a dict WITHOUT externalOrderId, i.e. it looks
            # exactly like "child not published yet". Retrying a rate-limited
            # endpoint is how a stall becomes an outage — stop, let the poll
            # loop come back later.
            return None
        return body if isinstance(body, dict) else None

    async def _adopt_child(self, parent_id: str, pine_id, from_entry,
                           leg_type, category: str = "STOP"
                           ) -> "tuple[str, dict | None]":
        """Adopt the NORMAL-book child of a conditional parent (STOP or OCO).

        :return: ``("adopted", detail)`` once the child id is known (adopted now,
            or already adopted); ``("dead", detail)`` when the parent is terminal
            WITHOUT ever naming a child — nothing will ever come, the caller
            retires it; ``("pending", detail)`` otherwise — and on ``pending``
            the caller MUST leave the parent row out of ``_last_seen``, because a
            shell's status and fill never change again: a row marked seen here
            would be deduped forever and the child's fill would stay invisible
            (#42-A, the latent form of #39).
        """
        attempts = self._adopt_attempts.get(parent_id, 0) + 1
        self._adopt_attempts[parent_id] = attempts
        # Degrade the cadence past the fast window instead of hammering forever.
        if attempts > self._adopt_fast_polls and attempts % self._adopt_slow_every:
            return ("pending", None)
        detail = await asyncio.to_thread(
            self._resolve_child_detail, parent_id, category)
        external = detail.get("externalOrderId") if detail else None
        if external:
            child = str(external)
            if child not in self._identity:
                self._identity[child] = (pine_id, from_entry, leg_type)
                self._order_category[child] = "NORMAL"
                # #36: the child is a SECOND ref on the parent's journal row —
                # the crash-window chase (journal-ROOTED adoption) needs it.
                journal_child_ref(self.store_ctx, parent_venue_id=parent_id,
                                  child_id=child)
                for ids in self._order_ids.values():
                    if parent_id in ids and child not in ids:
                        ids.append(child)
                log.broker_info(
                    "conditional ACTIVATED -> tracking child | parent=%s child=%s "
                    "pine=%s polls=%d", parent_id, child, pine_id, attempts)
            self._adopt_attempts.pop(parent_id, None)
            return ("adopted", detail)   # adopted now or already adopted
        if detail is not None:
            try:
                terminal = self._to_exchange_order(detail).status in _TERMINAL_STATUSES
            except Exception:                                        # noqa: BLE001
                terminal = False         # unparseable row: keep retrying
            if terminal:
                # Terminal WITHOUT a child: the parent died before spawning its
                # working order (DAY expiry, operator cancel, reject). Retire it
                # instead of grinding to the 2-minute escalation (#43 guard).
                self._adopt_attempts.pop(parent_id, None)
                return ("dead", detail)
        if attempts == 1 or attempts == self._adopt_fast_polls:
            log.broker_warning(
                "conditional [%s] has no externalOrderId after %d poll(s) | "
                "parent=%s pine=%s — the child fill is NOT trackable yet; retrying",
                category, attempts, parent_id, pine_id)
        if attempts >= self._adopt_give_up_polls:
            self._adopt_attempts.pop(parent_id, None)
            raise BrokerManualInterventionError(
                f"DNSE conditional {parent_id} (pine={pine_id}) never published "
                f"its child order id after {attempts} polls: its fill CANNOT be "
                f"tracked and the account may hold an unmanaged position — check "
                f"the venue and flatten manually")
        return ("pending", detail)

    def _resolve_oco_lo(self, oco_id: str, attempts: int = 6, delay: float = 0.15
                        ) -> "ExchangeOrder | None":
        """Return the OCO's working NORMAL LO as an ``ExchangeOrder`` (or None).

        The OCO spawns a NORMAL LO on activation (~instant); the OCO *detail*'s
        ``externalOrderId`` names it (the list view omits it), and the LO's own
        ``metadata.conditionOrderId`` points back. Poll briefly for activation,
        then fetch the LO. (Synchronous; the brief poll blocks the caller.)
        """
        for i in range(attempts):
            status, body = self.client.get_order_detail(
                self.account_id, oco_id, self.market_type, order_category="OCO")
            if status == 429:
                # A 429 body is a dict WITHOUT externalOrderId — keep polling and
                # a stall becomes an outage. Give up: the place-time caller queues
                # the umbrella for the watch-loop drain instead (#43).
                return None
            external = body.get("externalOrderId") if isinstance(body, dict) else None
            if external:
                lo_id = str(external)
                _, detail = self.client.get_order_detail(
                    self.account_id, lo_id, self.market_type, order_category="NORMAL")
                if isinstance(detail, dict):
                    return self._to_exchange_order(detail)
                return self._to_exchange_order({"id": lo_id})
            if i + 1 < attempts:
                time.sleep(delay)   # never after the LAST attempt
        return None

    # --- BrokerPlugin abstracts: execution ---

    @override
    async def execute_entry(self, envelope) -> list[ExchangeOrder]:
        from pynecore.core.broker.models import LegType
        intent = envelope.intent
        if intent.stop is not None:
            if self._stop_already_crossed(intent.side, intent.stop):
                # Crossed at placement (#34): Pine semantics = enter NOW (oracle
                # fills at the next open). A conditional here would either be
                # refused or emit its LO at trigger±slippage — arbitrarily far
                # behind the market. Plain stop -> marketable LO (band edge,
                # the same shape as a market intent); stop-limit -> LO at the
                # user's cap, which may rest (exactly TV's crossed stop-limit).
                log.broker_warning(
                    "crossed stop at placement -> immediate %s LO (Pine: instant "
                    "entry) | %s stop=%s",
                    "capped" if intent.limit is not None else "marketable",
                    self._ident_str(envelope, LegType.ENTRY), intent.stop)
                return self._place(envelope, intent.side, intent.qty,
                                   price=(intent.limit if intent.limit is not None
                                          else self._marketable_price(intent.side)),
                                   leg_type=LegType.ENTRY)
            # stop or stop-limit entry -> native STOP. An explicit ``limit`` is the
            # user asking for a stop-LIMIT, so it is honoured verbatim; a bare
            # ``stop`` means stop-MARKET, which DNSE cannot express, so the emitted
            # LO is priced through the trigger (see ``_stop_fill_price``).
            return self._place(envelope, intent.side, intent.qty,
                               price=(intent.limit if intent.limit is not None
                                      else self._stop_fill_price(intent.side, intent.stop)),
                               category="STOP",
                               stop_price=intent.stop, leg_type=LegType.ENTRY)
        if intent.limit is not None:
            return self._place(envelope, intent.side, intent.qty,
                               price=intent.limit, leg_type=LegType.ENTRY)
        return self._place(envelope, intent.side, intent.qty,
                           price=self._marketable_price(intent.side),
                           leg_type=LegType.ENTRY)

    @override
    async def execute_exit(self, envelope) -> list[ExchangeOrder]:
        """Bracket exit via native OCO (one server-side order), else STOP / LO.

        - tp + sl -> native OCO (``orderCategory=OCO``): the venue places a TP LO
          that auto-amends to the SL price if the SL condition hits, running the
          one-cancels-other server-side. The OCO spawns a working NORMAL LO on
          activation; ``_place`` tracks THAT (via ``externalOrderId``) so fills and
          cancels route by the id that actually acts.
        - sl only -> native STOP; tp only -> NORMAL LO.
        """
        from pynecore.core.broker.models import LegType
        from pynecore.core.broker.exceptions import OrderSkippedByPlugin
        intent = envelope.intent
        tp, sl = intent.tp_price, intent.sl_price
        # A stop-loss must FILL when it fires — price the LO it emits through the
        # trigger, never at it (see ``_stop_fill_price``). The TP leg keeps its exact
        # limit: a take-profit is a limit order by nature and must not slip.
        if tp is not None and sl is not None:
            return self._place(envelope, intent.side, intent.qty, price=tp,
                               category="OCO", stop_price=sl,
                               stop_order_price=self._stop_fill_price(intent.side, sl),
                               leg_type=LegType.TAKE_PROFIT)
        if sl is not None:
            return self._place(envelope, intent.side, intent.qty,
                               price=self._stop_fill_price(intent.side, sl),
                               category="STOP", stop_price=sl,
                               leg_type=LegType.STOP_LOSS)
        if tp is not None:
            return self._place(envelope, intent.side, intent.qty, price=tp,
                               leg_type=LegType.TAKE_PROFIT)
        raise OrderSkippedByPlugin(
            "DNSE plugin cannot express this exit: no tp_price/sl_price "
            "(trailing stops are not implemented)",
            intent_key=getattr(intent, "intent_key", ""))

    @override
    async def execute_close(self, envelope) -> ExchangeOrder:
        from pynecore.core.broker.models import LegType
        intent = envelope.intent
        return self._place(envelope, intent.side, intent.qty,
                           price=self._marketable_price(intent.side),
                           leg_type=LegType.CLOSE)[0]

    def _identity_for(self, order_id: str) -> tuple:
        return self._identity.get(order_id, (None, None, None))

    def _ids_for(self, envelope) -> list[str]:
        key = getattr(envelope.intent, "intent_key", None)
        return list(self._order_ids.get(key, [])) if key else []

    def _order_category_for(self, order_id: str):
        """The book a placed order lives in, recorded at place time (authoritative).

        No record -> None, so a cancel probes EVERY book. There is no safe
        leg-based guess (#45): the old "NORMAL" catch-all made `_cancel_one`
        probe one wrong book whose 404 then read as gone-from-every-book —
        three false cancel_one=True on a live conditional, measured 2026-08-24.
        Even STOP_LOSS cannot narrow to "STOP": an OCO umbrella carries an SL
        leg too. (Identity and category records share a lifecycle — both
        written at place, both in-memory #36 — so a leg-known/category-unknown
        id does not occur in practice anyway.)
        """
        return self._order_category.get(order_id) or None

    @override
    async def execute_cancel(self, envelope) -> bool:
        """Bool contract over the same disposition core as
        :meth:`execute_cancel_with_outcome` (#55).

        DECLARED CHANGE (#55 panel): True now means every id resolved to a
        POSITIVE cancelled-with-no-fill class (``CANCEL_CONFIRMED`` /
        ``TOO_LATE_TO_CANCEL``). Absence-from-every-book and fill-raced
        cancels previously returned True — the same lie the outcome variant
        told the engine. Core collapses both bools to UNKNOWN on the outcome
        path anyway (core/plugin/broker.py), and False means "retry" — the
        safe direction.
        """
        ids = self._ids_for(envelope)
        if not ids:
            return False
        outcomes = [await self._cancel_one_disposition(str(order_id))
                    for order_id in ids]
        return all(outcome in (CancelDispositionOutcome.CANCEL_CONFIRMED,
                               CancelDispositionOutcome.TOO_LATE_TO_CANCEL)
                   for outcome in outcomes)

    def _cancel_dependent_exits(self, entry_order_id: str) -> None:
        """RETIRED — do not call. Kept only as documentation of a measured dead end.

        This plugin-side cascade was reverted on 2026-08-14 after test T5 measured it
        breaking the ENGINE's ownership model live: the engine never asked for the exit
        leg's cancel, so its next sync saw a bot-owned order cancelled on the venue,
        RE-PLACED the exit (a brand-new orphan, id 575506 — the very thing the cascade
        was meant to prevent) and QUARANTINED the account, refusing every further
        dispatch ("Entry dispatch blocked by quarantine ... signal dropped").

        A cancel of a dependent exit must be ENGINE-initiated so its order ids land in
        ``sync_engine._strategy_cancel_expected_ids`` and its intent is retired — i.e.
        the fix for rubycell/pynecore#19 belongs in the engine's diff/cancel path, not
        here. Until then: strategies must cancel exit ids explicitly, as staged test T3
        does.

        Original rationale (still true, just mis-layered): DNSE does not cascade an
        entry cancel to its exit legs.

        DNSE does NOT cascade. Measured 2026-08-13 (rubycell/pynecore#19): an entry and
        its ``strategy.exit`` stop were placed in one OCA group; cancelling only the entry
        left the exit ``New`` at the venue — a naked buy-stop above the market, with no
        position to protect. Left there it does not merely linger: if it triggers it OPENS
        a position on a flat account, turning protection into exposure.

        The plugin already knows the binding — ``_identity`` maps every venue order id to
        ``(pine_id, from_entry, leg_type)`` — so an entry's dependants are exactly the
        tracked orders whose ``from_entry`` is that entry's ``pine_id``. Best-effort and
        idempotent: an already-terminal leg simply reports gone.
        """
        raise NotImplementedError(
            "RETIRED 2026-08-14 — see docstring. The executable cascade was "
            "removed with #55: its _cancel_one dependency was replaced by the "
            "disposition core, and this path must never run anyway.")

    async def _readback_disposition(self, order_id: str, category: str
                                    ) -> CancelDispositionOutcome:
        """Poll the order detail for a POSITIVE terminal classification.

        DNSE answers a conditional cancel with **200 and the order object** —
        an acknowledgement only; a resting STOP stayed ``New`` for >12 s after
        three ACKed cancels (measured 2026-08-13). So the venue itself must
        answer, and the answer must say WHY the order is done: ``Canceled``/
        ``Expired`` with zero fill confirms, any fill is ``ALREADY_FILLED``
        (G6), and a still-working read-back stays ``UNKNOWN`` (G5) so the
        engine retries on its own cadence — never ``STILL_OPEN``, which the
        engine treats as a confirmed cancel (models.py naming trap).

        Client calls run in a worker thread (G9: the thread touches ONLY the
        client — every broker-map mutation stays on the loop side) and pacing
        is ``await asyncio.sleep``, so the loop stays live for the very
        ``watch_orders`` fill feed that resolves the race (#55 panel). A
        detail 404 falls through to the history book: absence is not a
        disposition.
        """
        for attempt in range(self._cancel_verify_attempts):
            status, body = await asyncio.to_thread(
                lambda: _guard_transport(lambda: self.client.get_order_detail(
                    self.account_id, order_id, self.market_type,
                    order_category=category)))
            if status == 200 and isinstance(body, dict):
                try:
                    order = self._to_exchange_order(body)
                except Exception:                                   # noqa: BLE001
                    order = None            # unparseable row: fall through, retry
                if order is not None:
                    outcome = _classify_readback(order.status, order.filled_qty)
                    if outcome is not CancelDispositionOutcome.UNKNOWN:
                        return outcome
            elif errors.code_of(body) in errors.NOT_FOUND_CODES:
                return await self._history_disposition(order_id)
            if attempt + 1 < self._cancel_verify_attempts:
                await asyncio.sleep(self._cancel_verify_delay)
        return CancelDispositionOutcome.UNKNOWN

    async def _history_disposition(self, order_id: str
                                   ) -> CancelDispositionOutcome:
        """Absence is not a disposition — only a POSITIVE ``/orders/history``
        row may classify an id no book answers for (one call covers BOTH
        books; rows are date-prefixed ``20260818_538916`` under ``data``,
        measured 2026-08-19). No row -> ``UNKNOWN``: same-day row timeliness
        is an UNPROVEN venue premise (#55), and this failure shape degrades to
        a retry, never to a wrong terminal verdict."""
        today = datetime.now(timezone(timedelta(hours=7))).date()
        status, body = await asyncio.to_thread(lambda: _guard_transport(
            lambda: self.client.get_order_history(
                self.account_id, self.market_type,
                from_date=str(today - timedelta(days=1)), to_date=str(today),
                page_size=200)))
        if status == 200 and isinstance(body, dict):
            for row in body.get("data") or []:
                if str(row.get("id", "")).split("_")[-1] != str(order_id):
                    continue
                try:
                    order = self._to_exchange_order(row)
                except Exception:                                   # noqa: BLE001
                    return CancelDispositionOutcome.UNKNOWN
                return _classify_readback(order.status, order.filled_qty)
        return CancelDispositionOutcome.UNKNOWN

    async def _terminal_reject_disposition(self, order_id: str, category: str
                                           ) -> CancelDispositionOutcome:
        """The venue refused the cancel because the order is DONE — read WHY
        before classifying (the old ``TERMINAL_CODES -> treated-gone`` short
        circuit is the #55 double-open). On a conditional book the order may
        be an Activated shell whose economics moved to the NORMAL-book child
        (#41): classification follows the CHILD via ``externalOrderId`` (G2);
        a shell that names no child yet stays ``UNKNOWN``."""
        if category in ("STOP", "OCO"):
            detail = await asyncio.to_thread(
                self._resolve_child_detail, order_id, category)
            child_id = (detail or {}).get("externalOrderId")
            if child_id:
                return await self._readback_disposition(str(child_id), "NORMAL")
            if detail:
                try:
                    order = self._to_exchange_order(detail)
                except Exception:                                   # noqa: BLE001
                    return CancelDispositionOutcome.UNKNOWN
                return _classify_readback(order.status, order.filled_qty)
            return CancelDispositionOutcome.UNKNOWN
        return await self._readback_disposition(order_id, category)

    async def _cancel_one_disposition(self, order_id: str
                                      ) -> CancelDispositionOutcome:
        """Cancel by the recorded book (probe each if unknown, #45) and return
        a POSITIVE-observation disposition.

        Replaces the bool ``_cancel_one``, whose three True paths could not
        tell "cancelled, no fill" from "filled before the cancel landed"
        (#55): a FILLED read-back "took effect", ``TERMINAL_CODES`` was
        "treated-gone" unread, and absence from every book counted as
        success. A write refusal (#51 session binding, transient, reject) is
        a FAILED WRITE, not a disposition (G3) -> ``UNKNOWN`` so the engine
        retries; absence from every book asks the history — never concluded
        from silence.
        """
        hinted = self._order_category_for(order_id)
        categories = [hinted] if hinted else list(_CANCEL_PROBE_BOOKS)
        write_refused = False
        for category in categories:
            status, body = await asyncio.to_thread(
                self._write,
                lambda tok, category=category: self.client.cancel_order(
                    self.account_id, order_id, self.market_type, tok,
                    order_category=category))
            if status in (200, 204):
                # A 2xx from DNSE's cancel is an ACKNOWLEDGEMENT, not a
                # completion — the venue read-back decides the disposition.
                outcome = await self._readback_disposition(order_id, category)
                if outcome is CancelDispositionOutcome.UNKNOWN:
                    log.broker_warning("%s", (
                        f"cancel[{category}] http={status} ACKED but no terminal "
                        f"read-back within the budget -> UNKNOWN so the engine "
                        f"retries | order={order_id}"))
                else:
                    self._pending_oco.discard(order_id)
                    journal_terminal(self.store_ctx, venue_id=order_id,
                                     terminal_status=outcome.value)
                return outcome
            code = errors.code_of(body)
            if code in errors.TERMINAL_CODES:
                log.broker_info("%s", (
                    f"cancel[{category}] code={code} http={status} -> order is "
                    f"done; reading WHY before classifying | order={order_id}"))
                outcome = await self._terminal_reject_disposition(order_id, category)
                if outcome is not CancelDispositionOutcome.UNKNOWN:
                    self._pending_oco.discard(order_id)
                    journal_terminal(self.store_ctx, venue_id=order_id,
                                     terminal_status=outcome.value)
                return outcome
            if code in errors.NOT_FOUND_CODES:  # not in THIS book -> probe the next
                continue
            classified = errors.classify(status, body, is_write=True)
            if classified is not None:
                self._emit(classified, action=f"cancel[{category}]", ident=order_id)
            write_refused = True                # G3: refusal is not a disposition
        if write_refused:
            return CancelDispositionOutcome.UNKNOWN
        outcome = await self._history_disposition(order_id)
        if outcome is not CancelDispositionOutcome.UNKNOWN:
            self._pending_oco.discard(order_id)
        return outcome

    @override
    async def execute_cancel_with_outcome(self, envelope):
        ids = self._ids_for(envelope)
        if not ids:
            return CancelDispositionOutcome.UNKNOWN
        # Every id the envelope maps to (#47): after an adoption ids is
        # [consumed parent shell, working child]. Aggregation is conservative
        # (cancel_disposition.aggregate): any ALREADY_FILLED wins, then any
        # UNKNOWN keeps the engine retrying, then confirmed-class.
        return _aggregate_dispositions(
            [await self._cancel_one_disposition(str(order_id))
             for order_id in ids])

    @override
    async def modify_entry(self, old, new) -> list[ExchangeOrder]:
        return await self._amend(old, new, is_exit=False)

    @override
    async def modify_exit(self, old, new) -> list[ExchangeOrder]:
        return await self._amend(old, new, is_exit=True)

    async def _amend(self, old, new, *, is_exit: bool) -> list[ExchangeOrder]:
        ids = self._ids_for(old)
        if not ids:
            return await (super().modify_exit(old, new) if is_exit
                          else super().modify_entry(old, new))
        order_id = str(ids[0])
        intent = new.intent
        price = (getattr(intent, "limit", None) or getattr(intent, "stop", None)
                 or getattr(intent, "tp_price", None) or getattr(intent, "sl_price", None))
        category = self._order_category_for(order_id) or "NORMAL"
        payload = {"price": round(float(price), 1) if price else 0.0,
                   "quantity": int(intent.qty)}
        status, body = self._write(lambda tok: self.client.put_order(
            self.account_id, order_id, self.market_type, payload, tok,
            order_category=category))
        self._raise_write_error(
            status, body, action="amend",
            ident=f"{order_id} intent={getattr(intent, 'intent_key', '?')}", coid=order_id)
        if not isinstance(body, dict):
            raise ExchangeOrderRejectedError(
                f"DNSE amend: non-dict success body: {body!r}")
        return [self._to_exchange_order(body)]

    # --- BrokerPlugin abstracts: state ---

    def _read_book_rows_sync(self, category: str
                             ) -> "tuple[list[dict] | None, object | None]":
        """``(rows, None)`` for a complete book; ``(None, classified)`` on a
        failed read; ``(None, None)`` when pagination is unprovable.

        Drains ``totalPages`` (documented in the orders envelope; ignored
        until #61 — a book past 100 rows silently truncated). None-rows on
        any failed page, and on an over-cap/unparseable ``totalPages`` (G5'):
        a partial drain must never be returned as the book. The failure
        classification is RETURNED, never self-recorded (#54 panel): this
        helper serves both the watch loop and ``get_open_orders``, and only
        the watch loop feeds the feed-health ladder.
        """
        rows: list[dict] = []
        pages = 1
        page_index = 0
        while page_index < pages:
            status, body = _guard_transport(lambda: self.client.get_orders(
                self.account_id, self.market_type, order_category=category,
                page_index=page_index, page_size=100))
            if status != 200 or not isinstance(body, dict):
                # High-frequency poll -> DEBUG so a transient blip stays
                # inspectable without flooding the operator's log.
                log.broker_debug("read:orders[%s] p%s -> transient | http=%s code=%s",
                                 category, page_index, status, errors.code_of(body))
                return None, errors.classify(status, body, is_write=False)
            rows.extend(body.get("orders") or [])
            if page_index == 0:
                page_plan = book_page_count(body.get("totalPages"))
                if page_plan is None:
                    log.broker_debug("read:orders[%s] totalPages=%s -> unprovable",
                                     category, body.get("totalPages"))
                    return None, None
                pages = page_plan
            page_index += 1
        return rows, None

    def _iter_orders(self):
        """Yield raw order rows across NORMAL + conditional books (best-effort).

        A book whose completeness is unprovable yields NOTHING this cycle —
        the 0.5 s watch loop self-heals next poll (change-detector), so the
        tolerant semantics stay here; the strict ANY-book-unreadable raise
        lives in :meth:`get_open_orders`, the decision read (#62 G3')."""
        for category in _CATEGORIES:
            rows, _classified = self._read_book_rows_sync(category)
            if rows is None:
                continue
            yield from rows

    async def _drain_pending_oco(self):
        """Retry child-resolution for OCO umbrellas unresolved at PLACE time.

        Such an umbrella lives on a book ``_CATEGORIES`` never scans, so nothing
        row-driven ever retries it (#43) — this runs every ``watch_orders``
        cycle instead, on the same poll-counted cadence as the Activated-STOP
        adoption. Yields a terminal :class:`OrderEvent` for an umbrella that
        died childless, so the engine's exit intent is released rather than
        staying blind forever.
        """
        from pynecore.core.broker.models import OrderEvent
        for parent_id in list(self._pending_oco):
            pine_id, from_entry, leg_type = self._identity_for(parent_id)
            try:
                adoption, detail = await self._adopt_child(
                    parent_id, pine_id, from_entry, leg_type, category="OCO")
            except BrokerManualInterventionError:
                self._pending_oco.discard(parent_id)
                raise                # designed escalation: the engine halts
            except Exception as exc:                              # noqa: BLE001
                # One odd reply must never kill fill detection for every other
                # order — mirror the per-row guard in ``watch_orders``.
                log.broker_warning(
                    "pending-OCO drain raised for parent=%s: %s: %s — retrying",
                    parent_id, type(exc).__name__, exc)
                continue
            if adoption == "adopted":
                self._pending_oco.discard(parent_id)
            elif adoption == "dead":
                self._pending_oco.discard(parent_id)
                order = self._to_exchange_order(detail)
                event_type = ("cancelled" if order.status is OrderStatus.CANCELLED
                              else "rejected" if order.status is OrderStatus.REJECTED
                              else "filled" if order.status is OrderStatus.FILLED
                              else "cancelled")   # EXPIRED releases as cancelled
                yield OrderEvent(
                    order=order, event_type=event_type, fill_price=None,
                    fill_qty=None, timestamp=int(time.time()),
                    pine_id=pine_id, from_entry=from_entry, leg_type=leg_type)

    async def watch_orders(self):
        """Detect fills/cancels by polling the order books (REST, off-loop).

        #54 feed-health: a persistently failing poll must not leave the feed
        PERMANENTLY SILENT (measured: 18,936 DEBUG-only polls under a dead
        credential). Per-book consecutive-failure counters drive
        feed-attributed warnings (throttled by the ladder), and an ALL-books
        AUTH streak raises the DESIGNED halt — the engine latches
        ``BrokerManualInterventionError`` via ``_record_halt``; any other
        raise would kill the stream task with one log line and no restart.
        The poll is SINGLE-FLIGHT with a wait deadline: a hung socket read
        counts as stuck cycles and is re-awaited, never abandoned per cycle
        (the shared default executor must not fill with dead workers).
        """
        health = FeedHealth(
            warn_after=self._feed_warn_after,
            rewarn_every=self._feed_rewarn_every,
            halt_after=self._feed_halt_after,
            books=tuple(_CATEGORIES))
        poll_inflight = None
        while True:
            await asyncio.sleep(self._poll_interval)
            if self._pending_oco:
                try:
                    # Umbrellas queued at place time live on the unscanned OCO
                    # book — drain BEFORE the row scan so an adopted child's
                    # fill can surface in this same cycle (#43).
                    async for pending_event in self._drain_pending_oco():
                        yield pending_event
                    health.record_success("drain")
                except BrokerManualInterventionError:
                    raise            # designed escalation: the engine halts
                except Exception as exc:                          # noqa: BLE001
                    # G7: one drain failure must not kill the stream via the
                    # engine's terminate-on-raise supervisor; persistent
                    # failure escalates through the ladder instead.
                    health.record_failure("drain", type(exc).__name__)
            if poll_inflight is None:
                poll_inflight = asyncio.ensure_future(
                    asyncio.to_thread(self._poll_books_sync))
            try:
                rows, book_outcomes = await asyncio.wait_for(
                    asyncio.shield(poll_inflight), self._watch_read_deadline_s)
                poll_inflight = None
            except (asyncio.TimeoutError, TimeoutError):
                # Single-flight: the SAME read is re-awaited next cycle — a
                # hung socket costs stuck observations, never a growing stack
                # of abandoned worker threads.
                for book in _CATEGORIES:
                    health.record_failure(book, "stuck-read")
                self._emit_feed_warnings(health)
                continue
            except Exception as exc:                              # noqa: BLE001
                poll_inflight = None
                health.record_failure("poll", type(exc).__name__)
                self._emit_feed_warnings(health)
                continue
            health.record_success("poll")
            for book, outcome in book_outcomes.items():
                if outcome is None:
                    health.record_success(book)
                else:
                    failure_kind, failure_is_auth = outcome
                    health.record_failure(book, failure_kind,
                                          is_auth=failure_is_auth)
            self._emit_feed_warnings(health)
            halt_message = health.halt_due()
            if halt_message is not None:
                raise BrokerManualInterventionError(halt_message)
            scan_crash_kind = None
            for raw in rows:
                try:
                    event = await self._scan_row(raw)
                except BrokerManualInterventionError:
                    raise            # designed escalation: the engine halts
                except Exception as exc:                          # noqa: BLE001
                    # G7: a poisoned row must not kill fill detection for
                    # every other order. NOT marked seen -> retried next poll.
                    scan_crash_kind = type(exc).__name__
                    continue
                if event is not None:
                    yield event
            if scan_crash_kind is None:
                health.record_success("scan")
            else:
                health.record_failure("scan", scan_crash_kind)
            self._emit_feed_warnings(health)

    @staticmethod
    def _emit_feed_warnings(health: FeedHealth) -> None:
        for message in health.warnings_due():
            log.broker_warning("%s", message)

    def _poll_books_sync(self) -> "tuple[list[dict], dict[str, tuple | None]]":
        """One watch cycle's raw rows + per-book outcome (None = healthy;
        otherwise ``(kind, is_auth)`` for the feed-health ladder)."""
        rows: list[dict] = []
        outcomes: dict = {}
        for category in _CATEGORIES:
            book_rows, classified = self._read_book_rows_sync(category)
            if book_rows is None:
                if classified is not None:
                    outcomes[category] = (
                        f"{classified.code} http={classified.http_status}",
                        classified.disposition in _AUTH_DISPOSITIONS)
                else:
                    outcomes[category] = ("unprovable-pagination", False)
            else:
                rows.extend(book_rows)
                outcomes[category] = None
        return rows, outcomes

    async def _scan_row(self, raw: dict) -> "OrderEvent | None":
        """Process ONE polled row; return its OrderEvent or None.

        Extracted from the watch loop so each row runs under the G7 guard
        (#54) — the control flow is the loop body's, with ``continue``
        translated to ``return None``.
        """
        order_id = str(raw.get("id"))
        order = self._to_exchange_order(raw)
        cumulative = float(raw.get("fillQuantity") or 0)
        # Dedup on the RAW venue status, not the mapped OrderStatus: the
        # map collapses New and Activated to OPEN, which made a stop's
        # trigger transition invisible — the exact moment the child
        # normal-book order must be adopted (#39, measured live 08-18).
        raw_status = str(raw.get("orderStatus") or "")
        previous, prev_status = self._last_seen.get(order_id, (0.0, None))
        if cumulative == previous and raw_status == prev_status:
            return None
        pine_id, from_entry, leg_type = self._identity_for(order_id)
        if pine_id is None:
            # NOT marked seen: identity can arrive later (a stop's child
            # is adopted only at the parent's Activated transition — #39),
            # and a row marked seen pre-adoption would dedup its fill away.
            return None  # not ours (yet)
        if (raw_status.upper() == "ACTIVATED"
                and self._order_category.get(order_id) == "STOP"):
            # Two-book mechanic (CLAUDE.md): Activated = the conditional
            # CLOSED and a NEW order now works the NORMAL book. The fill
            # will arrive under the CHILD's id — adopt it into the
            # parent's identity so the scan's normal path reports it.
            try:
                adoption, _ = await self._adopt_child(
                    order_id, pine_id, from_entry, leg_type)
            except BrokerManualInterventionError:
                raise            # designed escalation: the engine halts
            except Exception as exc:                      # noqa: BLE001
                # One odd reply must never kill fill detection for every
                # other order (this loop has no other per-row guard).
                log.broker_warning(
                    "child adoption raised for parent=%s: %s: %s — retrying",
                    order_id, type(exc).__name__, exc)
                adoption = "pending"
            # "dead" is treated as pending here: an Activated shell stays
            # Activated forever (#41), and if a stale row read Activated
            # while the detail is already terminal, the row itself will
            # report the real status on a later poll via the normal path.
            if adoption != "adopted":
                # Deliberately NOT marked seen: the shell never changes
                # again, so this is the only thing that makes the next
                # poll retry (#42-A).
                return None
            self._last_seen[order_id] = (cumulative, raw_status)
            return None  # events come from the child row, not the shell
        # Any other status retires a pending adoption for this parent.
        self._adopt_attempts.pop(order_id, None)
        self._last_seen[order_id] = (cumulative, raw_status)
        if order.status in _TERMINAL_STATUSES:
            journal_terminal(self.store_ctx, venue_id=order_id,
                             terminal_status=raw_status, filled_qty=cumulative)
        delta = max(cumulative - previous, 0.0)
        event_type = ("filled" if order.status is OrderStatus.FILLED
                      else "partial" if order.status is OrderStatus.PARTIALLY_FILLED
                      else "cancelled" if order.status is OrderStatus.CANCELLED
                      else "rejected" if order.status is OrderStatus.REJECTED
                      else "created")
        return OrderEvent(
            order=order, event_type=event_type,
            fill_price=float(raw.get("averagePrice") or 0) or None,
            fill_qty=delta or None, timestamp=int(time.time()),
            pine_id=pine_id, from_entry=from_entry, leg_type=leg_type)

    @override
    async def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        """Union of the NORMAL + conditional books; a failed fetch never looks empty."""
        wanted = self.resolve_contract(symbol) if symbol else None
        orders: list[ExchangeOrder] = []
        for category in _CATEGORIES:
            try:
                rows, _classified = await asyncio.wait_for(
                    asyncio.to_thread(self._read_book_rows_sync, category),
                    timeout=self._book_read_deadline_s)
            except (asyncio.TimeoutError, TimeoutError):
                rows = None
            if rows is None:
                # #62 G3': ONE unreadable/unprovable book poisons the whole
                # answer — the old any_ok union returned a PARTIAL book set
                # as complete (false-clean `flat`, restart COID collision).
                log.broker_warning("%s", (
                    f"read:orders[{category}] unreadable/unprovable -> "
                    f"refusing a partial book union"))
                raise ExchangeConnectionError(f"DNSE {category} book unreadable")
            for raw in rows:
                if wanted and raw.get("symbol") != wanted:
                    continue
                order = self._to_exchange_order(raw)
                if order.status not in _TERMINAL_STATUSES:
                    orders.append(order)
        return orders

    @override
    async def get_position(self, symbol: str) -> ExchangePosition | None:
        """Net position for ``symbol`` from ``/positions`` (netting venue)."""
        try:
            status, body = await asyncio.wait_for(
                asyncio.to_thread(lambda: _guard_transport(
                    lambda: self.client.get_positions(
                        self.account_id, self.market_type, POSITIONS_PAGE_SIZE))),
                timeout=self._book_read_deadline_s)
        except (asyncio.TimeoutError, TimeoutError):
            raise ExchangeConnectionError("DNSE positions read timed out")
        if status != 200 or not isinstance(body, dict):
            classified = errors.classify(status, body, is_write=False)
            if classified is not None:
                self._emit(classified, action="read:positions", ident=symbol)
            raise ExchangeConnectionError(f"DNSE positions unavailable: {status}")
        raw_rows = body.get("positions") or body.get("data") or []
        # #57/#62 G1/G2b: judged on the RAW row count, before the CLOSED
        # filter — a truncated page must NEVER look FLAT (None arms the
        # engine's external-flatten wipe). Absent ``total`` (STOCK) never
        # infers truncation.
        if not positions_complete(len(raw_rows), body.get("total")):
            raise ExchangeConnectionError(
                f"DNSE positions page truncated: {len(raw_rows)} rows delivered "
                f"but total={body.get('total')} — refusing to conclude")
        wanted = self.resolve_contract(symbol)
        net, cost = 0.0, 0.0
        for row in raw_rows:
            if row.get("symbol") != wanted or not is_exposure_row(row):
                continue
            size = float(row.get("openQuantity") or row.get("quantity") or 0)
            signed = size if str(row.get("side", "")).upper() in ("NB", "LONG") else -size
            net += signed
            cost += abs(signed) * float(row.get("costPrice")
                                        or row.get("averagePrice") or row.get("price") or 0)
        if net == 0:
            return None
        volume = abs(net)
        # "long"/"short" is the ExchangePosition contract (models.py:322) and the
        # ONLY vocabulary the engine decodes — "buy"/"sell" silently disabled
        # startup size adoption and could halt a defensive-close settle (#49).
        return ExchangePosition(
            symbol=symbol, side="long" if net > 0 else "short", size=volume,
            entry_price=(cost / volume) if volume else 0.0,
            unrealized_pnl=0.0, liquidation_price=None,
            leverage=1.0, margin_mode="cross")

    @override
    async def get_balance(self) -> dict[str, float]:
        status, body = self.client.get_balances(self.account_id)
        if status != 200 or not isinstance(body, dict):
            return {}
        derivative = body.get("derivative") or {}
        stock = body.get("stock") or {}
        return {"VND": float(derivative.get("remainSecure")
                             or stock.get("availableCash") or 0)}
