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
    CapabilityLevel, ExchangeCapabilities, ExchangeOrder, ExchangePosition,
    OrderStatus, OrderType,
)
from pynecore.types.ohlcv import OHLCV
from pynecore.lib import log
from pynecore.core.broker.exceptions import (
    AuthenticationError, ExchangeConnectionError, ExchangeOrderRejectedError,
    ExchangeRateLimitError, InsufficientMarginError, OrderDispositionUnknownError,
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

#: TradingView timeframe -> seconds (bar-period math for the closed-bar poll).
_TF_SECONDS = {"1": 60, "3": 180, "5": 300, "15": 900, "30": 1800,
               "60": 3600, "1H": 3600, "1D": 86400}

#: Order-book categories to poll/scan for WORKING orders + fills. An OCO's real
#: working order is the spawned NORMAL LO (tracked via its externalOrderId), and a
#: STOP is its own working order — so we scan NORMAL + STOP and skip the OCO
#: umbrella book (whose records would double-count the LO / linger as zombies).
_CATEGORIES = ("NORMAL", "STOP")

#: LegType.name -> idempotency KIND, for the disposition-unknown client_order_id.
_LEG_KIND = {
    "ENTRY": KIND_ENTRY, "TAKE_PROFIT": KIND_EXIT_TP,
    "STOP_LOSS": KIND_EXIT_SL, "CLOSE": KIND_CLOSE,
}


@dataclass
class DNSEBrokerConfig(DNSEConfig):
    """:ivar account_no: DNSE trading account. Empty = resolve via ``/accounts``.
    :ivar trading_token: Bootstrap token; used only if the state file is absent.
    :ivar token_file: State file written by the OTP minter (the live source).
    :ivar stop_slippage_ticks: Fallback offset (in ticks) applied *through* a stop's
        trigger when pricing the LO it emits, used only when the strategy declares no
        ``strategy(slippage=)``. DNSE has no stop-market order, so a triggered stop
        posts a limit; pricing it at the trigger means a gap through never fills. The
        strategy's own ``slippage x 2`` takes precedence; this is the floor so the
        Pine default of 0 cannot silently recreate a never-filling stop.
    """

    account_no: str = ""
    trading_token: str = ""
    token_file: str = "workdir/state/dnse_trading_token.json"
    stop_slippage_ticks: int = 10


class DNSEBroker(DNSEProvider, BrokerPlugin[DNSEBrokerConfig]):
    """DNSE broker: Vietnam derivatives (native STOP/OCO) and stocks."""

    plugin_name = "DNSE Broker"
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
        self._poll_interval: float = 2.0
        self._bar_poll_interval: float = 3.0
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
            oca_cancel=CapabilityLevel.NATIVE,
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
        # REST-only: nothing to connect. Touch the client so the endpoint banner
        # is logged and credentials are validated up front.
        _ = self.client
        self._connected = True

    @override
    async def disconnect(self) -> None:
        self._connected = False

    @property
    @override
    def is_connected(self) -> bool:
        return self._connected

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        """Yield the next CLOSED bar by polling REST ``/price/ohlc``."""
        resolution = self.to_exchange_timeframe(timeframe)
        period = _TF_SECONDS.get(timeframe, 300)
        while True:
            now = int(time.time())
            status, body = await asyncio.to_thread(
                self.client.get_ohlc, self.market_type,
                {"symbol": self.symbol, "resolution": resolution,
                 "from": now - period * 5, "to": now})
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

    @staticmethod
    def _gtd(days: int = 7) -> str:
        """A far-future RFC3339 expiry for a GTD STOP (engine cancels earlier)."""
        return (datetime.now(timezone.utc) + timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")

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
        """``call(token) -> (status, body)``; retry ONCE on INVALID_TRADING_TOKEN with
        a freshly re-read token (the cron may have refreshed the state file)."""
        status, body = call(self._token())
        if errors.code_of(body) == "INVALID_TRADING_TOKEN":
            log.broker_warning("%s", "write code=INVALID_TRADING_TOKEN -> "
                                     "token-reread (retry once)")
            status, body = call(self._token())
        return status, body

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
        status, body = self._write(lambda tok: self.client.post_order(
            self.account_id, self.market_type, payload, tok, order_category=category))
        self._raise_write_error(status, body, action="place", ident=ident,
                                coid=self._coid(envelope, leg_type))
        if not isinstance(body, dict):
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
        key = getattr(envelope.intent, "intent_key", None)
        if key:
            self._order_ids.setdefault(key, []).append(order.id)
        intent = envelope.intent
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

    def _resolve_oco_lo(self, oco_id: str, attempts: int = 6, delay: float = 0.15
                        ) -> "ExchangeOrder | None":
        """Return the OCO's working NORMAL LO as an ``ExchangeOrder`` (or None).

        The OCO spawns a NORMAL LO on activation (~instant); the OCO *detail*'s
        ``externalOrderId`` names it (the list view omits it), and the LO's own
        ``metadata.conditionOrderId`` points back. Poll briefly for activation,
        then fetch the LO. (Synchronous; the brief poll blocks the caller.)
        """
        for _ in range(attempts):
            status, body = self.client.get_order_detail(
                self.account_id, oco_id, self.market_type, order_category="OCO")
            external = body.get("externalOrderId") if isinstance(body, dict) else None
            if external:
                lo_id = str(external)
                _, detail = self.client.get_order_detail(
                    self.account_id, lo_id, self.market_type, order_category="NORMAL")
                if isinstance(detail, dict):
                    return self._to_exchange_order(detail)
                return self._to_exchange_order({"id": lo_id})
            time.sleep(delay)
        return None

    # --- BrokerPlugin abstracts: execution ---

    @override
    async def execute_entry(self, envelope) -> list[ExchangeOrder]:
        from pynecore.core.broker.models import LegType
        intent = envelope.intent
        if intent.stop is not None:
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

    @override
    def _identity_for(self, order_id: str) -> tuple:
        return self._identity.get(order_id, (None, None, None))

    def _ids_for(self, envelope) -> list[str]:
        key = getattr(envelope.intent, "intent_key", None)
        return list(self._order_ids.get(key, [])) if key else []

    def _order_category_for(self, order_id: str):
        """The book a placed order lives in, recorded at place time (authoritative)."""
        recorded = self._order_category.get(order_id)
        if recorded:
            return recorded
        # Fallback for ids with no record (e.g. a restart before re-hydration):
        # infer from the leg, and leave ENTRY unknown so cancel probes every book.
        _, _, leg_type = self._identity_for(order_id)
        name = getattr(leg_type, "name", "")
        if name == "STOP_LOSS":
            return "STOP"
        if name == "ENTRY":
            return None
        return "NORMAL"

    @override
    async def execute_cancel(self, envelope) -> bool:
        ids = self._ids_for(envelope)
        if not ids:
            return False
        ok = True
        for order_id in ids:
            ok = self._cancel_one(str(order_id)) and ok
        return ok

    def _cancel_one(self, order_id: str) -> bool:
        """Cancel by the recorded book; if unknown, probe each.

        A ``RESOURCE_NOT_FOUND`` from the WRONG book must NOT count as success — that
        silently orphaned STOP entries (a STOP id is absent from the NORMAL book, so
        the NORMAL-first attempt 404'd and short-circuited before ever trying STOP).
        Only conclude "already gone" if a cancel succeeded, the order is already
        terminal (``ORDER_IS_DONE`` / ``CO-ORD-013``), or EVERY book agrees it is not
        there. Any other failure (session-refused, transient, reject) is logged.
        """
        hinted = self._order_category_for(order_id)
        categories = [hinted] if hinted else list(_CATEGORIES)
        all_not_found = True
        for category in categories:
            status, body = self._write(lambda tok: self.client.cancel_order(
                self.account_id, order_id, self.market_type, tok,
                order_category=category))
            if status in (200, 204):
                return True
            code = errors.code_of(body)
            if code in errors.TERMINAL_CODES:  # found, already done -> cancel is moot
                log.broker_info("%s", f"cancel[{category}] code={code} http={status} -> "
                                      f"treated-gone (already terminal) | order={order_id}")
                return True
            if code in errors.NOT_FOUND_CODES:  # not in THIS book -> probe the next
                continue
            classified = errors.classify(status, body, is_write=True)
            if classified is not None:
                self._emit(classified, action=f"cancel[{category}]", ident=order_id)
            all_not_found = False
        return all_not_found

    @override
    async def execute_cancel_with_outcome(self, envelope):
        from pynecore.core.broker.models import CancelDispositionOutcome
        ids = self._ids_for(envelope)
        if not ids:
            return CancelDispositionOutcome.UNKNOWN
        return (CancelDispositionOutcome.CANCEL_CONFIRMED
                if self._cancel_one(str(ids[0]))
                else CancelDispositionOutcome.UNKNOWN)

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

    def _iter_orders(self):
        """Yield raw order rows across NORMAL + conditional books (best-effort)."""
        for category in _CATEGORIES:
            status, body = self.client.get_orders(
                self.account_id, self.market_type, order_category=category,
                page_index=0, page_size=100)
            if status == 200 and isinstance(body, dict):
                yield from body.get("orders") or []
            else:
                # High-frequency poll -> DEBUG so a transient blip stays inspectable
                # without flooding the operator's log.
                log.broker_debug("read:orders[%s] -> transient | http=%s code=%s",
                                 category, status, errors.code_of(body))

    @override
    async def watch_orders(self):
        """Detect fills/cancels by polling the order books (REST, off-loop)."""
        from pynecore.core.broker.models import OrderEvent
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                rows = await asyncio.to_thread(lambda: list(self._iter_orders()))
            except Exception:
                continue
            for raw in rows:
                order_id = str(raw.get("id"))
                order = self._to_exchange_order(raw)
                cumulative = float(raw.get("fillQuantity") or 0)
                previous, prev_status = self._last_seen.get(order_id, (0.0, None))
                if cumulative == previous and order.status is prev_status:
                    continue
                self._last_seen[order_id] = (cumulative, order.status)
                pine_id, from_entry, leg_type = self._identity_for(order_id)
                if pine_id is None:
                    continue  # not ours
                delta = max(cumulative - previous, 0.0)
                event_type = ("filled" if order.status is OrderStatus.FILLED
                              else "partial" if order.status is OrderStatus.PARTIALLY_FILLED
                              else "cancelled" if order.status is OrderStatus.CANCELLED
                              else "rejected" if order.status is OrderStatus.REJECTED
                              else "created")
                yield OrderEvent(
                    order=order, event_type=event_type,
                    fill_price=float(raw.get("averagePrice") or 0) or None,
                    fill_qty=delta or None, timestamp=int(time.time()),
                    pine_id=pine_id, from_entry=from_entry, leg_type=leg_type)

    @override
    async def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        """Union of the NORMAL + conditional books; a failed fetch never looks empty."""
        wanted = self.resolve_contract(symbol) if symbol else None
        orders: list[ExchangeOrder] = []
        any_ok = False
        for category in _CATEGORIES:
            status, body = self.client.get_orders(
                self.account_id, self.market_type, order_category=category,
                page_index=0, page_size=100)
            if status != 200 or not isinstance(body, dict):
                continue
            any_ok = True
            for raw in body.get("orders") or []:
                if wanted and raw.get("symbol") != wanted:
                    continue
                order = self._to_exchange_order(raw)
                if order.status not in _TERMINAL_STATUSES:
                    orders.append(order)
        if not any_ok:
            log.broker_warning("%s", "read:orders -> reconnect | both order books unavailable")
            raise ExchangeConnectionError("DNSE order books unavailable")
        return orders

    @override
    async def get_position(self, symbol: str) -> ExchangePosition | None:
        """Net position for ``symbol`` from ``/positions`` (netting venue)."""
        status, body = self.client.get_positions(self.account_id, self.market_type)
        if status != 200 or not isinstance(body, dict):
            classified = errors.classify(status, body, is_write=False)
            if classified is not None:
                self._emit(classified, action="read:positions", ident=symbol)
            raise ExchangeConnectionError(f"DNSE positions unavailable: {status}")
        wanted = self.resolve_contract(symbol)
        net, cost = 0.0, 0.0
        for row in body.get("positions") or body.get("data") or []:
            if row.get("symbol") != wanted:
                continue
            size = float(row.get("openQuantity") or row.get("quantity") or 0)
            signed = size if str(row.get("side", "")).upper() in ("NB", "LONG") else -size
            net += signed
            cost += abs(signed) * float(row.get("costPrice")
                                        or row.get("averagePrice") or row.get("price") or 0)
        if net == 0:
            return None
        volume = abs(net)
        return ExchangePosition(
            symbol=symbol, side="buy" if net > 0 else "sell", size=volume,
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
