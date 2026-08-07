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

from .provider import DNSEConfig, DNSEProvider

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

#: Order-book categories to poll/scan. NORMAL carries fills (an activated
#: conditional becomes a NORMAL order); STOP/OCO carry the conditional lifecycle.
_CATEGORIES = ("NORMAL", "STOP", "OCO")


@dataclass
class DNSEBrokerConfig(DNSEConfig):
    """:ivar account_no: DNSE trading account. Empty = resolve via ``/accounts``.
    :ivar trading_token: Bootstrap token; used only if the state file is absent.
    :ivar token_file: State file written by the OTP minter (the live source).
    """

    account_no: str = ""
    trading_token: str = ""
    token_file: str = "workdir/state/dnse_trading_token.json"


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
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError(f"cannot resolve account: {status} {body}")
        self._account_no = body["accounts"][0]["id"]
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
            # cancellable). Used for entry-stops and the SL leg of a bracket.
            stop_order=CapabilityLevel.NATIVE,
            # Bracket = a NORMAL TP leg + a native STOP SL leg, run as two orders;
            # the ENGINE drives the one-cancels-other. NOT native OCO (which
            # activates within ~1s and then can't be cancelled/modified).
            tp_sl_bracket=CapabilityLevel.SOFTWARE,
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
        row = self._secdef(self.symbol or "")
        ceiling = float(row.get("ceilingPrice") or 0)
        floor = float(row.get("floorPrice") or 0)
        if not ceiling or not floor:
            raise RuntimeError(
                f"cannot price a market {side}: secdef has no ceiling/floor for "
                f"{self.symbol!r}")
        return ceiling if side == "buy" else floor

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

    def _place(self, envelope, side: str, qty: float, *, price: float,
               category: str = "NORMAL", stop_price: float | None = None,
               stop_order_price: float | None = None, leg_type=None
               ) -> list[ExchangeOrder]:
        """Place one native order (NORMAL / STOP / OCO) and record its identity."""
        from pynecore.core.broker.exceptions import (
            ExchangeOrderRejectedError, OrderDispositionUnknownError)

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

        status, body = self.client.post_order(
            self.account_id, self.market_type, payload, self._token(),
            order_category=category)
        if status == 0:
            raise OrderDispositionUnknownError(
                f"DNSE order disposition unknown: {body}",
                client_order_id=envelope.client_order_id())
        if status not in (200, 201) or not isinstance(body, dict):
            raise ExchangeOrderRejectedError(f"DNSE rejected order: {status} {body}")

        order = self._to_exchange_order(body)
        key = getattr(envelope.intent, "intent_key", None)
        if key:
            self._order_ids.setdefault(key, []).append(order.id)
        intent = envelope.intent
        self._identity[order.id] = (
            getattr(intent, "pine_id", None),
            getattr(intent, "from_entry", None),
            leg_type,
        )
        return [order]

    # --- BrokerPlugin abstracts: execution ---

    @override
    async def execute_entry(self, envelope) -> list[ExchangeOrder]:
        from pynecore.core.broker.models import LegType
        intent = envelope.intent
        if intent.stop is not None:
            # stop or stop-limit entry -> native STOP (price = limit if given)
            return self._place(envelope, intent.side, intent.qty,
                               price=intent.limit or intent.stop, category="STOP",
                               stop_price=intent.stop, leg_type=LegType.ENTRY)
        if intent.limit is not None:
            return self._place(envelope, intent.side, intent.qty,
                               price=intent.limit, leg_type=LegType.ENTRY)
        return self._place(envelope, intent.side, intent.qty,
                           price=self._marketable_price(intent.side),
                           leg_type=LegType.ENTRY)

    @override
    async def execute_exit(self, envelope) -> list[ExchangeOrder]:
        """Bracket exit as SEPARATE legs (engine-managed OCA), NOT native OCO.

        Native OCO proved unsuitable for plugin brackets (verified live 2026-08-07):
        it activates within ~1s of placement (its NORMAL TP leg rests immediately),
        after which it can't be cancelled or modified (400 "order status is not
        new"), and cancelling it does NOT cascade to the TP leg. So TP is a NORMAL
        LO and SL is a native STOP — both cleanly cancellable/amendable — and the
        engine runs the one-cancels-other (tp_sl_bracket / oca_cancel = SOFTWARE).
        """
        from pynecore.core.broker.models import LegType
        from pynecore.core.broker.exceptions import OrderSkippedByPlugin
        intent = envelope.intent
        orders: list[ExchangeOrder] = []
        if intent.tp_price is not None:
            orders += self._place(envelope, intent.side, intent.qty,
                                  price=intent.tp_price, leg_type=LegType.TAKE_PROFIT)
        if intent.sl_price is not None:
            orders += self._place(envelope, intent.side, intent.qty,
                                  price=intent.sl_price, category="STOP",
                                  stop_price=intent.sl_price, leg_type=LegType.STOP_LOSS)
        if not orders:
            raise OrderSkippedByPlugin(
                "DNSE plugin cannot express this exit: no tp_price/sl_price "
                "(trailing stops are not implemented)")
        return orders

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
        """Category a placed order was recorded under (leg_type -> STOP/OCO/NORMAL)."""
        _, _, leg_type = self._identity_for(order_id)
        name = getattr(leg_type, "name", "")
        if name == "STOP_LOSS":
            return "STOP"
        if name == "ENTRY":
            return None  # could be STOP or NORMAL — cancel tries both
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
        """Cancel an order, trying conditional categories then NORMAL."""
        hinted = self._order_category_for(order_id)
        for category in ([hinted] if hinted else _CATEGORIES):
            status, body = self.client.cancel_order(
                self.account_id, order_id, self.market_type, self._token(),
                order_category=category)
            if status in (200, 204):
                return True
            code = (body or {}).get("code") if isinstance(body, dict) else None
            if code == "RESOURCE_NOT_FOUND":
                return True  # already gone (filled/cancelled)
        return False

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
        status, body = self.client.put_order(
            self.account_id, order_id, self.market_type, payload, self._token(),
            order_category=category)
        if status not in (200, 201) or not isinstance(body, dict):
            from pynecore.core.broker.exceptions import ExchangeOrderRejectedError
            raise ExchangeOrderRejectedError(f"DNSE amend failed: {status} {body}")
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
        from pynecore.core.broker.exceptions import ExchangeConnectionError
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
            raise ExchangeConnectionError("DNSE order books unavailable")
        return orders

    @override
    async def get_position(self, symbol: str) -> ExchangePosition | None:
        """Net position for ``symbol`` from ``/positions`` (netting venue)."""
        status, body = self.client.get_positions(self.account_id, self.market_type)
        if status != 200 or not isinstance(body, dict):
            from pynecore.core.broker.exceptions import ExchangeConnectionError
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
