"""DNSE broker plugin — order execution for PyneCore.

Builds on :class:`DNSEProvider` (history) and adds live streaming plus the
eight ``BrokerPlugin`` abstracts.

Design decisions, grounded against the live API and recorded on card #8:

* ``position_port = None`` — DNSE derivatives are netted per symbol, so the
  core one-way emulator is not wired. ``get_position`` aggregates DNSE's
  per-deal rows itself.
* ``idempotency = SOFTWARE`` — the place-order payload accepts NO
  client-supplied order id, so exactly-once by construction is unavailable.
* ``tp_sl_bracket`` is never ``NATIVE``: any DNSE order can partially fill,
  and ``NATIVE`` suppresses the engine's partial-fill bracket-amend path.
* Order events are STREAMED (``order.{marketType}.{encoding}``), not polled.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import uuid4

from pynecore.core.plugin import override
from pynecore.core.plugin.broker import BrokerPlugin
from pynecore.core.broker.models import (
    CapabilityLevel, ExchangeCapabilities, ExchangeOrder, ExchangePosition,
    OrderStatus, OrderType,
)
from pynecore.types.ohlcv import OHLCV

from .provider import DNSEConfig, DNSEProvider

#: DNSE side codes.
_SIDE_TO_DNSE = {"buy": "NB", "sell": "NS"}
_DNSE_TO_SIDE = {"NB": "buy", "NS": "sell"}

#: DNSE order status -> PyneCore OrderStatus (superset; unknown maps to OPEN).
_STATUS_MAP = {
    "PENDING": OrderStatus.PENDING, "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED, "FILLED": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED, "CANCELED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED, "EXPIRED": OrderStatus.EXPIRED,
}

#: Both DNSE order books. ``get_open_orders`` unions these; a book whose fetch
#: FAILS must be reported as absent, never as empty.
_ORDER_CATEGORIES = ("NORMAL", "CONDITIONAL")


@dataclass
class DNSEBrokerConfig(DNSEConfig):
    """:ivar account_no: DNSE trading account. Empty = resolve via /accounts.
    :ivar trading_token: Short-lived OTP-derived token for order operations.
    """

    account_no: str = ""
    trading_token: str = ""


class DNSEBroker(DNSEProvider, BrokerPlugin[DNSEBrokerConfig]):
    """DNSE broker: Vietnam derivatives and stocks."""

    plugin_name = "DNSE Broker"
    Config = DNSEBrokerConfig

    #: Netting-native venue — no hedged-leg emulation. See card #8.
    position_port = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._connected = False
        self._ws_session = None
        self._ws = None
        self._bar_queue: asyncio.Queue[OHLCV] = asyncio.Queue()
        self._account_no: str | None = None

    # --- account ---

    @property
    def account_id(self) -> str:
        if self._account_no:
            return self._account_no
        assert self.config is not None
        if self.config.account_no:
            self._account_no = self.config.account_no
            return self._account_no
        status, body = self.client.get_accounts()
        if status != 200:
            raise RuntimeError(f"cannot resolve account: {status} {body}")
        self._account_no = body["accounts"][0]["id"]
        return self._account_no

    def _token(self) -> str:
        assert self.config is not None
        if not self.config.trading_token:
            raise RuntimeError(
                "no trading_token — mint one with send_email_otp() + "
                "create_trading_token(otp_type, passcode) and set it in "
                "workdir/config/plugins/dnse_broker.toml"
            )
        return self.config.trading_token

    # --- capabilities ---

    @override
    def get_capabilities(self) -> ExchangeCapabilities:
        return ExchangeCapabilities(
            # Conditional book exists but its exact orderCategory value is
            # unconfirmed; declare SOFTWARE so the engine keeps its own watch.
            stop_order=CapabilityLevel.SOFTWARE,
            trailing_stop=CapabilityLevel.SOFTWARE,
            # NEVER native: any order can partially fill, and NATIVE would
            # suppress the engine's partial-fill bracket-amend path.
            tp_sl_bracket=CapabilityLevel.SOFTWARE,
            partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
            partial_qty_bracket_exit_pyramiding=CapabilityLevel.SOFTWARE,
            oca_cancel=CapabilityLevel.SOFTWARE,
            # PUT /orders/{id} amends price+quantity atomically.
            amend_order=CapabilityLevel.NATIVE,
            cancel_all=CapabilityLevel.SOFTWARE,
            reduce_only=CapabilityLevel.SOFTWARE,
            # order.{marketType}.{encoding} private stream.
            watch_orders=CapabilityLevel.NATIVE,
            fetch_position=CapabilityLevel.NATIVE,
            # No client-supplied order id in the place payload.
            idempotency=CapabilityLevel.SOFTWARE,
            short_selling=CapabilityLevel.NATIVE,
        )

    # --- live data (LiveProviderPlugin) ---

    @override
    async def connect(self) -> None:
        import aiohttp
        assert self.config is not None
        origin = self.config.ws_url.rstrip("/")
        url = f"{origin}/v1/stream?encoding=json"
        self._ws_session = aiohttp.ClientSession()
        self._ws = await self._ws_session.ws_connect(url, heartbeat=25)

        timestamp = int(time.time())
        nonce = str(int(time.time() * 1_000_000))
        message = f"{self.config.api_key}:{timestamp}:{nonce}"
        signature = hmac.new(self.config.api_secret.encode(),
                             message.encode(), hashlib.sha256).hexdigest()
        await self._ws.send_json({
            "action": "auth", "api_key": self.config.api_key,
            "signature": signature, "timestamp": timestamp, "nonce": nonce,
        })
        self._connected = True

    @override
    async def disconnect(self) -> None:
        self._connected = False
        if self._ws is not None:
            await self._ws.close()
        if self._ws_session is not None:
            await self._ws_session.close()
        self._ws, self._ws_session = None, None

    @property
    @override
    def is_connected(self) -> bool:
        return self._connected

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        """Yield one bar from the ohlc_closed channel.

        Subscribes lazily on first call. The 8-hour server-side connection cap
        surfaces as a closed socket; the framework's reconnect path re-runs
        ``connect()``, which mints a fresh timestamp/nonce as DNSE requires.
        """
        assert self._ws is not None, "connect() first"
        resolution = self.to_exchange_timeframe(timeframe)
        if not getattr(self, "_subscribed", False):
            await self._ws.send_json({
                "action": "subscribe",
                "channels": [{"name": f"ohlc_closed.{resolution}.json",
                              "symbols": [symbol.upper()]}],
            })
            self._subscribed = True

        while True:
            msg = await self._ws.receive_json()
            if msg.get("action") == "ping":
                await self._ws.send_json({"action": "pong",
                                          "timestamp": msg.get("timestamp")})
                continue
            data = msg.get("data") or {}
            if "o" not in data:
                continue
            return OHLCV(
                timestamp=int(data["t"]) * 1000,
                open=float(data["o"]), high=float(data["h"]),
                low=float(data["l"]), close=float(data["c"]),
                volume=float(data.get("v", 0)), is_closed=True,
            )

    # --- order mapping ---

    def _to_exchange_order(self, raw: dict) -> ExchangeOrder:
        filled = float(raw.get("fillQuantity") or 0)
        qty = float(raw.get("quantity") or 0)
        return ExchangeOrder(
            id=str(raw.get("id")),
            symbol=raw.get("symbol") or self.symbol or "",
            side=_DNSE_TO_SIDE.get(raw.get("side", ""), "buy"),
            order_type=OrderType.LIMIT,
            qty=qty, filled_qty=filled,
            remaining_qty=float(raw.get("leaveQuantity") or max(qty - filled, 0)),
            price=float(raw.get("price") or 0) or None,
            stop_price=None,
            average_fill_price=float(raw.get("averagePrice") or 0) or None,
            status=_STATUS_MAP.get(str(raw.get("orderStatus", "")).upper(),
                                   OrderStatus.OPEN),
            timestamp=int(time.time() * 1000),
            fee=0.0, fee_currency="VND",
        )

    def _place(self, side: str, qty: float, price: float | None,
               category: str = "NORMAL") -> list[ExchangeOrder]:
        payload = {
            "accountNo": self.account_id,
            # Orders need the tradable KRX contract, not the symbolType alias.
            "symbol": self.resolve_contract(),
            "side": _SIDE_TO_DNSE[side],
            # LO only: this plugin never sends an unpriced market order.
            "orderType": "LO",
            "price": round(float(price), 1) if price else 0.0,
            "quantity": int(qty),
            "loanPackageId": self._loan_package_id(),
        }
        status, body = self.client.post_order(self.market_type, payload,
                                              self._token(), order_category=category)
        if status not in (200, 201) or not isinstance(body, dict):
            from pynecore.core.broker.exceptions import ExchangeOrderRejectedError
            raise ExchangeOrderRejectedError(f"DNSE rejected order: {status} {body}")
        return [self._to_exchange_order(body)]

    def _loan_package_id(self) -> int:
        if getattr(self, "_loan_id", None) is None:
            status, body = self.client.get_loan_packages(self.account_id, self.market_type)
            if status != 200 or not body.get("loanPackages"):
                raise RuntimeError(f"cannot resolve loanPackageId: {status} {body}")
            self._loan_id = body["loanPackages"][0]["id"]
        return self._loan_id

    # --- BrokerPlugin abstracts ---

    @override
    async def execute_entry(self, envelope) -> list[ExchangeOrder]:
        intent = envelope.intent
        return self._place(intent.side, intent.qty, intent.limit or intent.stop)

    @override
    async def execute_exit(self, envelope) -> list[ExchangeOrder]:
        intent = envelope.intent
        orders: list[ExchangeOrder] = []
        for price in (intent.tp_price, intent.sl_price):
            if price is not None:
                orders += self._place(intent.side, intent.qty, price,
                                      category="CONDITIONAL")
        return orders

    @override
    async def execute_close(self, envelope) -> ExchangeOrder:
        intent = envelope.intent
        return self._place(intent.side, intent.qty, None)[0]

    @override
    async def execute_cancel(self, envelope) -> bool:
        order_id = getattr(envelope.intent, "exchange_order_id", None)
        if not order_id:
            return False
        status, _ = self.client.cancel_order(self.account_id, str(order_id),
                                             self.market_type, self._token())
        return status in (200, 204)

    @override
    async def modify_entry(self, old, new) -> list[ExchangeOrder]:
        """In-place amend via ``PUT /accounts/{acct}/orders/{id}``.

        Overrides the inherited cancel-then-recreate default, which would open
        a window where the order is absent — the ``amend_order=NATIVE``
        declaration promises there is no such window.
        """
        return await self._amend(old, new)

    @override
    async def modify_exit(self, old, new) -> list[ExchangeOrder]:
        """Atomic amend of a protective leg. The default cancel+recreate
        would leave the position briefly unprotected."""
        return await self._amend(old, new)

    async def _amend(self, old, new) -> list[ExchangeOrder]:
        order_id = getattr(old.intent, "exchange_order_id", None) or \
            getattr(old, "exchange_order_id", None)
        if not order_id:
            # Nothing to amend in place — fall back to the base behaviour.
            return await super().modify_entry(old, new)
        intent = new.intent
        price = getattr(intent, "limit", None) or getattr(intent, "stop", None) \
            or getattr(intent, "tp_price", None) or getattr(intent, "sl_price", None)
        payload = {"price": round(float(price), 1) if price else 0.0,
                   "quantity": int(intent.qty)}
        status, body = self.client.request(
            "PUT", f"/accounts/{self.account_id}/orders/{order_id}",
            query={"marketType": self.market_type, "orderCategory": "NORMAL"},
            body=payload, headers={"trading-token": self._token()})
        if status not in (200, 201) or not isinstance(body, dict):
            from pynecore.core.broker.exceptions import ExchangeOrderRejectedError
            raise ExchangeOrderRejectedError(f"DNSE amend failed: {status} {body}")
        return [self._to_exchange_order(body)]

    @override
    async def execute_cancel_with_outcome(self, envelope):
        """Classify DNSE's cancel response instead of collapsing to UNKNOWN.

        Without this the cancel-tentative state machine can only resolve
        through a broker-pushed FILL/CANCEL event.
        """
        from pynecore.core.broker.models import CancelDispositionOutcome
        order_id = getattr(envelope.intent, "exchange_order_id", None)
        if not order_id:
            return CancelDispositionOutcome.UNKNOWN
        status, body = self.client.cancel_order(
            self.account_id, str(order_id), self.market_type, self._token())
        if status in (200, 204):
            return CancelDispositionOutcome.CANCEL_CONFIRMED
        code = (body or {}).get("code", "") if isinstance(body, dict) else ""
        if code == "RESOURCE_NOT_FOUND":
            # The order is gone from the repository — it either filled or was
            # already cancelled. The engine resolves which via the event stream.
            return CancelDispositionOutcome.UNKNOWN
        if code == "CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION":
            return CancelDispositionOutcome.STILL_OPEN
        return CancelDispositionOutcome.UNKNOWN

    @override
    async def watch_orders(self):
        """Stream order updates from ``order.{marketType}.{encoding}``.

        DNSE pushes order events, so the engine never has to poll
        ``get_open_orders`` for fills. ``fillQuantity`` is CUMULATIVE, which is
        exactly what the tracker wants (it clamps deltas into
        ``[row.filled_qty, row.qty]``).
        """
        from pynecore.core.broker.models import OrderEvent
        assert self._ws is not None, "connect() first"
        await self._ws.send_json({
            "action": "subscribe",
            "channels": [{"name": f"order.{self.market_type}.json", "symbols": []}],
        })
        while True:
            msg = await self._ws.receive_json()
            if msg.get("action") == "ping":
                await self._ws.send_json({"action": "pong",
                                          "timestamp": msg.get("timestamp")})
                continue
            raw = msg.get("data") or {}
            if "orderStatus" not in raw:
                continue
            order = self._to_exchange_order(raw)
            filled = float(raw.get("fillQuantity") or 0)
            status = str(raw.get("orderStatus", "")).upper()
            event_type = ("fill" if status in ("FILLED", "PARTIALLY_FILLED")
                          else "cancelled" if status in ("CANCELLED", "CANCELED")
                          else "rejected" if status == "REJECTED" else "open")
            yield OrderEvent(
                order=order, event_type=event_type,
                fill_price=float(raw.get("averagePrice") or 0) or None,
                fill_qty=filled or None,
                timestamp=int(time.time() * 1000),
            )

    @override
    async def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        """Union of BOTH order books.

        A book whose fetch FAILS is skipped rather than reported empty — an
        incomplete snapshot must never look like a complete absence.
        """
        orders: list[ExchangeOrder] = []
        for category in _ORDER_CATEGORIES:
            status, body = self.client.get_orders(self.account_id, self.market_type,
                                                  category)
            if status != 200 or not isinstance(body, dict):
                continue  # incomplete: do NOT treat as "no orders"
            for raw in body.get("orders") or []:
                if symbol and raw.get("symbol") != symbol:
                    continue
                orders.append(self._to_exchange_order(raw))
        return orders

    @override
    async def get_position(self, symbol: str) -> ExchangePosition | None:
        """Net DNSE's per-deal rows into one position (position_port is None)."""
        status, body = self.client.get_positions(self.account_id, self.market_type)
        if status != 200 or not isinstance(body, dict):
            return None
        net, cost = 0.0, 0.0
        for row in body.get("positions") or []:
            if row.get("symbol") != symbol:
                continue
            size = float(row.get("quantity") or 0)
            signed = size if str(row.get("side", "")).upper() in ("NB", "LONG") else -size
            net += signed
            cost += abs(signed) * float(row.get("averagePrice") or row.get("price") or 0)
        if net == 0:
            return None
        volume = abs(net)
        return ExchangePosition(
            symbol=symbol, side="buy" if net > 0 else "sell", size=volume,
            entry_price=(cost / volume) if volume else 0.0,
            unrealized_pnl=0.0, liquidation_price=None,
            leverage=1.0, margin_mode="cross",
        )

    @override
    async def get_balance(self) -> dict[str, float]:
        status, body = self.client.get_balances(self.account_id)
        if status != 200 or not isinstance(body, dict):
            return {}
        derivative = body.get("derivative") or {}
        stock = body.get("stock") or {}
        return {
            "VND": float(derivative.get("remainSecure")
                         or stock.get("availableCash") or 0),
        }
