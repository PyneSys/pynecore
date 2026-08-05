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

from pynecore.core.plugin import override, ProviderError
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

#: DNSE order status -> PyneCore OrderStatus. Keys are UPPERCASED at lookup:
#: the venue returns mixed case ("Pending", "New", "PendingCancel", "Canceled"
#: with one L, "Filled") — all observed live on 2026-08-05.
_STATUS_MAP = {
    "PENDING": OrderStatus.PENDING, "PENDINGNEW": OrderStatus.PENDING,
    "NEW": OrderStatus.OPEN, "OPEN": OrderStatus.OPEN,
    "PENDINGCANCEL": OrderStatus.OPEN,   # cancel in flight; still live
    "PARTIALLYFILLED": OrderStatus.PARTIALLY_FILLED,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED, "CANCELED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED, "EXPIRED": OrderStatus.EXPIRED,
}

#: Statuses that mean the order is no longer working.
_TERMINAL_STATUSES = frozenset({
    OrderStatus.FILLED, OrderStatus.CANCELLED,
    OrderStatus.REJECTED, OrderStatus.EXPIRED,
})

#: ``orderCategory`` does NOT filter reads — verified live 2026-08-05:
#: None/NORMAL/CONDITIONAL/STOP all return the identical set. So the order book
#: is fetched ONCE; iterating categories would return every order twice.
_READ_CATEGORY = None


@dataclass
class DNSEBrokerConfig(DNSEConfig):
    """:ivar account_no: DNSE trading account. Empty = resolve via /accounts.
    :ivar trading_token: Short-lived OTP-derived token for order operations.
    """

    account_no: str = ""
    trading_token: str = ""


#: TradingView timeframe -> seconds, for tick aggregation.
_TF_SECONDS = {"1": 60, "3": 180, "5": 300, "15": 900, "30": 1800,
               "60": 3600, "1H": 3600, "1D": 86400}


class _TickAggregator:
    """Build closed candles from the trade-print stream.

    DNSE's documented ``ohlc.{res}`` / ``ohlc_closed.{res}`` channels accept a
    subscription (the gateway reports ``status=active`` for ANY name, including
    a deliberately bogus one) but emit nothing for VN30F1M — 1,696 trade frames
    and zero bar frames were observed over 4 minutes of an open session on
    2026-08-05. Trades carry everything a candle needs, so the provider builds
    them itself rather than depending on a channel that may never fire.

    Session values on the tick (``openPrice``/``highestPrice``/``lowestPrice``)
    are DAY figures, not per-candle, so only ``matchPrice``/``matchQtty`` are
    used. A candle is emitted when the first tick of the following period
    arrives, which is what makes it final.
    """

    __slots__ = ("period", "_bucket", "_o", "_h", "_l", "_c", "_v")

    def __init__(self, period_seconds: int):
        self.period = period_seconds
        self._bucket: int | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._v = 0.0

    def add(self, price: float, qty: float, ts_seconds: int) -> dict | None:
        """Feed one trade print; return a CLOSED candle when the period rolls."""
        bucket = (ts_seconds // self.period) * self.period
        closed = None
        if self._bucket is None:
            self._bucket = bucket
            self._o = self._h = self._l = self._c = price
            self._v = qty
            return None
        if bucket != self._bucket:
            closed = {"t": self._bucket, "o": self._o, "h": self._h,
                      "l": self._l, "c": self._c, "v": self._v}
            self._bucket = bucket
            self._o = self._h = self._l = self._c = price
            self._v = qty
            return closed
        self._h = max(self._h, price)
        self._l = min(self._l, price)
        self._c = price
        self._v += qty
        return None


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
        #: intent_key -> [venue order ids]. DNSE accepts no client order id, so
        #: this is the only handle a later cancel/amend has on a placed order.
        self._order_ids: dict[str, list[str]] = {}
        #: venue order id -> (pine_id, from_entry, leg_type), so a pushed order
        #: event can carry the Pine identity the engine needs.
        self._identity: dict[str, tuple] = {}
        #: venue order id -> cumulative filled qty already reported.
        self._filled_so_far: dict[str, float] = {}
        #: venue order id -> (cumulative_fill, status) from the last poll.
        self._last_seen: dict[str, tuple] = {}
        #: seconds between order-book polls (fill latency ~= this)
        self._poll_interval: float = 2.0
        #: seconds between closed-bar polls of /price/ohlc
        self._bar_poll_interval: float = 3.0
        #: timestamp (s) of the last closed bar served, to avoid re-yielding
        self._last_bar_ts: int = 0

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
        # Await the auth verdict. Previously this was fire-and-forget, so a bad
        # signature left is_connected True while every read silently skipped.
        # Handshake, captured live 2026-08-05: the server sends `welcome`
        # immediately, then `auth_success` (or `auth_error`) some frames later.
        # Do NOT block on a single reply — `welcome` is not the auth verdict.
        reply = await self._ws.receive_json()
        if reply.get("action") == "auth_error":
            raise ProviderError(f"DNSE websocket auth rejected: {reply}")

        # One reader owns the socket. aiohttp forbids concurrent receive(), and
        # watch_ohlcv + watch_orders would otherwise both call it.
        self._bar_queue = asyncio.Queue()
        self._order_queue: asyncio.Queue = asyncio.Queue()
        self._subscribed = False
        self._reader_task = asyncio.create_task(self._read_loop())
        self._connected = True

    async def _read_loop(self) -> None:
        """Sole reader of the websocket; fans messages into two queues."""
        try:
            while self._ws is not None and not self._ws.closed:
                msg = await self._ws.receive_json()
                if msg.get("action") == "ping":
                    await self._ws.send_json({"action": "pong",
                                              "timestamp": msg.get("timestamp")})
                    continue
                action = msg.get("action")
                if action in ("welcome", "subscribed", "auth_success", "pong"):
                    continue
                if action == "auth_error":
                    self._connected = False
                    continue

                # Payloads are FLAT and tagged by "T" (captured live):
                #   "t"  = tick/trade   "b" = ohlc bar   orders carry orderStatus
                # There is no {"channel","data"} wrapper — an earlier inferred
                # shape discarded every frame silently.
                tag = str(msg.get("T", ""))
                if "orderStatus" in msg:
                    await self._order_queue.put(msg)
                elif tag in ("b", "bc"):
                    # Native bar, if DNSE ever starts publishing one.
                    await self._bar_queue.put(msg)
                elif tag == "t":
                    agg = getattr(self, "_aggregator", None)
                    price = float(msg.get("matchPrice") or 0)
                    if agg is None or not price:
                        continue
                    candle = agg.add(price, float(msg.get("matchQtty") or 0),
                                     int((msg.get("time") or {}).get("Seconds", 0)))
                    if candle is not None:
                        await self._bar_queue.put(candle)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Socket died (8-hour cap, network). The framework reconnects via
            # connect(), which rebuilds queues and restarts this task.
            self._connected = False

    async def _ensure_subscribed(self) -> None:
        if getattr(self, "_subscribed", False) or self._ws is None:
            return
        contract = self.resolve_contract()
        board = self._secdef(self.symbol or "").get("boardId") or "G1"
        period = _TF_SECONDS.get(self.timeframe or "5", 300)
        self._aggregator = _TickAggregator(period)
        # Bars are built from trades: the ohlc channels do not emit (see
        # _TickAggregator). The order channel is genuine and does emit.
        await self._ws.send_json({"action": "subscribe", "channels": [
            {"name": f"tick.{board}.json", "symbols": [contract]},
            {"name": f"order.{self.market_type}.json", "symbols": []},
        ]})
        self._subscribed = True

    @override
    async def disconnect(self) -> None:
        self._connected = False
        # _subscribed MUST reset, or a post-reconnect socket is never
        # subscribed and the feed stays silent forever.
        self._subscribed = False
        task = getattr(self, "_reader_task", None)
        if task is not None:
            task.cancel()
            self._reader_task = None
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
        """Yield the next CLOSED bar by polling REST ``/price/ohlc``.

        Measured 2026-08-05: tick-aggregated candles drift from DNSE's official
        bars — open off ~0.6, volume ~50% low — because the public ``tick.G1``
        print is a partial view of the matching engine's trade feed. REST
        ``/price/ohlc`` returns the authoritative OHLCV, so closed bars are
        taken from there (the same "REST is the source of truth, the WS channel
        is silent" pattern as fills). Ticks are not used to synthesise bars.

        Polls until a bar strictly newer than the last one served appears.
        """
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
                # Newest FULLY closed bar = the last one whose period has ended.
                idx = len(times) - 1
                while idx >= 0 and int(times[idx]) + period > now:
                    idx -= 1            # skip the still-forming bar
                if idx >= 0:
                    ts = int(times[idx])
                    if ts > getattr(self, "_last_bar_ts", 0):
                        self._last_bar_ts = ts
                        return OHLCV(
                            timestamp=ts * 1000,
                            open=float(body["o"][idx]), high=float(body["h"][idx]),
                            low=float(body["l"][idx]), close=float(body["c"][idx]),
                            volume=float(body["v"][idx]), is_closed=True)
            await asyncio.sleep(self._bar_poll_interval)

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
            status=_STATUS_MAP.get(
                str(raw.get("orderStatus", "")).upper().replace("_", "").replace("-", ""),
                OrderStatus.PENDING),
            timestamp=int(time.time() * 1000),
            fee=0.0, fee_currency="VND",
        )

    def _marketable_price(self, side: str) -> float:
        """Price that fills immediately, for an intent with no limit/stop.

        DNSE's market order types (MTL/MOK/MAK) are unverified against this
        account, so a MARKET intent is expressed as a *marketable limit* at the
        daily band edge — buy at the ceiling, sell at the floor. Only the ``LO``
        type is used, which is the one confirmed to work.

        Sending ``price=0.0`` (the previous behaviour) is rejected outright.
        """
        row = self._secdef(self.symbol or "")
        ceiling = float(row.get("ceilingPrice") or 0)
        floor = float(row.get("floorPrice") or 0)
        if not ceiling or not floor:
            raise RuntimeError(
                f"cannot price a market {side}: secdef has no ceiling/floor for "
                f"{self.symbol!r}"
            )
        return ceiling if side == "buy" else floor

    def _place(self, envelope, side: str, qty: float, price: float | None,
               category: str = "NORMAL", leg_type=None) -> list[ExchangeOrder]:
        """Place one order and remember its venue id against the intent key.

        DNSE accepts no client-supplied order id, so ``intent_key`` -> venue id
        is the only way a later cancel/amend can name this order.
        """
        from pynecore.core.broker.exceptions import (
            ExchangeOrderRejectedError, OrderDispositionUnknownError)

        payload = {
            "accountNo": self.account_id,
            # Orders need the tradable KRX contract, not the symbolType alias.
            "symbol": self.resolve_contract(),
            "side": _SIDE_TO_DNSE[side],
            "orderType": "LO",
            "price": round(float(price) if price else self._marketable_price(side), 1),
            "quantity": int(qty),
            "loanPackageId": self._loan_package_id(),
        }
        status, body = self.client.post_order(self.market_type, payload,
                                              self._token(), order_category=category)
        if status == 0:
            # Socket dropped mid-flight: the order may well have landed, so this
            # is NOT a rejection. The engine parks it for verification instead.
            raise OrderDispositionUnknownError(
                f"DNSE order disposition unknown: {body}",
                client_order_id=envelope.client_order_id(),
            )
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
        from pynecore.core.broker.models import LegType
        return self._place(envelope, intent.side, intent.qty,
                           intent.limit or intent.stop, leg_type=LegType.ENTRY)

    @override
    async def execute_exit(self, envelope) -> list[ExchangeOrder]:
        from pynecore.core.broker.models import LegType
        intent = envelope.intent
        orders: list[ExchangeOrder] = []
        for price, leg in ((intent.tp_price, LegType.TAKE_PROFIT),
                           (intent.sl_price, LegType.STOP_LOSS)):
            if price is not None:
                orders += self._place(envelope, intent.side, intent.qty, price,
                                      category="CONDITIONAL", leg_type=leg)
        if not orders:
            # A trailing-only exit would otherwise return [] and the engine
            # would believe the exit was dispatched. Fail loudly instead.
            from pynecore.core.broker.exceptions import OrderSkippedByPlugin
            raise OrderSkippedByPlugin(
                "DNSE plugin cannot express this exit: no tp_price/sl_price "
                "(trailing stops are not implemented)")
        return orders

    @override
    async def execute_close(self, envelope) -> ExchangeOrder:
        intent = envelope.intent
        from pynecore.core.broker.models import LegType
        return self._place(envelope, intent.side, intent.qty, None,
                           leg_type=LegType.CLOSE)[0]

    @override
    def _identity_for(self, order_id: str) -> tuple:
        """Pine identity recorded when this order was placed."""
        return self._identity.get(order_id, (None, None, None))

    def _ids_for(self, envelope) -> list[str]:
        """Venue order ids previously placed for this envelope's intent."""
        key = getattr(envelope.intent, "intent_key", None)
        return list(self._order_ids.get(key, [])) if key else []

    @override
    async def execute_cancel(self, envelope) -> bool:
        ids = self._ids_for(envelope)
        if not ids:
            return False
        ok = True
        for order_id in ids:
            status, _ = self.client.cancel_order(self.account_id, str(order_id),
                                                 self.market_type, self._token())
            ok = ok and status in (200, 204)
        return ok

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
        return await self._amend(old, new, is_exit=True)

    async def _amend(self, old, new, *, is_exit: bool = False) -> list[ExchangeOrder]:
        ids = self._ids_for(old)
        if not ids:
            # Nothing known to amend — fall back to the CORRECT base method.
            # Using modify_entry for an exit would place an ENTRY order in
            # response to a bracket change, adding to the position.
            return await (super().modify_exit(old, new) if is_exit
                          else super().modify_entry(old, new))
        order_id = ids[0]
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
        ids = self._ids_for(envelope)
        if not ids:
            return CancelDispositionOutcome.UNKNOWN
        status, body = self.client.cancel_order(
            self.account_id, str(ids[0]), self.market_type, self._token())
        if status in (200, 204):
            return CancelDispositionOutcome.CANCEL_CONFIRMED
        code = (body or {}).get("code", "") if isinstance(body, dict) else ""
        if code == "RESOURCE_NOT_FOUND":
            # The order is gone from the repository — it either filled or was
            # already cancelled. The engine resolves which via the event stream.
            return CancelDispositionOutcome.UNKNOWN
        if code == "CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION":
            # NOT STILL_OPEN: that means "the cancel succeeded and the order is
            # gone", and the engine would abort the bracket legs of a position
            # that is still very much live. The cancel simply did not resolve.
            return CancelDispositionOutcome.UNKNOWN
        return CancelDispositionOutcome.UNKNOWN

    @override
    async def watch_orders(self):
        """Detect fills by POLLING, not by the websocket order channel.

        ``order.{marketType}.json`` accepts a subscription but emits nothing on
        this connection — 1,696 tick frames and ZERO order frames were observed
        over 4 minutes of an open session (2026-08-05). Relying on it meant the
        engine never learned an entry had filled, kept believing the order was
        resting, and tried to amend a filled order on the next bar.

        So the plugin honours its own ``idempotency=SOFTWARE`` declaration and
        derives events from successive REST snapshots. The blocking call runs
        off-loop so it cannot stall the websocket heartbeat.
        """
        from pynecore.core.broker.models import OrderEvent

        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                status, body = await asyncio.to_thread(
                    self.client.get_orders, self.account_id, self.market_type, None)
            except Exception:
                continue
            if status != 200 or not isinstance(body, dict):
                continue

            for raw in body.get("orders") or []:
                order_id = str(raw.get("id"))
                order = self._to_exchange_order(raw)
                cumulative = float(raw.get("fillQuantity") or 0)
                previous, prev_status = self._last_seen.get(order_id, (0.0, None))
                if cumulative == previous and order.status is prev_status:
                    continue                      # nothing changed
                self._last_seen[order_id] = (cumulative, order.status)

                # Only orders this run placed carry a Pine identity; anything
                # else on the account is not ours to report.
                pine_id, from_entry, leg_type = self._identity_for(order_id)
                if pine_id is None:
                    continue

                delta = max(cumulative - previous, 0.0)
                if order.status is OrderStatus.FILLED:
                    event_type = "filled"
                elif order.status is OrderStatus.PARTIALLY_FILLED:
                    event_type = "partial"
                elif order.status is OrderStatus.CANCELLED:
                    event_type = "cancelled"
                elif order.status is OrderStatus.REJECTED:
                    event_type = "rejected"
                else:
                    event_type = "created"

                yield OrderEvent(
                    order=order, event_type=event_type,
                    fill_price=float(raw.get("averagePrice") or 0) or None,
                    fill_qty=delta or None,
                    timestamp=int(time.time()),
                    pine_id=pine_id, from_entry=from_entry, leg_type=leg_type,
                )

    @override
    async def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        """Union of BOTH order books.

        A book whose fetch FAILS is skipped rather than reported empty — an
        incomplete snapshot must never look like a complete absence.
        """
        from pynecore.core.broker.exceptions import ExchangeConnectionError

        status, body = self.client.get_orders(self.account_id, self.market_type,
                                              _READ_CATEGORY)
        if status != 200 or not isinstance(body, dict):
            # An incomplete snapshot must never look like a complete absence:
            # returning [] here makes the engine believe every order vanished.
            raise ExchangeConnectionError(
                f"DNSE order book unavailable: {status} {body}")

        # The engine passes syminfo.ticker (the alias, e.g. VN30F1M) while
        # orders carry the tradable KRX contract (41I1G8000).
        wanted = self.resolve_contract(symbol) if symbol else None
        orders: list[ExchangeOrder] = []
        for raw in body.get("orders") or []:
            if wanted and raw.get("symbol") != wanted:
                continue
            order = self._to_exchange_order(raw)
            if order.status in _TERMINAL_STATUSES:
                continue      # filled/cancelled/rejected are not OPEN orders
            orders.append(order)
        return orders

    @override
    async def get_position(self, symbol: str) -> ExchangePosition | None:
        """Net DNSE's per-deal rows into one position (position_port is None)."""
        status, body = self.client.get_positions(self.account_id, self.market_type)
        if status != 200 or not isinstance(body, dict):
            return None
        wanted = self.resolve_contract(symbol)
        net, cost = 0.0, 0.0
        for row in body.get("positions") or []:
            if row.get("symbol") != wanted:
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
