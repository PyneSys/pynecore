"""Binance SPOT broker plugin for PyneCore (ccxt transport).

Builds on :class:`BinanceProvider` (history + ccxt.pro bars) and implements the
``BrokerPlugin`` abstracts against Binance spot REST via ccxt.

Design:

* SPOT, long-only: ``short_selling = UNSUPPORTED`` (enforced at startup by
  ``validate_at_startup``); the Pine position is synthesized by the core
  spot-inventory layer (``spot_inventory_port``) from this bot's fill ledger.
* Pine intent -> venue order:
  ``entry(market)`` -> MARKET; ``entry(limit)`` -> LIMIT;
  ``entry(stop)`` -> STOP_LOSS_LIMIT priced *through* the trigger;
  ``exit(tp+sl)`` -> native spot OCO (LIMIT_MAKER above + STOP_LOSS_LIMIT
  below, venue-side one-cancels-other); ``exit(tp)`` -> LIMIT;
  ``exit(sl)`` -> STOP_LOSS_LIMIT; ``close`` -> MARKET.
* Idempotency is NATIVE: every write carries ``newClientOrderId`` (the
  engine-minted deterministic id; Binance echoes it and rejects duplicates).
* Events are REST-polled (SOFTWARE): fills come from ``myTrades`` behind an
  id cursor (also feeding the inventory ledger); cancels/rejects from
  re-reading tracked order ids. ``fill_id`` is the venue ``tradeId`` on every
  path, so the engine's duplicate-fill gate holds across poll and reconcile.
* SAFETY: refuses to construct against mainnet unless ``allow_mainnet = true``
  is set explicitly — ``sandbox = true`` (spot testnet) is the proving ground.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal

from pynecore.core.plugin import override
from pynecore.core.plugin.broker import BrokerPlugin
from pynecore.core.broker.models import (
    CapabilityLevel, ExchangeCapabilities, ExchangeOrder, ExchangePosition,
    LegType, OrderEvent, OrderStatus, OrderType,
)
from pynecore.core.broker.exceptions import (
    ExchangeCapabilityError, ExchangeConnectionError, ExchangeOrderRejectedError,
    OrderSkippedByPlugin,
)
from pynecore.core.broker.idempotency import (
    KIND_CLOSE, KIND_ENTRY, KIND_EXIT_SL, KIND_EXIT_TP)
from pynecore.lib import log

from .errors import is_order_gone, is_trigger_immediate, map_ccxt_exception
from .inventory import BinanceSpotPort, execution_from_trade
from .provider import BinanceConfig, BinanceProvider

__all__ = ['BinanceBroker', 'BinanceBrokerConfig']

#: Binance order status -> PyneCore OrderStatus (raw ``info.status`` wins over
#: the ccxt-unified string; both spellings are covered).
_STATUS_MAP = {
    "NEW": OrderStatus.OPEN, "open": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED, "closed": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED, "canceled": OrderStatus.CANCELLED,
    "PENDING_CANCEL": OrderStatus.OPEN,
    "REJECTED": OrderStatus.REJECTED, "rejected": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED, "expired": OrderStatus.EXPIRED,
    "EXPIRED_IN_MATCH": OrderStatus.EXPIRED,
}

_TERMINAL_STATUSES = frozenset({
    OrderStatus.FILLED, OrderStatus.CANCELLED,
    OrderStatus.REJECTED, OrderStatus.EXPIRED,
})

#: LegType -> idempotency KIND for client-order-id minting.
_LEG_KIND = {
    LegType.ENTRY: KIND_ENTRY, LegType.TAKE_PROFIT: KIND_EXIT_TP,
    LegType.STOP_LOSS: KIND_EXIT_SL, LegType.CLOSE: KIND_CLOSE,
}

#: Run the inventory reconcile every Nth poll cycle.
_RECONCILE_EVERY = 5


@dataclass
class BinanceBrokerConfig(BinanceConfig):
    """:ivar allow_mainnet: Explicit opt-in for LIVE trading. With
        ``sandbox = false`` and this ``false`` (default) the broker refuses to
        start — the testnet is the proving ground.
    :ivar stop_slippage_ticks: Offset (in ticks) applied *through* a stop's
        trigger when pricing the STOP_LOSS_LIMIT it emits. Binance spot has no
        stop-market; a stop-limit priced exactly at the trigger never fills on
        a gap through, so the limit is pushed past the trigger by this much.
    :ivar poll_interval: Seconds between order/fill poll cycles.
    """

    allow_mainnet: bool = False
    stop_slippage_ticks: int = 10
    poll_interval: float = 2.0


class BinanceBroker(BinanceProvider, BrokerPlugin[BinanceBrokerConfig]):
    """Binance spot broker: long-only, native OCO brackets, REST-polled events."""

    plugin_name = "Binance Broker"
    Config = BinanceBrokerConfig

    #: Spot pooled balances — no hedged legs to emulate.
    position_port = None

    #: Binance spot ``newClientOrderId`` accepts up to 36 chars.
    client_order_id_max_len = 36

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if (self.config is not None and not self.config.sandbox
                and not self.config.allow_mainnet):
            raise ExchangeCapabilityError(
                "Binance broker refuses MAINNET: set sandbox = true (testnet "
                "keys from https://testnet.binance.vision) or explicitly set "
                "allow_mainnet = true in workdir/config/plugins/binance_broker.toml")
        self._connected = False
        #: intent_key -> [venue order ids] (the handle a later cancel uses).
        self._order_ids: dict[str, list[str]] = {}
        #: venue order id -> (pine_id, from_entry, leg_type) for event tagging.
        self._identity: dict[str, tuple] = {}
        #: client order id -> same identity (reconcile-recovered fills).
        self._identity_by_coid: dict[str, tuple] = {}
        #: venue order id -> client order id; also the "is this fill ours" set.
        self._coid_by_order_id: dict[str, str] = {}
        self._coids: set[str] = set()
        #: venue order id -> last ExchangeOrder snapshot (qty, cum fill, …).
        self._orders: dict[str, ExchangeOrder] = {}
        #: order ids still worth polling for terminal transitions.
        self._live_ids: set[str] = set()
        #: venue order id -> OCO orderListId (cancel targets the list once).
        self._oco_list_id: dict[str, str] = {}
        self._cancelled_lists: set[str] = set()
        #: in-memory trade-id cursor for the live fill poll (anchored at start).
        self._trade_cursor: int | None = None
        self._market: dict | None = None
        self._spot_manager = None
        self._spot_port: BinanceSpotPort | None = None
        self._broker_started = False
        self._broker_start_lock = asyncio.Lock()
        #: last seen close (from bars) for position synthesis marks.
        self._last_price: float | None = None
        #: no-store fallback ledger (tests / persistence off): net base + cost.
        self._mem_net = Decimal(0)
        self._mem_cost = Decimal(0)
        #: This run's 4-char idempotency tag, captured from the first envelope.
        self._run_tag: str | None = None

    # --- account identity ---

    @property
    def account_id(self) -> str:
        """Lazily resolved on first access — broker storage derives the run
        identity BEFORE ``connect()`` runs, so this must not wait for it."""
        if self._account_id is None:
            balance = self._client.fetch_balance()
            uid = str((balance.get('info') or {}).get('uid') or 'spot')
            mode = 'testnet' if (self.config and self.config.sandbox) else 'live'
            self._account_id = f"binance-{mode}-{uid}"
        return self._account_id

    # --- capabilities ---

    @override
    def get_capabilities(self) -> ExchangeCapabilities:
        return ExchangeCapabilities(
            stop_order=CapabilityLevel.NATIVE,       # STOP_LOSS_LIMIT, server-side
            tp_sl_bracket=CapabilityLevel.NATIVE,    # spot OCO
            # SOFTWARE, not NATIVE: only EXIT tp+sl pairs are venue-run (spot
            # OCO orderList). ENTRY OCA groups have no venue link — measured
            # live 2026-08-17 (BF7): with NATIVE the engine suppressed its
            # sibling-cancel and the far entry leg stayed working. SOFTWARE
            # lets the engine cancel siblings; its extra cancel on an
            # already-gone OCO leg is idempotent (verified-gone path).
            oca_cancel=CapabilityLevel.SOFTWARE,
            trailing_stop=CapabilityLevel.UNSUPPORTED,
            partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
            partial_qty_bracket_exit_pyramiding=CapabilityLevel.SOFTWARE,
            amend_order=CapabilityLevel.SOFTWARE,    # cancel + replace (base class)
            cancel_all=CapabilityLevel.NATIVE,       # DELETE /api/v3/openOrders
            reduce_only=CapabilityLevel.SOFTWARE,    # inventory-capped sells
            watch_orders=CapabilityLevel.SOFTWARE,   # REST poll
            fetch_position=CapabilityLevel.SOFTWARE, # synthesized from the ledger
            idempotency=CapabilityLevel.NATIVE,      # newClientOrderId
            short_selling=CapabilityLevel.UNSUPPORTED,
        )

    # --- live plumbing ---

    @override
    async def connect(self) -> None:
        await super().connect()                      # ccxt.pro bar stream
        # Auth probe (account_id resolves lazily, possibly already done).
        account = await asyncio.to_thread(lambda: self.account_id)
        self._connected = True
        log.broker_info("%s", f"Binance connected: account={account}")

    @override
    async def disconnect(self) -> None:
        self._connected = False
        await super().disconnect()

    @property
    @override
    def is_connected(self) -> bool:
        return self._connected

    async def _ensure_broker_started(self) -> None:
        """One-shot: market filters, inventory port + manager, fill cursor."""
        if self._broker_started:
            return
        async with self._broker_start_lock:
            if self._broker_started:
                return
            await self._venue("load_markets", "", self._client.load_markets)
            self._market = self._client.market(self.symbol)
            port = BinanceSpotPort(self, self._market)
            self.spot_inventory_port = port
            self._spot_port = port
            if self.store_ctx is not None:
                from pynecore.core.broker.spot_inventory import SpotInventoryManager
                manager = SpotInventoryManager(
                    self.store_ctx, port,
                    account_id=self.account_id,
                    symbol=self.symbol or port.product_id,
                    request_quarantine=self.quarantine_sink,
                    on_inventory_conflict=self.on_inventory_conflict,
                )
                result = await manager.startup()
                self._spot_manager = manager
                if result.quarantined:
                    log.broker_error("%s", f"spot inventory startup quarantined: "
                                           f"{result.reason}")
                else:
                    log.broker_info("%s", (
                        f"spot inventory ready: net_base={result.fold.net_base} "
                        f"fills={result.fold.fill_count} "
                        f"(recovered={result.recovered_fills}, "
                        f"adopted={result.adopted_fills})"))
            if self._trade_cursor is None:
                self._trade_cursor = await asyncio.to_thread(self.newest_trade_id)
            self._broker_started = True

    # --- venue call helpers (sync ccxt behind a thread + taxonomy mapping) ---

    async def _venue(self, action: str, coid: str, fn, *args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:                                    # noqa: BLE001
            mapped = map_ccxt_exception(exc, action=action, client_order_id=coid)
            if mapped is not None:
                raise mapped from exc
            raise

    @override
    def _map_exception(self, raw: Exception):
        return map_ccxt_exception(raw, action="call") or super()._map_exception(raw)

    # --- sync venue reads used by the inventory port (called off-loop) ---

    def newest_trade_id(self) -> int:
        """The account's newest ``myTrades`` id for this symbol (0 if none)."""
        rows = self._client.fetch_my_trades(self.symbol, limit=1)
        return int(rows[-1]['id']) if rows else 0

    def fetch_trades_after(self, cursor: int, limit: int) -> list[dict]:
        """Trades with id strictly greater than ``cursor``, oldest first."""
        return self._client.fetch_my_trades(
            self.symbol, limit=limit, params={'fromId': cursor + 1})

    def client_order_id_for_trade(self, trade: dict) -> str:
        """The BOT-owned client id behind a trade, or ``""`` for foreign fills."""
        order_id = str(trade.get('order') or '')
        if not order_id:
            return ""
        cached = self._coid_by_order_id.get(order_id)
        if cached is not None:
            return cached
        try:
            raw = self._client.fetch_order(order_id, self.symbol)
        except Exception:                                           # noqa: BLE001
            return ""                       # unattributable now -> retry later
        coid = str(raw.get('clientOrderId') or '')
        ours = coid if self._is_our_coid(coid) else ""
        self._coid_by_order_id[order_id] = ours
        return ours

    def _is_our_coid(self, coid: str) -> bool:
        """Ours = minted this process, or wire-form with THIS run's tag.

        ``run_tag`` is deterministic per logical run (stable across restarts),
        so crash-recovery fills from a previous process instance still
        attribute correctly.
        """
        if not coid:
            return False
        if coid in self._coids:
            return True
        if self._run_tag is None:
            return False
        from pynecore.core.broker.idempotency import parse_wire_client_order_id
        parsed = parse_wire_client_order_id(coid)
        return parsed is not None and parsed.run_tag == self._run_tag

    def total_asset_balance(self, asset: str) -> Decimal:
        """TOTAL (free + locked) balance of ``asset`` as an exact Decimal."""
        balance = self._client.fetch_balance()
        for row in (balance.get('info') or {}).get('balances') or []:
            if row.get('asset') == asset:
                return (Decimal(str(row.get('free') or '0'))
                        + Decimal(str(row.get('locked') or '0')))
        return Decimal(0)

    # --- market filters / pricing ---

    @property
    def qty_step(self) -> float:
        market = self._market or {}
        return float((market.get('limits', {}).get('amount', {}) or {}).get('min')
                     or market.get('precision', {}).get('amount') or 0)

    def _mintick(self) -> float:
        market = self._market or {}
        tick = market.get('precision', {}).get('price')
        return float(tick) if tick else 1e-8

    def _fmt_price(self, price: float) -> float:
        return float(self._client.price_to_precision(self.symbol, price))

    def _fmt_amount(self, qty: float) -> float:
        return float(self._client.amount_to_precision(self.symbol, qty))

    def _preflight(self, envelope, qty: float, price: float | None) -> float:
        """Quantize + validate against LOT_SIZE / MIN_NOTIONAL; skip below-min."""
        market = self._market or {}
        limits = market.get('limits', {})
        amount = self._fmt_amount(qty)
        min_amount = (limits.get('amount', {}) or {}).get('min')
        if amount <= 0 or (min_amount and amount < float(min_amount)):
            raise OrderSkippedByPlugin(
                f"qty {qty} below LOT_SIZE minimum {min_amount}",
                intent_key=getattr(envelope.intent, 'intent_key', ''))
        notional_price = price or self._last_price
        min_cost = (limits.get('cost', {}) or {}).get('min')
        if min_cost and notional_price and amount * notional_price < float(min_cost):
            raise OrderSkippedByPlugin(
                f"notional {amount * notional_price:.2f} below MIN_NOTIONAL "
                f"{min_cost}",
                intent_key=getattr(envelope.intent, 'intent_key', ''))
        return amount

    def _stop_fill_price(self, side: str, stop_price: float) -> float:
        """Price a stop's limit THROUGH the trigger so a gap still fills."""
        assert self.config is not None
        offset = self._mintick() * self.config.stop_slippage_ticks
        return self._fmt_price(stop_price + offset if side == 'buy'
                               else stop_price - offset)

    # --- order construction ---

    def _to_exchange_order(self, raw: dict) -> ExchangeOrder:
        """Map a ccxt-unified order (or raw Binance report) to ExchangeOrder."""
        info = raw.get('info') or raw
        status_key = str(info.get('status') or raw.get('status') or 'NEW')
        status = _STATUS_MAP.get(status_key, OrderStatus.OPEN)
        qty = float(raw.get('amount') or info.get('origQty') or 0)
        filled = float(raw.get('filled') or info.get('executedQty') or 0)
        if status is OrderStatus.OPEN and filled > 0:
            status = OrderStatus.PARTIALLY_FILLED
        price = float(raw.get('price') or info.get('price') or 0) or None
        stop_price = float(raw.get('stopPrice') or raw.get('triggerPrice')
                           or info.get('stopPrice') or 0) or None
        executed_quote = float(info.get('cummulativeQuoteQty') or 0)
        average = (float(raw.get('average') or 0)
                   or (executed_quote / filled if filled else 0)) or None
        order_type = (OrderType.MARKET
                      if str(raw.get('type') or info.get('type') or '').lower()
                      == 'market' else
                      OrderType.STOP if stop_price is not None else OrderType.LIMIT)
        timestamp_ms = raw.get('timestamp') or info.get('transactTime') or 0
        fee = raw.get('fee') or {}
        return ExchangeOrder(
            id=str(raw.get('id') or info.get('orderId')),
            symbol=self.symbol or str(info.get('symbol') or ''),
            side=str(raw.get('side') or info.get('side') or 'buy').lower(),
            order_type=order_type,
            qty=qty, filled_qty=filled,
            remaining_qty=max(qty - filled, 0.0),
            price=price, stop_price=stop_price,
            average_fill_price=average,
            status=status,
            timestamp=float(timestamp_ms) / 1000.0 if timestamp_ms else time.time(),
            fee=float(fee.get('cost') or 0), fee_currency=str(fee.get('currency') or ''),
            client_order_id=str(raw.get('clientOrderId')
                                or info.get('clientOrderId') or '') or None,
        )

    def _track(self, order: ExchangeOrder, envelope, leg_type: LegType,
               coid: str) -> ExchangeOrder:
        self._run_tag = getattr(envelope, 'run_tag', None) or self._run_tag
        intent = envelope.intent
        identity = (getattr(intent, 'pine_id', None),
                    getattr(intent, 'from_entry', None), leg_type)
        key = getattr(intent, 'intent_key', None)
        if key:
            self._order_ids.setdefault(key, []).append(order.id)
        self._identity[order.id] = identity
        self._identity_by_coid[coid] = identity
        self._coid_by_order_id[order.id] = coid
        self._coids.add(coid)
        self._orders[order.id] = order
        if order.status not in _TERMINAL_STATUSES:
            self._live_ids.add(order.id)
        return order

    # --- BrokerPlugin abstracts: execution ---

    @override
    async def execute_entry(self, envelope) -> list[ExchangeOrder]:
        await self._ensure_broker_started()
        intent = envelope.intent
        coid = envelope.client_order_id(KIND_ENTRY)
        params: dict = {'newClientOrderId': coid}
        if intent.stop is not None:
            price = self._fmt_price(intent.limit if intent.limit is not None
                                    else self._stop_fill_price(intent.side, intent.stop))
            amount = self._preflight(envelope, intent.qty, price)
            params.update({'stopPrice': self._fmt_price(intent.stop),
                           'timeInForce': 'GTC'})
            raw = await self._place_stop_with_market_fallback(
                "place:entry-stop", coid, intent.side, amount, price, params,
                fallback_limit=(self._fmt_price(intent.limit)
                                if intent.limit is not None else None))
        elif intent.limit is not None:
            price = self._fmt_price(intent.limit)
            amount = self._preflight(envelope, intent.qty, price)
            params['timeInForce'] = 'GTC'
            raw = await self._venue("place:entry-limit", coid,
                                    self._client.create_order, self.symbol,
                                    'limit', intent.side, amount, price, params)
        else:
            amount = self._preflight(envelope, intent.qty, None)
            raw = await self._venue("place:entry-market", coid,
                                    self._client.create_order, self.symbol,
                                    'market', intent.side, amount, None, params)
        return [self._track(self._to_exchange_order(raw), envelope,
                            LegType.ENTRY, coid)]

    @override
    async def execute_exit(self, envelope) -> list[ExchangeOrder]:
        """TP+SL -> native spot OCO; single-leg exits -> LIMIT / stop-limit."""
        await self._ensure_broker_started()
        intent = envelope.intent
        tp, sl = intent.tp_price, intent.sl_price
        if tp is not None and sl is not None:
            return await self._place_oco(envelope, tp_price=tp, sl_price=sl)
        if sl is not None:
            coid = envelope.client_order_id(KIND_EXIT_SL)
            price = self._stop_fill_price(intent.side, sl)
            amount = self._preflight(envelope, intent.qty, price)
            raw = await self._place_stop_with_market_fallback(
                "place:exit-sl", coid, intent.side, amount, price,
                {'newClientOrderId': coid,
                 'stopPrice': self._fmt_price(sl),
                 'timeInForce': 'GTC'})
            return [self._track(self._to_exchange_order(raw), envelope,
                                LegType.STOP_LOSS, coid)]
        if tp is not None:
            coid = envelope.client_order_id(KIND_EXIT_TP)
            price = self._fmt_price(tp)
            amount = self._preflight(envelope, intent.qty, price)
            raw = await self._venue("place:exit-tp", coid,
                                    self._client.create_order, self.symbol,
                                    'limit', intent.side, amount, price,
                                    {'newClientOrderId': coid,
                                     'timeInForce': 'GTC'})
            return [self._track(self._to_exchange_order(raw), envelope,
                                LegType.TAKE_PROFIT, coid)]
        raise OrderSkippedByPlugin(
            "Binance spot plugin cannot express this exit: no tp_price/sl_price "
            "(trailing stops are not implemented)",
            intent_key=getattr(intent, 'intent_key', ''))

    async def _place_stop_with_market_fallback(self, action: str, coid: str,
                                               side: str, amount: float,
                                               price: float, params: dict,
                                               fallback_limit: float | None = None
                                               ) -> dict:
        """Place a STOP_LOSS_LIMIT; if the venue refuses because the market has
        already crossed the trigger (-2010 "would trigger immediately"), fall
        back per Pine crossed-stop semantics: a plain stop fills at MARKET;
        a stop-LIMIT keeps the user's price cap and rests as a LIMIT
        (``fallback_limit``). The coid is reusable: the rejected stop never
        consumed it. Measured live 2026-08-17 (BF3/BF5)."""
        try:
            return await self._venue(action, coid, self._client.create_order,
                                     self.symbol, 'limit', side, amount, price,
                                     params)
        except Exception as exc:                                    # noqa: BLE001
            if not (is_trigger_immediate(exc)
                    or is_trigger_immediate(exc.__cause__ or exc)):
                raise
            if fallback_limit is not None:
                log.broker_warning("%s", (
                    f"{action}: trigger already crossed (venue -2010) -> "
                    f"falling back to LIMIT @ {fallback_limit} (stop-limit "
                    f"keeps its price cap)"))
                return await self._venue(f"{action}-as-limit", coid,
                                         self._client.create_order, self.symbol,
                                         'limit', side, amount, fallback_limit,
                                         {'newClientOrderId': coid,
                                          'timeInForce': 'GTC'})
            log.broker_warning("%s", (
                f"{action}: trigger already crossed (venue -2010) -> "
                f"falling back to MARKET (Pine crossed-stop semantics)"))
            return await self._venue(f"{action}-as-market", coid,
                                     self._client.create_order, self.symbol,
                                     'market', side, amount, None,
                                     {'newClientOrderId': coid})

    async def _place_oco(self, envelope, *, tp_price: float,
                         sl_price: float) -> list[ExchangeOrder]:
        """One venue-side OCO: TP LIMIT_MAKER above + SL STOP_LOSS_LIMIT below.

        Only the SELL direction exists on a long-only spot book (the engine
        never asks for a short's buy-side bracket — ``short_selling`` is
        declared unsupported).
        """
        intent = envelope.intent
        if intent.side != 'sell':
            raise ExchangeCapabilityError(
                "Binance spot OCO bracket is sell-side only (long-only venue)")
        coid_tp = envelope.client_order_id(KIND_EXIT_TP)
        coid_sl = envelope.client_order_id(KIND_EXIT_SL)
        amount = self._preflight(envelope, intent.qty, tp_price)
        assert self._market is not None
        payload = {
            'symbol': self._market['id'],
            'side': 'SELL',
            'quantity': self._client.amount_to_precision(self.symbol, amount),
            'aboveType': 'LIMIT_MAKER',
            'abovePrice': self._client.price_to_precision(self.symbol, tp_price),
            'aboveClientOrderId': coid_tp,
            'belowType': 'STOP_LOSS_LIMIT',
            'belowStopPrice': self._client.price_to_precision(self.symbol, sl_price),
            'belowPrice': self._client.price_to_precision(
                self.symbol, self._stop_fill_price('sell', sl_price)),
            'belowTimeInForce': 'GTC',
            'belowClientOrderId': coid_sl,
        }
        raw = await self._venue("place:exit-oco", coid_tp,
                                self._client.private_post_orderlist_oco, payload)
        list_id = str(raw.get('orderListId') or '')
        reports = raw.get('orderReports') or raw.get('orders') or []
        orders: list[ExchangeOrder] = []
        for report in reports:
            order = self._to_exchange_order(report)
            is_sl = (order.stop_price is not None
                     or order.client_order_id == coid_sl)
            leg = LegType.STOP_LOSS if is_sl else LegType.TAKE_PROFIT
            coid = coid_sl if is_sl else coid_tp
            orders.append(self._track(order, envelope, leg, coid))
            if list_id:
                self._oco_list_id[order.id] = list_id
        if not orders:
            raise ExchangeOrderRejectedError(
                f"binance OCO: unparseable response: {raw!r}")
        return orders

    @override
    async def execute_close(self, envelope) -> ExchangeOrder:
        await self._ensure_broker_started()
        intent = envelope.intent
        coid = envelope.client_order_id(KIND_CLOSE)
        amount = self._preflight(envelope, intent.qty, None)
        raw = await self._venue("place:close", coid,
                                self._client.create_order, self.symbol,
                                'market', intent.side, amount, None,
                                {'newClientOrderId': coid})
        return self._track(self._to_exchange_order(raw), envelope,
                           LegType.CLOSE, coid)

    # --- cancel ---

    def _ids_for(self, envelope) -> list[str]:
        key = getattr(envelope.intent, 'intent_key', None)
        return list(self._order_ids.get(key, [])) if key else []

    async def _cancel_one(self, order_id: str) -> bool:
        """Cancel one tracked order; an OCO leg cancels its list exactly once.

        "Unknown order" is verified against ``fetch_order`` before counting as
        gone; the venue answering CANCELED/terminal is the only success proof.
        """
        list_id = self._oco_list_id.get(order_id)
        try:
            if list_id:
                if list_id in self._cancelled_lists:
                    return True
                assert self._market is not None
                await self._venue("cancel:oco", order_id,
                                  self._client.private_delete_orderlist,
                                  {'symbol': self._market['id'],
                                   'orderListId': list_id})
                self._cancelled_lists.add(list_id)
                return True
            await self._venue("cancel", order_id,
                              self._client.cancel_order, order_id, self.symbol)
            return True
        except Exception as exc:                                    # noqa: BLE001
            if not is_order_gone(exc) and not is_order_gone(exc.__cause__ or exc):
                raise
        # "Unknown order" — confirm the disposition instead of assuming gone.
        try:
            raw = await asyncio.to_thread(
                self._client.fetch_order, order_id, self.symbol)
        except Exception:                                           # noqa: BLE001
            return True                     # not on the book at all
        return self._to_exchange_order(raw).status in _TERMINAL_STATUSES

    @override
    async def execute_cancel(self, envelope) -> bool:
        ids = self._ids_for(envelope)
        if not ids:
            return False
        ok = True
        for order_id in ids:
            ok = await self._cancel_one(str(order_id)) and ok
        return ok

    @override
    async def execute_cancel_with_outcome(self, envelope):
        from pynecore.core.broker.models import CancelDispositionOutcome
        ids = self._ids_for(envelope)
        if not ids:
            return CancelDispositionOutcome.UNKNOWN
        order_id = str(ids[0])
        try:
            cancelled = await self._cancel_one(order_id)
        except Exception:                                           # noqa: BLE001
            return CancelDispositionOutcome.UNKNOWN
        if not cancelled:
            return CancelDispositionOutcome.UNKNOWN
        try:
            raw = await asyncio.to_thread(
                self._client.fetch_order, order_id, self.symbol)
        except Exception:                                           # noqa: BLE001
            return CancelDispositionOutcome.CANCEL_CONFIRMED
        status = self._to_exchange_order(raw).status
        if status is OrderStatus.FILLED:
            return CancelDispositionOutcome.ALREADY_FILLED
        if status in _TERMINAL_STATUSES:
            return CancelDispositionOutcome.CANCEL_CONFIRMED
        return CancelDispositionOutcome.UNKNOWN

    @override
    async def execute_cancel_all(self, symbol: str | None = None) -> int:
        await self._ensure_broker_started()
        open_orders = await self.get_open_orders(symbol)
        if not open_orders:
            return 0
        # Register every mapped id as expected-to-cancel BEFORE the bulk call,
        # or the pushed CANCELED events read as external and trip quarantine.
        if self.native_cancel_all_expected_sink is not None:
            self.native_cancel_all_expected_sink(symbol)
        await self._venue("cancel:all", "",
                          self._client.cancel_all_orders, self.symbol)
        return len(open_orders)

    # --- BrokerPlugin abstracts: state ---

    @override
    async def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        await self._ensure_broker_started()
        rows = await self._venue("read:open-orders", "",
                                 self._client.fetch_open_orders, self.symbol)
        return [self._to_exchange_order(raw) for raw in rows]

    @override
    async def get_position(self, symbol: str) -> ExchangePosition | None:
        await self._ensure_broker_started()
        manager = self._spot_manager
        if manager is not None:
            mark = self._last_price
            if mark is None:
                vwap = manager.fold.vwap
                mark = float(vwap) if vwap is not None else 0.0
            return manager.synthesize_position(mark)
        # Persistence off (tests): synthesize from the in-memory fallback.
        if self._mem_net <= 0:
            return None
        size = float(self._mem_net)
        vwap = float(self._mem_cost / self._mem_net) if self._mem_net else 0.0
        mark = self._last_price or vwap
        return ExchangePosition(
            symbol=symbol, side='long', size=size, entry_price=vwap,
            unrealized_pnl=(mark - vwap) * size, liquidation_price=None,
            leverage=1.0, margin_mode='cash')

    @override
    async def get_balance(self) -> dict[str, float]:
        balance = await self._venue("read:balance", "", self._client.fetch_balance)
        totals = balance.get('total') or {}
        return {asset: float(amount) for asset, amount in totals.items() if amount}

    # --- live order stream (REST poll) ---

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str):
        bar = await super().watch_ohlcv(symbol, timeframe)
        self._last_price = bar.close
        return bar

    @override
    async def watch_orders(self):
        """Fills from the myTrades cursor; cancels from re-reading tracked ids."""
        await self._ensure_broker_started()
        assert self.config is not None
        cycle = 0
        while True:
            await asyncio.sleep(self.config.poll_interval)
            cycle += 1
            manager = self._spot_manager
            if manager is not None:
                halt = manager.consume_pending_halt()
                if halt is not None:
                    raise halt
            try:
                events = await self._poll_once()
            except Exception as exc:                                # noqa: BLE001
                log.broker_debug("poll -> transient | %s: %s",
                                 type(exc).__name__, exc)
                continue
            if manager is not None and cycle % _RECONCILE_EVERY == 0:
                try:
                    recovered = await manager.reconcile(int(time.time() * 1000))
                except Exception as exc:                            # noqa: BLE001
                    log.broker_warning("%s", f"inventory reconcile failed: "
                                             f"{type(exc).__name__}: {exc}")
                    recovered = []
                events.extend(self._events_for_recovered(recovered))
            for event in events:
                yield event

    async def _poll_once(self) -> list[OrderEvent]:
        events: list[OrderEvent] = []
        # 1) New fills behind the trade-id cursor (feeds the ledger too).
        assert self._trade_cursor is not None
        assert self._spot_port is not None
        trades = await asyncio.to_thread(
            self.fetch_trades_after, self._trade_cursor, 500)
        for trade in trades:
            self._trade_cursor = max(self._trade_cursor, int(trade['id']))
            coid = await asyncio.to_thread(self.client_order_id_for_trade, trade)
            if not coid:
                continue                                # foreign fill
            execution = execution_from_trade(
                trade, base_asset=self._spot_port.base_asset,
                quote_asset=self._spot_port.quote_asset, client_order_id=coid)
            if self._spot_manager is not None:
                if not self._spot_manager.record_live_fill(execution):
                    continue                            # ledger replay
            else:
                self._mem_net += execution.base_delta
                self._mem_cost += (execution.price
                                   * (execution.base_delta
                                      if execution.base_delta > 0 else 0))
                if execution.base_delta < 0:
                    vwap = (self._mem_cost / (self._mem_net - execution.base_delta)
                            if (self._mem_net - execution.base_delta) else 0)
                    self._mem_cost += execution.base_delta * vwap
            event = self._fill_event(trade, coid)
            if event is not None:
                events.append(event)
        # 2) Terminal transitions with no fill row (cancel / reject / expire).
        for order_id in list(self._live_ids):
            raw = await asyncio.to_thread(
                self._client.fetch_order, order_id, self.symbol)
            order = self._to_exchange_order(raw)
            self._orders[order_id] = order
            if order.status not in _TERMINAL_STATUSES:
                continue
            self._live_ids.discard(order_id)
            if order.status is OrderStatus.FILLED:
                continue                    # fills are emitted from the trades path
            pine_id, from_entry, leg_type = self._identity.get(
                order_id, (None, None, None))
            if pine_id is None:
                continue
            events.append(OrderEvent(
                order=order,
                event_type=('cancelled' if order.status is OrderStatus.CANCELLED
                            else 'rejected' if order.status is OrderStatus.REJECTED
                            else 'cancelled'),
                fill_price=None, fill_qty=None, timestamp=time.time(),
                pine_id=pine_id, from_entry=from_entry, leg_type=leg_type))
        return events

    def _fill_event(self, trade: dict, coid: str) -> OrderEvent | None:
        order_id = str(trade.get('order') or '')
        identity = (self._identity.get(order_id)
                    or self._identity_by_coid.get(coid))
        if identity is None:
            return None
        pine_id, from_entry, leg_type = identity
        fill_qty = float(trade['amount'])
        order = self._orders.get(order_id)
        if order is None:
            order = self._to_exchange_order(
                {'id': order_id, 'clientOrderId': coid,
                 'side': trade.get('side'), 'amount': fill_qty,
                 'filled': fill_qty, 'status': 'FILLED'})
            self._orders[order_id] = order
        else:
            order.filled_qty = min(order.filled_qty + fill_qty, order.qty or float('inf'))
            order.remaining_qty = max((order.qty or 0) - order.filled_qty, 0.0)
        step = self.qty_step
        is_complete = (order.qty > 0
                       and order.filled_qty >= order.qty - max(step, 1e-12))
        order.status = (OrderStatus.FILLED if is_complete
                        else OrderStatus.PARTIALLY_FILLED)
        order.average_fill_price = float(trade['price'])
        if is_complete:
            self._live_ids.discard(order_id)
        fee = trade.get('fee') or {}
        return OrderEvent(
            order=order,
            event_type='filled' if is_complete else 'partial',
            fill_price=float(trade['price']), fill_qty=fill_qty,
            timestamp=float(trade['timestamp']) / 1000.0,
            pine_id=pine_id, from_entry=from_entry, leg_type=leg_type,
            fee=float(fee.get('cost') or 0),
            fee_currency=str(fee.get('currency') or ''),
            fill_id=str(trade['id']))

    def _events_for_recovered(self, rows) -> list[OrderEvent]:
        """OrderEvents for ledger rows a runtime catch-up recovered.

        Identity comes from the row's client id; a row minted by a previous
        process instance whose identity this one cannot reconstruct is logged
        and skipped — the ledger (and thus the position) already carries it.
        """
        events: list[OrderEvent] = []
        for row in rows or []:
            coid = str(getattr(row, 'client_order_id', '') or '')
            identity = self._identity_by_coid.get(coid)
            if identity is None:
                log.broker_warning("%s", (
                    f"recovered fill {getattr(row, 'fill_id', '?')} has no "
                    f"in-process identity (coid={coid or '?'}); ledger keeps "
                    f"it, engine event skipped"))
                continue
            pine_id, from_entry, leg_type = identity
            base_delta = getattr(row, 'base_delta', 0)
            fill_qty = abs(float(base_delta))
            price = float(getattr(row, 'price', 0) or 0)
            order_id = str(getattr(row, 'exchange_order_id', '') or coid)
            order = self._orders.get(order_id) or self._to_exchange_order(
                {'id': order_id, 'clientOrderId': coid,
                 'side': 'buy' if float(base_delta) > 0 else 'sell',
                 'amount': fill_qty, 'filled': fill_qty, 'status': 'FILLED'})
            events.append(OrderEvent(
                order=order, event_type='filled',
                fill_price=price or None, fill_qty=fill_qty,
                timestamp=float(getattr(row, 'ts_ms', 0) or 0) / 1000.0 or time.time(),
                pine_id=pine_id, from_entry=from_entry, leg_type=leg_type,
                fill_id=str(getattr(row, 'fill_id', '') or '') or None))
        return events
