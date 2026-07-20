"""
Position tracking for live broker trading.

:class:`BrokerPosition` extends :class:`~pynecore.lib.strategy.PositionBase`
with no simulation logic — the exchange is the source of truth for fills,
prices, fees, and margin state.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from pynecore import lib
from pynecore.core.broker.intent_builder import CLOSE_ALL_EXIT_ID, CLOSE_EXIT_ID_PREFIX
from pynecore.core.broker.models import LegType
from pynecore.lib.log import broker_warning as _blog_warning
from pynecore.lib.strategy import PositionBase, Trade
from pynecore.types.na import na_float

if TYPE_CHECKING:
    from pynecore.lib.strategy import direction
    from pynecore.lib.strategy import Order
    from pynecore.types.strategy import QtyType
    from pynecore.core.broker.models import OrderEvent

__all__ = ['BrokerPosition']


class BrokerPosition(PositionBase):
    """
    Position state tracker for live broker trading.

    The exchange determines fills, prices, fees, and margin state;
    :meth:`record_fill` consumes :class:`OrderEvent` objects emitted by a
    :class:`~pynecore.core.plugin.broker.BrokerPlugin` and updates the
    local view of the position accordingly.

    Trades are tracked FIFO: the first entry filled is the first closed
    when the position is reduced, matching TradingView default semantics.

    Note: margin, liquidation price, and fee currency conversion are all
    handled by the exchange. This class only records what the exchange
    tells it.
    """

    __slots__ = (
        'size', 'sign', 'avg_price',
        'netprofit', 'openprofit', 'grossprofit', 'grossloss',
        'open_commission',
        'eventrades', 'wintrades', 'losstrades',
        'closed_trades_count',
        'max_drawdown', 'max_runup', 'max_equity',
        'open_trades', 'closed_trades', 'new_closed_trades',
        'entry_orders', 'exit_orders',
        # Per-evaluation set of close keys already seen this script run, so a
        # second same-key ``strategy.close()`` THIS evaluation nets onto the
        # first while a next-tick re-issue (calc_on_every_tick) replaces it.
        '_closes_this_eval',
        # === Risk management state (mirrors SimPosition) ===
        # Configuration set by ``strategy.risk.*`` setters:
        'risk_allowed_direction',
        'risk_max_drawdown_value', 'risk_max_drawdown_type', 'risk_max_drawdown_alert',
        'risk_max_intraday_loss_value', 'risk_max_intraday_loss_type',
        'risk_max_intraday_loss_alert',
        'risk_max_cons_loss_days', 'risk_max_cons_loss_days_alert',
        'risk_max_intraday_filled_orders', 'risk_max_intraday_filled_orders_alert',
        'risk_max_position_size',
        # Runtime counters / day-rollover tracking:
        'risk_cons_loss_days', 'risk_last_trading_day', 'risk_last_day_equity',
        'risk_intraday_filled_orders', 'risk_intraday_start_equity',
        'risk_halt_trading',
        '_current_price',
    )

    def __init__(self) -> None:
        self.size: float = 0.0
        self.sign: float = 0.0
        self.avg_price = na_float

        self.netprofit: float = 0.0
        self.openprofit: float = 0.0
        self.grossprofit: float = 0.0
        self.grossloss: float = 0.0
        self.open_commission: float = 0.0

        self.eventrades: int = 0
        self.wintrades: int = 0
        self.losstrades: int = 0
        self.closed_trades_count: int = 0
        self.max_drawdown: float = 0.0
        self.max_runup: float = 0.0
        # Mark-to-market peak equity, used by ``_peak_equity`` for the
        # ``max_drawdown(percent_of_equity)`` threshold. Updated on every
        # :meth:`update_unrealized_pnl` and :meth:`record_fill` call.
        self.max_equity: float = -float("inf")

        self.open_trades: list[Trade] = []
        self.closed_trades: deque[Trade] = deque(maxlen=9000)
        self.new_closed_trades: list[Trade] = []

        self.entry_orders: dict[str | None, 'Order'] = {}
        # Composite key ``(exit_id, from_entry)`` mirrors
        # :class:`~pynecore.lib.strategy.SimPosition.exit_orders`. Single-field
        # keys collide on partial-TP fan-out (multiple exits for one entry)
        # and on ``from_entry=na`` fan-out (one exit_id, many per-entry rows).
        self.exit_orders: dict[tuple[str | None, str | None], 'Order'] = {}

        # Close keys (``(exit_id, order_id)``) already issued in the current
        # script evaluation; reset by :meth:`begin_evaluation`. See
        # :meth:`_add_order` for the same-eval netting it drives.
        self._closes_this_eval: set[tuple[str | None, str | None]] = set()

        # === Risk management state ===
        # Configuration (filled by the ``strategy.risk.*`` setters via
        # ``__init__.py``'s shared lib-property bridge):
        self.risk_allowed_direction: 'direction.Direction | None' = None
        self.risk_max_drawdown_value: float | None = None
        self.risk_max_drawdown_type: 'QtyType | None' = None
        self.risk_max_drawdown_alert: str | None = None
        self.risk_max_intraday_loss_value: float | None = None
        self.risk_max_intraday_loss_type: 'QtyType | None' = None
        self.risk_max_intraday_loss_alert: str | None = None
        self.risk_max_cons_loss_days: int | None = None
        self.risk_max_cons_loss_days_alert: str | None = None
        self.risk_max_intraday_filled_orders: int | None = None
        self.risk_max_intraday_filled_orders_alert: str | None = None
        self.risk_max_position_size: float | None = None
        # Runtime counters / day-rollover tracking:
        self.risk_cons_loss_days: int = 0
        self.risk_last_trading_day: int = -1
        self.risk_last_day_equity: float = 0.0
        self.risk_intraday_filled_orders: int = 0
        self.risk_intraday_start_equity: float = 0.0
        self.risk_halt_trading: bool = False

        self._current_price: float = 0.0
        # Inherited from PositionBase; unused on the live path (close stacking is
        # backtest-only), initialized so _next_close_seq() never raises if called.
        self._close_seq_counter: int = 0

    # === Pine API compatibility shims ======================================
    # Pine strategy.* functions read ``position.c`` / ``.o`` / ``.h`` / ``.l``
    # for the simulator's creation-time margin check. In broker mode those
    # attributes are served from the live OHLCV module; the exchange enforces
    # margin for real, so the Pine-level check still acts as a safety net
    # on script-side state without a separate simulator update path.

    @property
    def c(self) -> float:
        try:
            v = lib.close
        except AttributeError:
            return self._current_price or 0.0
        try:
            return float(v) if v is not None else self._current_price or 0.0
        except (TypeError, ValueError):
            return self._current_price or 0.0

    @property
    def o(self) -> float:
        try:
            return float(lib.open)
        except (AttributeError, TypeError, ValueError):
            return self.c

    @property
    def h(self) -> float:
        try:
            return float(lib.high)
        except (AttributeError, TypeError, ValueError):
            return self.c

    @property
    def l(self) -> float:  # noqa: E743 — mirrors the Pine attribute name
        try:
            return float(lib.low)
        except (AttributeError, TypeError, ValueError):
            return self.c

    # === Pine-side order book ===

    def begin_evaluation(self) -> None:
        """Mark the start of a fresh script evaluation (one ``main()`` run).

        Called by the runner before the libraries / ``main`` execute, in broker
        mode only. It clears the per-evaluation close-key set so that two
        ``strategy.close()`` calls issued in the SAME evaluation net onto one
        order, while the SAME close re-issued on the next ``calc_on_every_tick``
        evaluation replaces the pending order instead of doubling it. There is
        no other per-evaluation order-book reset in live mode (the Pine order
        book is purely event-driven), so this is the netting's idempotency
        anchor.
        """
        self._closes_this_eval.clear()

    def _add_order(self, order: 'Order') -> None:
        """Register an order locally (the sync engine forwards it to the exchange).

        Pre-submit risk gates run before the order is enqueued — same policy
        as :meth:`SimPosition.fill_order` enforces at fill time, but applied
        at the submit boundary because the broker fill is asynchronous.
        Rejected entry/normal orders are silently dropped (matching the sim
        ``_remove_order`` behavior on cap/direction violation); the
        :attr:`risk_halt_trading` flag is set out-of-band by
        :meth:`_enforce_post_bar_risk`.
        """
        order.bar_index = int(lib.bar_index)
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import (
            _order_type_close, _order_type_entry, _order_type_normal,
        )
        if order.order_type in (_order_type_entry, _order_type_normal):
            if self._is_intraday_filled_cap_reached():
                return
            adjusted = self._adjust_for_max_position_size(float(order.size), order.sign)
            if adjusted is None:
                return
            order.size = adjusted
            if self.size == 0.0 and not self._is_direction_allowed(order.sign):
                return
        if order.order_type == _order_type_close:
            key = (order.exit_id, order.order_id)
            existing = self.exit_orders.get(key)
            # Netting is for market closes only (``strategy.close(id)`` /
            # ``strategy.close_all()``), identified by their reserved exit-id
            # patterns. A sticky ``strategy.exit`` bracket re-issued in the SAME
            # evaluation must still REPLACE: summing its size would dispatch an
            # oversized protective order and the first leg's stale limit/stop/
            # trailing levels would survive (netting only carries metadata).
            exit_id = order.exit_id
            is_market_close = exit_id == CLOSE_ALL_EXIT_ID or (
                exit_id is not None and exit_id.startswith(CLOSE_EXIT_ID_PREFIX)
            )
            if is_market_close and key in self._closes_this_eval and existing is not None:
                # Second+ same-key close THIS evaluation: net the slices into
                # one reduce-only market close. Both ``strategy.close`` qty
                # expressions are evaluated against the same ``position.size``
                # (no fill lands mid-evaluation), so the slices simply sum; the
                # over-close cap is applied later by the sync engine. Metadata
                # is last-wins, matching the prior overwrite behaviour for this
                # collision class. A NEXT-evaluation re-issue takes the ``else``
                # branch (``begin_evaluation`` cleared the key) and replaces the
                # pending order, keeping calc_on_every_tick idempotent.
                existing.size += order.size
                existing.reserved_size = abs(existing.size)
                existing.comment = order.comment
                existing.alert_message = order.alert_message
            else:
                self._closes_this_eval.add(key)
                self.exit_orders[key] = order
        else:
            self.entry_orders[order.order_id] = order

    def _remove_order(self, order: 'Order') -> None:
        """Cancel an order locally."""
        order.cancelled = True
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import _order_type_close
        if order.order_type == _order_type_close:
            self.exit_orders.pop((order.exit_id, order.order_id), None)
        else:
            self.entry_orders.pop(order.order_id, None)

    def _remove_order_by_id(self, order_id: str) -> None:
        # TV-verified semantics: ``strategy.cancel(id)`` matches an exit by
        # its ``exit_id`` and an entry by its entry id; no cross-matching.
        for exit_order in list(self.exit_orders.values()):
            if exit_order.exit_id == order_id:
                self._remove_order(exit_order)
        entry = self.entry_orders.get(order_id)
        if entry is not None:
            self._remove_order(entry)

    def _cancel_all_orders(self) -> None:
        # No ``orderbook`` attribute — that lives on ``SimPosition`` and drives
        # the simulator's price-keyed fill loop, which has no analog in live
        # trading. Clearing the two Pine-side dicts is enough; the next
        # ``OrderSyncEngine.sync()`` diffs against ``_active_intents`` and
        # dispatches a per-id cancel for every previously tracked intent.
        self.entry_orders.clear()
        self.exit_orders.clear()

    # === Restart-time Pine-side reconstruction ===

    def reconstruct_exit_order(
            self,
            *,
            pine_id: str,
            from_entry: str,
            side: str,
            qty: float,
            tp_price: float | None,
            sl_price: float | None,
            trail_price: float | None,
            trail_offset: float | None,
            oca_name: str | None = None,
            oca_type: str | None = None,
    ) -> None:
        """Re-install a persistent ``strategy.exit`` bracket order after a restart.

        Pine ``strategy.exit`` orders are persistent: once placed they live in
        :attr:`exit_orders` across bars (the script need not re-emit them) until
        they fill or are cancelled. A fresh process starts with empty order
        dicts, so the bracket the previous run armed is invisible to
        :func:`~pynecore.core.broker.intent_builder.build_intents` until it is
        rebuilt here from the durable broker-side ledger (the one-way
        bracket-ownership rows or the engine-trigger partial-leg ledger). The
        :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine` calls this
        once at startup so the diff sees the same exit it saw before the crash
        and adopts (rather than tears down) the live broker protection.

        ``side`` is the CLOSE side (``"buy"``/``"sell"``); the stored
        :class:`~pynecore.lib.strategy.Order` carries the opposite-of-position
        signed size so ``build_intents`` re-derives the same side. Tick fields
        are left ``None`` — the ledger persists resolved absolute prices, never
        the original tick distances.

        :param pine_id: The ``strategy.exit(id=...)`` value (the exit id).
        :param from_entry: The parent entry id the exit protects.
        :param side: The exit (close) side, ``"buy"`` or ``"sell"``.
        :param qty: Exit quantity magnitude.
        :param tp_price: Absolute take-profit price, or ``None``.
        :param sl_price: Absolute stop-loss price, or ``None``.
        :param trail_price: Absolute trailing-stop activation price, or ``None``.
        :param trail_offset: Trailing-stop offset (price units), or ``None``.
        :param oca_name: OCA group name, or ``None``.
        :param oca_type: OCA type string (``"reduce"`` / ``"cancel"`` /
            ``"none"``), or ``None`` for rows persisted before the OCA fields
            existed. Restored so ``build_intents`` re-derives the same group the
            exit was emitted under and the cross-bracket OCA-cancel cascade keeps
            firing across the restart.
        """
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import Order, _order_type_close, oca as _oca
        signed_size = qty if side == "buy" else -qty
        order = Order(
            from_entry,
            signed_size,
            order_type=_order_type_close,
            exit_id=pine_id,
            limit=tp_price,
            stop=sl_price,
            trail_price=trail_price,
            trail_offset=trail_offset,
            oca_name=oca_name,
            oca_type=_oca.Oca(oca_type) if oca_type is not None else None,
        )
        self.exit_orders[(pine_id, from_entry)] = order

    def reconstruct_parent_trade(
            self,
            *,
            entry_id: str,
            size: float,
            entry_price: float,
    ) -> None:
        """Seed one open :class:`~pynecore.lib.strategy.Trade` for an adopted parent.

        Startup adoption restores :attr:`size`/:attr:`avg_price` but leaves
        :attr:`open_trades` empty. A partial-quantity bracket needs the parent's
        open qty to classify
        :attr:`~pynecore.core.broker.models.ExitIntent.is_partial_qty_bracket`
        consistently across bars (``build_intents`` derives the parent total
        from :attr:`open_trades`); without it the exit would be misclassified as
        a whole-row bracket and adopted on the wrong dispatch path. The trade is
        inert until a real broker fill routes through :meth:`record_fill` (the
        sole :attr:`open_trades` mutator), so seeding it cannot emit an intent or
        re-open anything.

        :param entry_id: The parent entry id (``from_entry``).
        :param size: Signed parent open size (positive long, negative short).
        :param entry_price: Parent average entry price.
        """
        if any(t.entry_id == entry_id for t in self.open_trades):
            return
        # ``entry_equity`` stays 0.0: the parent opened in a prior process so
        # its true entry equity is unknowable, and startup adoption runs in
        # ``start_broker`` before the script is attached (``self.equity`` reads
        # ``lib._script``, not yet set). The broker close path never divides by
        # a trade's entry equity, so the placeholder is inert.
        self.open_trades.append(Trade(
            size=size,
            entry_id=entry_id,
            entry_bar_index=int(getattr(lib, 'bar_index', 0)),
            entry_time=0,
            entry_price=entry_price,
            commission=0.0,
            entry_equity=0.0,
        ))

    # === Exchange-side state updates ===

    def record_fill(self, event: 'OrderEvent') -> bool:
        """
        Record an exchange fill.

        :param event: An :class:`OrderEvent` with ``fill_qty`` and
            ``fill_price`` populated, plus Pine identity fields
            (``pine_id``, ``from_entry``, ``leg_type``) filled by the plugin.
        :return: ``True`` if the position side changed as a result of this fill.
        """
        fill_qty = event.fill_qty or 0.0
        fill_price = event.fill_price or 0.0
        if fill_qty <= 0.0 or fill_price <= 0.0:
            # A zero-or-missing qty/price means the broker plugin emitted a
            # fill event without resolving the actual fill quantity / price.
            # Without a warning the silent skip leaves ``position.size``
            # stuck at zero — the script keeps thinking it is flat and
            # re-fires the entry next bar.  Surface the offending event so
            # the operator can spot the broker-side data hole instead of
            # debugging by guesswork.
            _blog_warning(
                "ignoring fill with zero qty/price (qty=%s price=%s pine=%r leg=%s) "
                "— position.size NOT updated",
                fill_qty, fill_price,
                event.pine_id, event.leg_type,
            )
            return False

        signed_delta = fill_qty if event.order.side == "buy" else -fill_qty
        old_sign = self.sign
        new_size = self.size + signed_delta

        # Commission bookkeeping — realized fee becomes part of net P&L at close
        fee = event.fee

        # Risk management: count every filled order toward the intraday cap,
        # matching ``SimPosition._fill_order``. The counter is read by the
        # pre-submit gate in :meth:`_add_order` and the post-bar halt in
        # :meth:`_enforce_post_bar_risk`.
        self.risk_intraday_filled_orders += 1

        if self.size == 0.0 or (old_sign * signed_delta) > 0.0:
            # Opening or adding to an existing position (same direction)
            new_abs = abs(new_size)
            old_abs = abs(self.size)
            if old_abs == 0.0 or self.avg_price is na_float:
                self.avg_price = fill_price
            else:
                self.avg_price = (self.avg_price * old_abs + fill_price * fill_qty) / new_abs
            self.size = new_size
            self.sign = 1.0 if new_size > 0.0 else (-1.0 if new_size < 0.0 else 0.0)

            trade = Trade(
                size=signed_delta,
                entry_id=event.pine_id,
                entry_bar_index=int(getattr(lib, 'bar_index', 0)),
                entry_time=int(event.timestamp * 1000.0),
                entry_price=fill_price,
                commission=fee,
                entry_comment=None,
                entry_equity=self.equity,
            )
            self.open_trades.append(trade)
            self.open_commission += fee
            # The entry Order stays in ``entry_orders`` for intent stability, but
            # its filled slice now lives in ``open_trades``. Record how much of it
            # has filled so ``strategy.exit``'s bound-size reservation does not
            # count the same quantity twice (issue BYBIT-001).
            entry_order = self.entry_orders.get(event.pine_id)
            if entry_order is not None:
                entry_order.filled_qty += fill_qty
            return False

        # Reducing or flipping — FIFO close of existing trades
        remaining = fill_qty
        closed_profit = 0.0
        closed_fee = 0.0
        while remaining > 0.0 and self.open_trades:
            trade = self.open_trades[0]
            trade_abs = abs(trade.size)
            if trade_abs <= remaining + 1e-12:
                # Close this trade fully
                self._close_trade(trade, fill_price, event, fee_share=fee * (trade_abs / fill_qty))
                closed_profit += trade.profit
                closed_fee += trade.commission
                remaining -= trade_abs
            else:
                # Partial close: split the trade
                closed_piece = Trade(
                    size=trade.sign * remaining,
                    entry_id=trade.entry_id,
                    entry_bar_index=trade.entry_bar_index,
                    entry_time=trade.entry_time,
                    entry_price=trade.entry_price,
                    commission=trade.commission * (remaining / trade_abs),
                    entry_comment=trade.entry_comment,
                    entry_equity=trade.entry_equity,
                )
                self._close_trade(
                    closed_piece, fill_price, event,
                    fee_share=fee * (remaining / fill_qty),
                )
                closed_profit += closed_piece.profit
                # Shrink the remaining open trade
                trade.size -= closed_piece.size
                trade.commission -= closed_piece.commission
                remaining = 0.0

        self.size += signed_delta
        # Clamp tiny residuals to zero
        if abs(self.size) < 1e-12:
            self.size = 0.0
            self.sign = 0.0
            self.avg_price = na_float
        else:
            self.sign = 1.0 if self.size > 0.0 else -1.0

        # If there is leftover qty after closing all open_trades → side flip.
        # On a one-way account only an ENTRY leg may legitimately open the
        # opposite side (stop-and-reverse). A reduce-only/exit leg (close, TP,
        # SL, trailing) must NEVER flip: leftover qty means the local FIFO
        # under-counted exposure (e.g. a restart adopted the net size without
        # seeding open_trades). The authoritative net is already in ``self.size``
        # (``self.size += signed_delta`` above), so two cases split apart:
        #   * the net kept its original sign → the exit only PARTIALLY reduced;
        #     the leftover is a stale FIFO row shortfall, NOT an over-close.
        #     Keep the (already-correct) residual size — clamping to flat would
        #     drop live broker exposure and let the script re-fire the entry.
        #   * the net reached or crossed zero → the exit closed at least the
        #     whole position; reconcile toward flat instead of fabricating an
        #     opposite position the diff engine could then "close" with a real
        #     reversing order.
        if remaining > 0.0:
            if event.leg_type in (
                LegType.CLOSE, LegType.TAKE_PROFIT,
                LegType.STOP_LOSS, LegType.TRAILING_STOP,
            ):
                if old_sign * self.size > 0.0:
                    # Partial reduce-only fill against an under-counted FIFO:
                    # the residual already in ``self.size`` is authoritative.
                    _blog_warning(
                        "exit fill (leg=%s qty=%s) under-counted FIFO exposure "
                        "by %s — keeping residual size %s (partial reduce, "
                        "not an over-close)",
                        event.leg_type, fill_qty, remaining, self.size,
                    )
                else:
                    _blog_warning(
                        "exit fill (leg=%s qty=%s) exceeded known FIFO exposure "
                        "by %s — clamping to flat instead of opening an opposite "
                        "position",
                        event.leg_type, fill_qty, remaining,
                    )
                    self.size = 0.0
                    self.sign = 0.0
                    self.avg_price = na_float
            else:
                new_size = self.sign * remaining if self.sign != 0.0 else signed_delta
                self.size = new_size
                self.sign = 1.0 if new_size > 0.0 else (-1.0 if new_size < 0.0 else 0.0)
                self.avg_price = fill_price
                flipped = Trade(
                    size=new_size,
                    entry_id=event.pine_id,
                    entry_bar_index=int(getattr(lib, 'bar_index', 0)),
                    entry_time=int(event.timestamp * 1000.0),
                    entry_price=fill_price,
                    commission=0.0,
                    entry_comment=None,
                    entry_equity=self.equity,
                )
                self.open_trades.append(flipped)

        # Update running stats
        self.netprofit += closed_profit
        if closed_profit > 0.0:
            self.grossprofit += closed_profit
            self.wintrades += 1
        elif closed_profit < 0.0:
            self.grossloss += closed_profit
            self.losstrades += 1
        else:
            self.eventrades += 1

        self.open_commission = float(sum(t.commission for t in self.open_trades))

        return self.sign != old_sign

    def update_unrealized_pnl(self, current_price: float) -> None:
        """Mark-to-market: recompute :attr:`openprofit` at the given price.

        Also rolls the :attr:`max_equity` peak forward and the
        :attr:`max_drawdown` running maximum off the live price — these feed
        :meth:`_peak_equity` and :meth:`_is_max_drawdown_breached` so the
        broker risk gates see real-time equity, not just realized P&L.
        """
        self._current_price = current_price
        if not self.open_trades or current_price <= 0.0:
            self.openprofit = 0.0
        else:
            total = 0.0
            for trade in self.open_trades:
                total += (current_price - trade.entry_price) * trade.size
            self.openprofit = total
        eq = float(self.equity)
        if eq > self.max_equity:
            self.max_equity = eq
        # Drawdown is measured from the running peak — same metric the sim
        # ``max_drawdown`` field tracks, just sourced from mark-to-market.
        if self.max_equity > -float("inf"):
            dd = self.max_equity - eq
            if dd > self.max_drawdown:
                self.max_drawdown = dd

    def _peak_equity(self) -> float:
        """Reference equity for ``max_drawdown(percent_of_equity)``.

        Falls back to initial capital before the first
        :meth:`update_unrealized_pnl` (or fill) primes ``max_equity``.
        """
        # noinspection PyProtectedMember
        initial = float(lib._script.initial_capital)
        if self.max_equity == -float("inf"):
            return initial
        return max(initial, float(self.max_equity))

    # === Risk management hooks (called by the runner once per bar) =========

    def _handle_bar_open_risk(self) -> None:
        """Day-rollover bookkeeping at the start of each bar.

        Mirrors the rollover block in
        :meth:`SimPosition._process_at_bar_open`: on a new trading day,
        update ``risk_cons_loss_days`` from the prior-day equity delta,
        reset the intraday anchors, and immediately halt if the
        consecutive-loss-day cap is breached so queued entries cannot fill
        at this bar's open. Distinct name from the sim hook because the
        sim variant takes an ``ohlc`` argument and runs additional
        simulator-specific work — the broker version only does
        risk-related rollover.
        """
        if self.risk_halt_trading:
            return
        try:
            # Statically a value (module_property), at runtime still the function
            current_trading_day = int(lib.time_tradingday())
        except (AttributeError, TypeError, ValueError):
            return
        if current_trading_day == self.risk_last_trading_day:
            return
        current_equity = float(self.equity)
        # On the very first bar there is no prior day to compare against —
        # initialise the trailing-equity anchor without touching the counter.
        if self.risk_last_trading_day != -1:
            if current_equity < self.risk_last_day_equity:
                self.risk_cons_loss_days += 1
            else:
                self.risk_cons_loss_days = 0
        self.risk_last_day_equity = current_equity
        self.risk_intraday_start_equity = current_equity
        self.risk_last_trading_day = current_trading_day
        self.risk_intraday_filled_orders = 0
        if self._is_max_cons_loss_days_breached():
            self._trigger_risk_halt("Max consecutive loss days reached")

    def _enforce_post_bar_risk(self) -> None:
        """Run the post-bar ``strategy.risk.*`` checks.

        Called by the runner after the script executes and before the next
        :meth:`OrderSyncEngine.sync` so the queued risk-close goes out in
        the same dispatch cycle. The first triggered rule wins.
        """
        if self.risk_halt_trading:
            return
        if self._is_max_drawdown_breached():
            self._trigger_risk_halt("Max drawdown reached")
            return
        if self._is_max_intraday_loss_breached():
            self._trigger_risk_halt("Max intraday loss reached")
            return
        if self._is_max_cons_loss_days_breached():
            self._trigger_risk_halt("Max consecutive loss days reached")

    def _trigger_risk_halt(self, reason: str) -> None:
        """Cancel pending orders, queue a market close, set the halt flag.

        Differs from :meth:`SimPosition._trigger_risk_halt` in two ways: no
        synthetic fill (the exchange owns fills, not the position), and no
        OHLC arguments (the close is a plain market order, the broker
        decides the fill price). The next :meth:`OrderSyncEngine.sync`
        observes the cleared books plus the queued close and dispatches
        accordingly.
        """
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import Order, _order_type_close
        self.entry_orders.clear()
        self.exit_orders.clear()
        if self.size != 0.0:
            close_order = Order(
                None, -self.size,
                exit_id='Risk management close',
                order_type=_order_type_close,
                comment=f"Close Position ({reason})",
            )
            self._add_order(close_order)
        self.risk_halt_trading = True

    def record_liquidation(self, event: 'OrderEvent') -> None:
        """Record an exchange-initiated liquidation — close all open trades."""
        if not self.open_trades:
            return
        fill_price = event.fill_price or 0.0
        for trade in list(self.open_trades):
            self._close_trade(trade, fill_price, event, fee_share=event.fee / max(len(self.open_trades), 1))
            self.netprofit += trade.profit
            if trade.profit > 0.0:
                self.grossprofit += trade.profit
                self.wintrades += 1
            elif trade.profit < 0.0:
                self.grossloss += trade.profit
                self.losstrades += 1
            else:
                self.eventrades += 1
        self.size = 0.0
        self.sign = 0.0
        self.avg_price = na_float
        self.openprofit = 0.0
        self.open_commission = 0.0

    # === Internals ===

    def _close_trade(self, trade: Trade, fill_price: float,
                     event: 'OrderEvent', fee_share: float) -> None:
        """Move a (possibly split) Trade from open_trades to closed_trades."""
        trade.exit_id = event.pine_id or ""
        trade.exit_bar_index = int(getattr(lib, 'bar_index', 0))
        trade.exit_time = int(event.timestamp * 1000.0)
        trade.exit_price = fill_price
        trade.exit_comment = ''
        trade.commission += fee_share
        trade.profit = (fill_price - trade.entry_price) * trade.size - trade.commission
        trade.exit_equity = self.equity + trade.profit
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        self.new_closed_trades.append(trade)
        self.closed_trades_count += 1
