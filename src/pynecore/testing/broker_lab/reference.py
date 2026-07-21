"""Small reference venue used by examples and core conformance scenarios."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pynecore.core.broker.exceptions import (
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    OrderSkippedByPlugin,
)
from pynecore.core.broker.models import (
    CapabilityLevel,
    DispatchEnvelope,
    ExchangeCapabilities,
    ExchangeOrder,
    ExchangePosition,
    INTENT_KEY_SEP,
    LegType,
    OrderEvent,
    OrderStatus,
    OrderType,
)

from .model import Step


@dataclass
class VenueOrder:
    """Reference venue ownership record."""

    order: ExchangeOrder
    run_name: str
    pine_id: str
    leg_type: LegType
    intent_key: str
    from_entry: str | None = None


@dataclass
class VenueState:
    """Shared account state for one or more reference brokers."""

    symbol: str = "LABUSD"
    orders: dict[str, VenueOrder] = field(default_factory=dict)
    position: float = 0.0
    position_owners: dict[str, float] = field(default_factory=dict)
    position_legs: dict[tuple[str, str], float] = field(default_factory=dict)
    pending_events: list[tuple[str, OrderEvent]] = field(default_factory=list)
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    next_id: int = 1
    reject_entries: int = 0
    skip_entries: int = 0
    read_error: Exception | None = None
    stale_position: float | None = None
    terminal_snapshots: dict[str, tuple[OrderStatus, float, float]] = field(
        default_factory=dict
    )

    def new_id(self) -> str:
        value = f"lab-{self.next_id}"
        self.next_id += 1
        return value


class ReferenceBroker:
    """BrokerPlugin-shaped adapter over :class:`VenueState`."""

    client_order_id_max_len = 30
    on_unexpected_cancel = "stop"
    position_port = None

    def __init__(self, profile: "ReferenceVenueProfile", run_name: str) -> None:
        self.profile = profile
        self.state = profile.state
        self.run_name = run_name
        self.store_ctx: Any = None

    def get_capabilities(self) -> ExchangeCapabilities:
        return ExchangeCapabilities(short_selling=CapabilityLevel.NATIVE)

    def _place(
        self, envelope: DispatchEnvelope, leg: LegType, kind: str
    ) -> list[ExchangeOrder]:
        self.state.calls.append((self.run_name, kind, envelope.intent.intent_key))
        if leg is LegType.ENTRY and self.state.skip_entries:
            self.state.skip_entries -= 1
            raise OrderSkippedByPlugin(
                "reference profile skipped entry",
                intent_key=envelope.intent.intent_key,
                reason="below_min_size",
            )
        if leg is LegType.ENTRY and self.state.reject_entries:
            self.state.reject_entries -= 1
            raise ExchangeOrderRejectedError("reference profile rejected entry")
        intent = envelope.intent
        order = ExchangeOrder(
            id=self.state.new_id(),
            symbol=self.profile.wire_symbol,
            side=getattr(intent, "side", "buy"),
            order_type=(
                OrderType.STOP
                if getattr(intent, "stop", None) is not None
                else (
                    OrderType.LIMIT
                    if getattr(intent, "limit", None) is not None
                    else OrderType.MARKET
                )
            ),
            qty=float(getattr(intent, "qty", 0.0)),
            filled_qty=0.0,
            remaining_qty=float(getattr(intent, "qty", 0.0)),
            price=getattr(intent, "limit", None),
            stop_price=getattr(intent, "stop", None),
            average_fill_price=None,
            status=OrderStatus.OPEN,
            timestamp=0.0,
            fee=0.0,
            fee_currency="",
            client_order_id=envelope.client_order_id(kind),
        )
        pine_id = str(getattr(intent, "pine_id", ""))
        if self.store_ctx is not None:
            self.store_ctx.upsert_order(
                order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                state="confirmed",
                intent_key=intent.intent_key,
                exchange_order_id=order.id,
                from_entry=getattr(intent, "from_entry", None),
                pine_entry_id=pine_id or None,
                filled_qty=0.0,
            )
        self.state.orders[order.id] = VenueOrder(
            order=order,
            run_name=self.run_name,
            pine_id=pine_id,
            leg_type=leg,
            intent_key=intent.intent_key,
            from_entry=getattr(intent, "from_entry", None),
        )
        return [order]

    async def execute_entry(self, envelope: DispatchEnvelope) -> list[ExchangeOrder]:
        return self._place(envelope, LegType.ENTRY, "e")

    async def execute_exit(self, envelope: DispatchEnvelope) -> list[ExchangeOrder]:
        intent = envelope.intent
        limit = getattr(intent, "limit", None)
        stop = getattr(intent, "stop", None)
        orders: list[ExchangeOrder] = []
        if limit is not None:
            take_profit = self._place(envelope, LegType.TAKE_PROFIT, "t")[0]
            take_profit = replace(
                take_profit,
                order_type=OrderType.LIMIT,
                price=limit,
                stop_price=None,
            )
            self.state.orders[take_profit.id].order = take_profit
            orders.append(take_profit)
        if stop is not None:
            stop_loss = self._place(envelope, LegType.STOP_LOSS, "s")[0]
            stop_loss = replace(
                stop_loss,
                order_type=OrderType.STOP,
                price=None,
                stop_price=stop,
            )
            self.state.orders[stop_loss.id].order = stop_loss
            orders.append(stop_loss)
        return orders or self._place(envelope, LegType.TAKE_PROFIT, "t")

    async def execute_close(self, envelope: DispatchEnvelope) -> ExchangeOrder:
        return self._place(envelope, LegType.CLOSE, "c")[0]

    async def execute_cancel(self, envelope: DispatchEnvelope) -> bool:
        self.state.calls.append((self.run_name, "cancel", envelope.intent.intent_key))
        for venue_order in self.state.orders.values():
            if (
                venue_order.run_name == self.run_name
                and venue_order.intent_key == envelope.intent.intent_key
                and venue_order.order.status
                in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
            ):
                venue_order.order = replace(
                    venue_order.order,
                    status=OrderStatus.CANCELLED,
                    remaining_qty=0.0,
                )
        return True

    async def modify_entry(
        self,
        old: DispatchEnvelope,
        new: DispatchEnvelope,
    ) -> list[ExchangeOrder]:
        await self.execute_cancel(old)
        return await self.execute_entry(new)

    async def modify_exit(
        self,
        old: DispatchEnvelope,
        new: DispatchEnvelope,
    ) -> list[ExchangeOrder]:
        await self.execute_cancel(old)
        return await self.execute_exit(new)

    async def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        if self.state.read_error is not None:
            error = self.state.read_error
            self.state.read_error = None
            raise error
        return [
            record.order
            for record in self.state.orders.values()
            if record.run_name == self.run_name
            and record.order.status is OrderStatus.OPEN
            and (
                symbol is None
                or symbol == self.profile.symbol
                or record.order.symbol == symbol
            )
        ]

    async def get_position(self, symbol: str) -> ExchangePosition | None:
        if self.state.read_error is not None:
            error = self.state.read_error
            self.state.read_error = None
            raise error
        size = (
            self.state.position
            if self.state.stale_position is None
            else self.state.stale_position
        )
        if size == 0.0:
            return None
        return ExchangePosition(
            symbol=symbol,
            side="long" if size > 0 else "short",
            size=abs(size),
            entry_price=100.0,
            unrealized_pnl=0.0,
            liquidation_price=None,
            leverage=1.0,
            margin_mode="cross",
        )

    def watch_orders(self):
        raise NotImplementedError


class ReferenceVenueProfile:
    """Deterministic netting profile suitable for copied plugin examples."""

    plugin_name = "broker-lab-reference"
    account_id = "offline-account"
    symbol = "LABUSD"
    wire_symbol = "LABUSD"
    timeframe = "1"
    quantity_step = 1.0
    venue_mode = "netting"

    def __init__(self) -> None:
        self.state = VenueState(symbol=self.symbol)

    def create_broker(self, run_name: str, store_ctx: Any) -> ReferenceBroker:
        broker = ReferenceBroker(self, run_name)
        broker.store_ctx = store_ctx
        return broker

    def handle_step(self, runner: Any, step: Step) -> bool:
        if step.kind == "fill":
            self._fill(runner, step)
        elif step.kind in ("deliver", "delayed_deliver"):
            self._deliver(runner, step)
        elif step.kind == "duplicate_event":
            if not self.state.pending_events:
                raise ValueError("duplicate_event requires a pending event")
            self.state.pending_events.append(self.state.pending_events[-1])
        elif step.kind == "reorder_events":
            self.state.pending_events.reverse()
        elif step.kind == "drop_events":
            self.state.pending_events.clear()
        elif step.kind == "reject_entries":
            self.state.reject_entries = int(step.values.get("count", 1))
        elif step.kind == "skip_entries":
            self.state.skip_entries = int(step.values.get("count", 1))
        elif step.kind == "read_error":
            self.state.read_error = ExchangeConnectionError("injected read failure")
        elif step.kind == "stale_position":
            self.state.stale_position = float(step.values.get("position", 0.0))
        elif step.kind == "fresh_position":
            self.state.stale_position = None
        elif step.kind == "venue_cancel":
            self._venue_cancel(step)
        elif step.kind == "expect":
            self._expect(runner, step)
        else:
            return False
        return True

    def _select_order(self, step: Step) -> VenueOrder:
        order_id = step.values.get("order_id")
        candidates = [
            record
            for record in self.state.orders.values()
            if record.run_name == step.run
            and (order_id is None or record.order.id == order_id)
            and record.order.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
        ]
        if not candidates:
            raise ValueError(f"no open venue order for run {step.run!r}")
        return candidates[-1]

    def _fill(self, runner: Any, step: Step) -> None:
        record = self._select_order(step)
        qty = float(step.values.get("qty", record.order.remaining_qty))
        if qty <= 0.0 or qty > record.order.remaining_qty + 1e-12:
            raise ValueError(
                "fill qty must be positive and at most the remaining quantity"
            )
        remaining = max(0.0, record.order.remaining_qty - qty)
        filled = record.order.filled_qty + qty
        status = (
            OrderStatus.FILLED if remaining == 0.0 else OrderStatus.PARTIALLY_FILLED
        )
        price = float(step.values.get("price", 100.0))
        record.order = replace(
            record.order,
            filled_qty=filled,
            remaining_qty=remaining,
            average_fill_price=price,
            status=status,
        )
        runtime = runner.runs[step.run]
        runtime.store_ctx.set_filled(record.order.client_order_id, filled)
        signed = qty if record.order.side == "buy" else -qty
        if record.leg_type is LegType.ENTRY:
            self.state.position_owners[record.run_name] = (
                self.state.position_owners.get(record.run_name, 0.0) + signed
            )
            if self.venue_mode == "hedged":
                key = (record.run_name, record.pine_id)
                self.state.position_legs[key] = (
                    self.state.position_legs.get(key, 0.0) + signed
                )
        else:
            current = self.state.position_owners.get(record.run_name, 0.0)
            new_size = current + signed
            if current != 0.0 and current * new_size < 0.0:
                new_size = 0.0
            self.state.position_owners[record.run_name] = new_size
            if self.venue_mode == "hedged" and record.from_entry is not None:
                key = (record.run_name, record.from_entry)
                leg_size = self.state.position_legs.get(key, 0.0)
                reduced = leg_size + signed
                self.state.position_legs[key] = (
                    0.0 if leg_size * reduced < 0.0 else reduced
                )
        self.state.position = sum(self.state.position_owners.values())
        event = OrderEvent(
            order=record.order,
            event_type="filled" if remaining == 0.0 else "partial",
            fill_price=price,
            fill_qty=qty,
            timestamp=runner.now_ms / 1000.0,
            pine_id=record.pine_id,
            leg_type=record.leg_type,
            fill_id=str(step.values.get("fill_id", f"{record.order.id}:{filled}")),
        )
        self.state.pending_events.append((record.run_name, event))

    def _deliver(self, runner: Any, step: Step) -> None:
        selected = [item for item in self.state.pending_events if item[0] == step.run]
        self.state.pending_events = [
            item for item in self.state.pending_events if item[0] != step.run
        ]

        def dispatch() -> None:
            for _, event in selected:
                runtime = runner.runs[step.run]
                runtime.engine.on_order_event(event)
            if selected:
                runner.runs[step.run].engine.apply_async_events()

        delay_ms = int(step.values.get("delay_ms", 0))
        if delay_ms:
            runner.scheduler.schedule(delay_ms, dispatch)
        else:
            dispatch()

    def _venue_cancel(self, step: Step) -> None:
        record = self._select_order(step)
        record.order = replace(
            record.order, status=OrderStatus.CANCELLED, remaining_qty=0.0
        )

    def _expect(self, runner: Any, step: Step) -> None:
        values = step.values
        if "calls" in values and len(self.state.calls) != int(values["calls"]):
            raise AssertionError(
                f"expected {values['calls']} broker calls, got {len(self.state.calls)}"
            )
        if "position" in values:
            actual = self.state.position_owners.get(step.run, 0.0)
            if abs(actual - float(values["position"])) > 1e-9:
                raise AssertionError(
                    f"expected venue position {values['position']}, got {actual}"
                )
        if "account_position" in values:
            actual = self.state.position
            if abs(actual - float(values["account_position"])) > 1e-9:
                raise AssertionError(
                    f"expected account position {values['account_position']}, got {actual}"
                )
        if "engine_position" in values:
            actual = runner.runs[step.run].position.size
            if abs(actual - float(values["engine_position"])) > 1e-9:
                raise AssertionError(
                    f"expected engine position {values['engine_position']}, got {actual}"
                )
        if "open_orders" in values:
            actual = sum(
                record.run_name == step.run
                and record.order.status
                in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
                for record in self.state.orders.values()
            )
            if actual != int(values["open_orders"]):
                raise AssertionError(
                    f"expected {values['open_orders']} open orders, got {actual}"
                )
        if "total_orders" in values:
            actual = sum(
                record.run_name == step.run for record in self.state.orders.values()
            )
            if actual != int(values["total_orders"]):
                raise AssertionError(
                    f"expected {values['total_orders']} total orders, got {actual}"
                )
        if values.get("distinct_order_ids"):
            ids = [
                record.order.id
                for record in self.state.orders.values()
                if record.run_name == step.run
            ]
            if len(ids) != len(set(ids)):
                raise AssertionError(f"venue order ids are not distinct: {ids}")
        if "wire_symbols" in values:
            actual_symbols = {
                record.order.symbol
                for record in self.state.orders.values()
                if record.run_name == step.run
            }
            expected_symbols = set(values["wire_symbols"])
            if actual_symbols != expected_symbols:
                raise AssertionError(
                    f"expected wire symbols {expected_symbols}, got {actual_symbols}"
                )

    def check_invariants(self, runner: Any) -> Sequence[str]:
        violations: list[str] = []
        live_coids: dict[str, str] = {}
        active_intents: set[tuple[Any, ...]] = set()
        call_counts: dict[tuple[str, str, str], int] = {}
        for call in self.state.calls:
            call_counts[call] = call_counts.get(call, 0) + 1
        for call, count in call_counts.items():
            if count > 3:
                violations.append(
                    f"bounded retry exceeded for {call}: {count} dispatches"
                )
        for record in self.state.orders.values():
            order = record.order
            if record.run_name not in runner.runs:
                violations.append(
                    f"order {order.id} is owned by unknown run {record.run_name}"
                )
            if order.qty < 0.0 or order.filled_qty < 0.0 or order.remaining_qty < 0.0:
                violations.append(f"negative venue quantity on {order.id}")
            if self.quantity_step > 0.0:
                units = order.qty / self.quantity_step
                if abs(units - round(units)) > 1e-8:
                    violations.append(
                        f"quantity-grid residual on {order.id}: qty={order.qty} "
                        f"step={self.quantity_step}"
                    )
            if (
                order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED)
                and order.remaining_qty != 0.0
            ):
                violations.append(
                    f"terminal order {order.id} retains remaining quantity"
                )
            terminal = self.state.terminal_snapshots.get(order.id)
            if terminal is not None:
                if order.status not in (
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                ):
                    violations.append(
                        f"terminal order {order.id} resurrected as {order.status.value}"
                    )
                elif terminal != (order.status, order.filled_qty, order.qty):
                    violations.append(
                        f"terminal order {order.id} was modified after terminalization"
                    )
            elif order.status in (
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
            ):
                self.state.terminal_snapshots[order.id] = (
                    order.status,
                    order.filled_qty,
                    order.qty,
                )
            if order.client_order_id and order.status in (
                OrderStatus.OPEN,
                OrderStatus.PARTIALLY_FILLED,
            ):
                previous = live_coids.setdefault(
                    order.client_order_id, record.intent_key
                )
                if previous != record.intent_key:
                    violations.append(
                        f"client order id {order.client_order_id} aliases two intents"
                    )
                intent_owner = (
                    record.run_name,
                    record.intent_key,
                    record.leg_type,
                    order.side,
                    order.qty,
                    order.price,
                    order.stop_price,
                )
                if intent_owner in active_intents:
                    violations.append(
                        f"economic idempotence violated for {record.run_name} "
                        f"intent={record.intent_key} leg={record.leg_type.value}"
                    )
                active_intents.add(intent_owner)
        if not self.state.pending_events:
            owned_total = sum(self.state.position_owners.values())
            if abs(owned_total - self.state.position) > 1e-9:
                violations.append(
                    f"account position ownership mismatch: owners={owned_total} "
                    f"account={self.state.position}"
                )
            engine_total = sum(
                runtime.position.size for runtime in runner.runs.values()
            )
            if abs(engine_total - self.state.position) > 1e-9:
                violations.append(
                    f"economic account position mismatch: engines={engine_total} "
                    f"venue={self.state.position}"
                )
            owner_signs = {
                1 if size > 0.0 else -1
                for size in self.state.position_owners.values()
                if abs(size) > 1e-9
            }
            if self.venue_mode == "netting" and len(owner_signs) > 1:
                violations.append(
                    "opposing run ownership on a netting account cannot be isolated"
                )
            for run_name, runtime in runner.runs.items():
                venue_size = self.state.position_owners.get(run_name, 0.0)
                if abs(runtime.position.size - venue_size) > 1e-9:
                    violations.append(
                        f"economic position mismatch for {run_name}: "
                        f"engine={runtime.position.size} venue={venue_size}"
                    )
                if self.venue_mode == "hedged":
                    leg_total = sum(
                        size
                        for (owner, _), size in self.state.position_legs.items()
                        if owner == run_name
                    )
                    if abs(leg_total - venue_size) > 1e-9:
                        violations.append(
                            f"hedged leg reconstruction mismatch for {run_name}: "
                            f"legs={leg_total} venue={venue_size}"
                        )
                if abs(venue_size) <= 1e-9:
                    continue
                required: dict[tuple[str | None, str], float] = {}
                dispatched_intents = {
                    intent_key
                    for owner, _, intent_key in self.state.calls
                    if owner == run_name
                }
                for (
                    exit_id,
                    from_entry,
                ), logical_exit in runtime.position.exit_orders.items():
                    intent_key = f"{exit_id}{INTENT_KEY_SEP}{from_entry}"
                    if intent_key not in dispatched_intents:
                        continue
                    qty = abs(float(logical_exit.size))
                    if logical_exit.limit is not None:
                        key = (from_entry, "take-profit")
                        required[key] = required.get(key, 0.0) + qty
                    if logical_exit.stop is not None:
                        key = (from_entry, "stop-loss")
                        required[key] = required.get(key, 0.0) + qty
                coverage: dict[tuple[str | None, str], float] = {}
                for record in self.state.orders.values():
                    if record.run_name != run_name or record.order.status not in (
                        OrderStatus.OPEN,
                        OrderStatus.PARTIALLY_FILLED,
                    ):
                        continue
                    if record.leg_type is LegType.TAKE_PROFIT:
                        outcome = "take-profit"
                    elif record.leg_type in (LegType.STOP_LOSS, LegType.TRAILING_STOP):
                        outcome = "stop-loss"
                    else:
                        continue
                    key = (record.from_entry, outcome)
                    coverage[key] = coverage.get(key, 0.0) + record.order.remaining_qty
                for (from_entry, outcome), required_qty in required.items():
                    protected = coverage.get((from_entry, outcome), 0.0)
                    if protected + 1e-9 < required_qty:
                        violations.append(
                            f"{outcome} protection coverage shortfall for {run_name} "
                            f"from_entry={from_entry!r}: protected={protected} "
                            f"required={required_qty}"
                        )
        return violations

    def close(self) -> None:
        self.state.pending_events.clear()


class HedgedReferenceVenueProfile(ReferenceVenueProfile):
    """Reference venue retaining per-entry legs behind an aggregate position view."""

    plugin_name = "broker-lab-reference-hedged"
    venue_mode = "hedged"
