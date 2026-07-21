"""Opt-in reference suite for the public offline broker conformance lab."""

import socket
from dataclasses import replace

from pynecore.core.broker.models import OrderStatus

from pynecore.testing.broker_lab import (
    HedgedReferenceVenueProfile,
    ReferenceBroker,
    ReferenceVenueProfile,
    Scenario,
    Step,
    pairwise_cases,
)


class UnderProtectingBroker(ReferenceBroker):
    """Deliberately broken control: only the first logical exit is physical."""

    async def execute_exit(self, envelope):
        existing = [
            record.order
            for record in self.state.orders.values()
            if record.run_name == self.run_name and record.leg_type.value != "entry"
        ]
        if existing:
            self.state.calls.append((self.run_name, "t", envelope.intent.intent_key))
            return [existing[0]]
        return await super().execute_exit(envelope)


class UnderProtectingProfile(ReferenceVenueProfile):
    """Control profile proving that the coverage oracle detects a defect."""

    def create_broker(self, run_name, store_ctx):
        broker = UnderProtectingBroker(self, run_name)
        broker.store_ctx = store_ctx
        return broker


class HalfBracketBroker(ReferenceBroker):
    """Deliberately broken control: each OCO outcome covers only half."""

    async def execute_exit(self, envelope):
        orders = await super().execute_exit(envelope)
        for order in orders:
            half_qty = order.qty / 2.0
            shortened = replace(order, qty=half_qty, remaining_qty=half_qty)
            self.state.orders[order.id].order = shortened
        return [self.state.orders[order.id].order for order in orders]


class HalfBracketProfile(ReferenceVenueProfile):
    """Control profile proving that OCO outcomes are not additive coverage."""

    def create_broker(self, run_name, store_ctx):
        broker = HalfBracketBroker(self, run_name)
        broker.store_ctx = store_ctx
        return broker


class NetworkAttemptProfile(ReferenceVenueProfile):
    """Control profile proving that accidental network access fails immediately."""

    def handle_step(self, runner, step):
        if step.kind == "network_attempt":
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
            return True
        return super().handle_step(runner, step)


class BrokenInvariantProfile(ReferenceVenueProfile):
    """Deliberately corrupt venue state to validate independent oracles."""

    def handle_step(self, runner, step):
        if step.kind == "resurrect_terminal":
            record = next(iter(self.state.orders.values()))
            record.order = replace(
                record.order, status=OrderStatus.OPEN, remaining_qty=1.0
            )
            return True
        if step.kind == "duplicate_active":
            record = next(iter(self.state.orders.values()))
            duplicate = replace(record.order, id=self.state.new_id())
            self.state.orders[duplicate.id] = replace(record, order=duplicate)
            return True
        if step.kind == "off_grid_quantity":
            record = next(iter(self.state.orders.values()))
            record.order = replace(record.order, qty=1.5, remaining_qty=1.5)
            return True
        if step.kind == "foreign_owner":
            record = next(iter(self.state.orders.values()))
            record.run_name = "foreign"
            return True
        if step.kind == "retry_storm":
            self.state.calls.extend(("main", "e", "L") for _ in range(4))
            return True
        if step.kind == "venue_position_drift":
            self.state.position = 1.0
            return True
        return super().handle_step(runner, step)


class AliasedSymbolProfile(ReferenceVenueProfile):
    """Provider and broker use distinct symbols for the same instrument."""

    symbol = "LABUSD.P"
    wire_symbol = "LABUSD"


def _scenario(
    name: str,
    *steps: Step,
    runs: tuple[str, ...] = ("main",),
    seed: int = 0,
    tags: frozenset[str] = frozenset({"smoke"}),
    expected_violation: str | None = None,
) -> Scenario:
    return Scenario(
        name=name,
        profile_factory=ReferenceVenueProfile,
        steps=steps,
        runs=runs,
        seed=seed,
        tags=tags,
        expected_violation=expected_violation,
    )


SMOKE = (
    _scenario(
        "market-entry-fill",
        Step("entry", values={"id": "L", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("expect", values={"position": 1.0, "engine_position": 1.0}),
    ),
    _scenario(
        "market-long-short-open-close-repeated-rounds",
        Step("entry", values={"id": "Long1", "side": "buy", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("close", values={"id": "Long1", "side": "sell", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("entry", values={"id": "Short", "side": "sell", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("close", values={"id": "Short", "side": "buy", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("entry", values={"id": "Long2", "side": "buy", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("close", values={"id": "Long2", "side": "sell", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("expect", values={"position": 0.0, "engine_position": 0.0}),
    ),
    _scenario(
        "duplicate-fill-is-idempotent",
        Step("entry", values={"id": "L", "qty": 1.0}),
        Step("sync"),
        Step("fill", values={"fill_id": "stable-fill"}, check_invariants=False),
        Step("duplicate_event", check_invariants=False),
        Step("deliver"),
        Step("expect", values={"position": 1.0, "engine_position": 1.0}),
    ),
    _scenario(
        "restart-adopts-working-order",
        Step("entry", values={"id": "L", "qty": 1.0, "limit": 90.0}),
        Step("sync"),
        Step("restart", check_invariants=False),
        Step("entry", values={"id": "L", "qty": 1.0, "limit": 90.0}),
        Step("sync"),
        Step("expect", values={"open_orders": 1}),
    ),
    _scenario(
        "concurrent-run-ownership",
        Step("entry", run="A", values={"id": "A", "qty": 1.0}),
        Step("sync", run="A"),
        Step("fill", run="A", check_invariants=False),
        Step("deliver", run="A"),
        Step("sync", run="B"),
        Step("expect", run="A", values={"position": 1.0, "engine_position": 1.0}),
        Step("expect", run="B", values={"position": 0.0, "engine_position": 0.0}),
        runs=("A", "B"),
    ),
    _scenario(
        "global-exit-protection-coverage",
        Step("entry", values={"id": "A", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("entry", values={"id": "B", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("exit", values={"id": "X", "from_entry": "A", "qty": 1.0, "limit": 110.0}),
        Step("exit", values={"id": "X", "from_entry": "B", "qty": 1.0, "limit": 110.0}),
        Step("sync"),
    ),
    _scenario(
        "bounded-entry-rejection",
        Step("reject_entries", values={"count": 10}),
        Step("entry", values={"id": "L", "qty": 1.0, "limit": 90.0}),
        Step("sync"),
        Step("sync"),
        Step("sync"),
        Step("sync"),
        Step("sync"),
        Step("expect", values={"calls": 3}),
    ),
    _scenario(
        "read-failure-defers-new-exposure",
        Step("read_error"),
        Step("entry", values={"id": "L", "qty": 1.0}),
        Step("sync", check_invariants=False),
        Step("sync"),
    ),
    _scenario(
        "completed-intent-does-not-resurrect-after-restart",
        Step("entry", values={"id": "done", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("restart", check_invariants=False),
        Step("sync"),
        Step("expect", values={"calls": 1, "open_orders": 0}),
    ),
    _scenario(
        "cancel-ack-without-push-is-terminal",
        Step("entry", values={"id": "cancel-me", "qty": 1.0, "limit": 90.0}),
        Step("sync"),
        Step("cancel", values={"id": "cancel-me"}),
        Step("sync"),
        Step("expect", values={"open_orders": 0}),
    ),
    _scenario(
        "entry-amend-replaces-one-physical-order",
        Step("entry", values={"id": "amend", "qty": 1.0, "limit": 90.0}),
        Step("sync"),
        Step("amend", values={"id": "amend", "limit": 91.0}),
        Step("sync"),
        Step("expect", values={"open_orders": 1}),
    ),
    _scenario(
        "cancel-all-terminalizes-every-active-order",
        Step("entry", values={"id": "A", "qty": 1.0, "limit": 90.0}),
        Step("entry", values={"id": "B", "qty": 1.0, "stop": 110.0}),
        Step("sync"),
        Step("expect", values={"open_orders": 2}),
        Step("cancel_all"),
        Step("sync"),
        Step("expect", values={"open_orders": 0, "total_orders": 2}),
    ),
    _scenario(
        "same-id-cancel-then-recreate-uses-new-physical-order",
        Step("entry", values={"id": "L", "qty": 1.0, "limit": 90.0}),
        Step("sync"),
        Step("cancel", values={"id": "L"}),
        Step("sync"),
        Step("entry", values={"id": "L", "qty": 1.0, "limit": 91.0}),
        Step("sync"),
        Step(
            "expect",
            values={
                "open_orders": 1,
                "total_orders": 2,
                "distinct_order_ids": True,
            },
        ),
    ),
    _scenario(
        "partial-close-preserves-exact-residual",
        Step("entry", values={"id": "L", "qty": 3.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("close", values={"id": "L", "from_entry": "L", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("expect", values={"position": 2.0, "engine_position": 2.0}),
    ),
    _scenario(
        "over-close-is-capped-at-live-position",
        Step("entry", values={"id": "L", "qty": 2.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("close", values={"id": "L", "from_entry": "L", "qty": 3.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("expect", values={"position": 0.0, "engine_position": 0.0}),
    ),
    _scenario(
        "percentage-close-preserves-derived-residual",
        Step("entry", values={"id": "L", "qty": 4.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("close_percent", values={"id": "L", "side": "sell", "percent": 25.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("expect", values={"position": 3.0, "engine_position": 3.0}),
    ),
    _scenario(
        "long-to-short-reversal-has-exact-economic-size",
        Step("entry", values={"id": "R", "side": "buy", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("entry", values={"id": "R", "side": "sell", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("expect", values={"position": -1.0, "engine_position": -1.0}),
    ),
    _scenario(
        "short-to-long-reversal-has-exact-economic-size",
        Step("entry", values={"id": "R", "side": "sell", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("entry", values={"id": "R", "side": "buy", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("expect", values={"position": 1.0, "engine_position": 1.0}),
    ),
    _scenario(
        "minimum-below-order-is-observable-and-bounded",
        Step("skip_entries", values={"count": 10}),
        Step("entry", values={"id": "tiny", "qty": 1.0, "limit": 90.0}),
        Step("sync"),
        Step("sync"),
        Step("sync"),
        Step("expect", values={"calls": 1}),
    ),
    _scenario(
        "stale-read-fails-closed-before-dispatch",
        Step("read_error"),
        Step("entry", values={"id": "L", "qty": 1.0}),
        Step("sync", check_invariants=False),
        Step("expect", values={"calls": 0}),
        Step("sync"),
        Step("expect", values={"calls": 1}),
    ),
    Scenario(
        name="per-entry-hedge-reconstruction-survives-restart",
        profile_factory=HedgedReferenceVenueProfile,
        seed=0,
        steps=(
            Step("entry", values={"id": "A", "qty": 1.0}),
            Step("sync"),
            Step("fill", check_invariants=False),
            Step("deliver"),
            Step("entry", values={"id": "B", "qty": 1.0}),
            Step("sync"),
            Step("fill", check_invariants=False),
            Step("deliver"),
            Step("restart", check_invariants=False),
            Step("sync"),
            Step("expect", values={"position": 2.0, "engine_position": 2.0}),
        ),
    ),
    Scenario(
        name="provider-symbol-alias-survives-restart-adoption",
        profile_factory=AliasedSymbolProfile,
        seed=0,
        steps=(
            Step("entry", values={"id": "alias", "qty": 1.0, "limit": 90.0}),
            Step("sync"),
            Step("restart", check_invariants=False),
            Step("entry", values={"id": "alias", "qty": 1.0, "limit": 90.0}),
            Step("sync"),
            Step("expect", values={"open_orders": 1, "wire_symbols": ["LABUSD"]}),
        ),
    ),
    _scenario(
        "keyed-close-does-not-repeat-after-restart",
        Step("entry", values={"id": "L", "qty": 2.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("close", values={"id": "L", "from_entry": "L", "qty": 1.0}),
        Step("sync"),
        Step("fill", check_invariants=False),
        Step("deliver"),
        Step("restart", check_invariants=False),
        Step("sync"),
        Step("expect", values={"position": 1.0, "engine_position": 1.0, "calls": 3}),
    ),
    _scenario(
        "reordered-partial-fills-are-idempotent",
        Step("entry", values={"id": "L", "qty": 2.0}),
        Step("sync"),
        Step("fill", values={"qty": 1.0, "fill_id": "fill-a"}, check_invariants=False),
        Step("fill", values={"qty": 1.0, "fill_id": "fill-b"}, check_invariants=False),
        Step("reorder_events", check_invariants=False),
        Step("deliver"),
        Step("expect", values={"position": 2.0, "engine_position": 2.0}),
    ),
    _scenario(
        "marketable-limit-and-stop-are-direction-symmetric",
        Step(
            "entry",
            run="buy-limit",
            values={"id": "BL", "side": "buy", "qty": 1.0, "limit": 110.0},
        ),
        Step("sync", run="buy-limit", values={"last_price": 100.0}),
        Step(
            "entry",
            run="sell-limit",
            values={"id": "SL", "side": "sell", "qty": 1.0, "limit": 90.0},
        ),
        Step("sync", run="sell-limit", values={"last_price": 100.0}),
        Step(
            "entry",
            run="buy-stop",
            values={"id": "BS", "side": "buy", "qty": 1.0, "stop": 90.0},
        ),
        Step("sync", run="buy-stop", values={"last_price": 100.0}),
        Step(
            "entry",
            run="sell-stop",
            values={"id": "SS", "side": "sell", "qty": 1.0, "stop": 110.0},
        ),
        Step("sync", run="sell-stop", values={"last_price": 100.0}),
        Step("expect", run="buy-limit", values={"open_orders": 1}),
        Step("expect", run="sell-limit", values={"open_orders": 1}),
        Step("expect", run="buy-stop", values={"open_orders": 1}),
        Step("expect", run="sell-stop", values={"open_orders": 1}),
        runs=("buy-limit", "sell-limit", "buy-stop", "sell-stop"),
    ),
    Scenario(
        name="control-under-protection-is-detected",
        profile_factory=UnderProtectingProfile,
        seed=0,
        expected_violation="take-profit protection coverage shortfall",
        steps=(
            Step("entry", values={"id": "L", "qty": 2.0}),
            Step("sync"),
            Step("fill", check_invariants=False),
            Step("deliver"),
            Step(
                "exit",
                values={"id": "XA", "from_entry": "L", "qty": 1.0, "limit": 110.0},
            ),
            Step(
                "exit",
                values={"id": "XB", "from_entry": "L", "qty": 1.0, "limit": 111.0},
            ),
            Step("sync"),
        ),
    ),
    Scenario(
        name="control-half-covered-oco-bracket-is-detected",
        profile_factory=HalfBracketProfile,
        seed=0,
        expected_violation="take-profit protection coverage shortfall",
        steps=(
            Step("entry", values={"id": "L", "qty": 2.0}),
            Step("sync"),
            Step("fill", check_invariants=False),
            Step("deliver"),
            Step(
                "exit",
                values={
                    "id": "X",
                    "from_entry": "L",
                    "qty": 2.0,
                    "limit": 110.0,
                    "stop": 90.0,
                },
            ),
            Step("sync"),
        ),
    ),
    Scenario(
        name="account-global-restart-adopts-only-owned-exposure",
        profile_factory=ReferenceVenueProfile,
        runs=("A", "B"),
        seed=0,
        steps=(
            Step("entry", run="A", values={"id": "A", "qty": 1.0}),
            Step("sync", run="A"),
            Step("fill", run="A", check_invariants=False),
            Step("deliver", run="A"),
            Step("restart", run="B", check_invariants=False),
            Step(
                "expect",
                run="A",
                values={
                    "position": 1.0,
                    "engine_position": 1.0,
                    "account_position": 1.0,
                },
            ),
            Step("expect", run="B", values={"position": 0.0, "engine_position": 0.0}),
        ),
    ),
    Scenario(
        name="control-opposing-runs-on-netting-account-are-detected",
        profile_factory=ReferenceVenueProfile,
        runs=("A", "B"),
        seed=0,
        expected_violation="opposing run ownership on a netting account",
        steps=(
            Step("entry", run="A", values={"id": "A", "side": "buy", "qty": 1.0}),
            Step("sync", run="A"),
            Step("fill", run="A", check_invariants=False),
            Step("deliver", run="A"),
            Step("entry", run="B", values={"id": "B", "side": "sell", "qty": 1.0}),
            Step("sync", run="B"),
            Step("fill", run="B", check_invariants=False),
            Step("deliver", run="B"),
        ),
    ),
    Scenario(
        name="opposing-runs-remain-distinct-on-hedged-account",
        profile_factory=HedgedReferenceVenueProfile,
        runs=("A", "B"),
        seed=0,
        steps=(
            Step("entry", run="A", values={"id": "A", "side": "buy", "qty": 1.0}),
            Step("sync", run="A"),
            Step("fill", run="A", check_invariants=False),
            Step("deliver", run="A"),
            Step("entry", run="B", values={"id": "B", "side": "sell", "qty": 1.0}),
            Step("sync", run="B"),
            Step("fill", run="B", check_invariants=False),
            Step("deliver", run="B"),
            Step(
                "expect",
                run="A",
                values={
                    "position": 1.0,
                    "engine_position": 1.0,
                    "account_position": 0.0,
                },
            ),
            Step("expect", run="B", values={"position": -1.0, "engine_position": -1.0}),
        ),
    ),
    Scenario(
        name="control-network-attempt-is-blocked",
        profile_factory=NetworkAttemptProfile,
        seed=0,
        expected_violation="offline broker lab blocked a network attempt",
        steps=(Step("network_attempt"),),
    ),
    Scenario(
        name="control-terminal-resurrection-is-detected",
        profile_factory=BrokenInvariantProfile,
        seed=0,
        expected_violation="terminal order lab-1 resurrected",
        steps=(
            Step("entry", values={"id": "L", "qty": 1.0}),
            Step("sync"),
            Step("fill", check_invariants=False),
            Step("deliver"),
            Step("resurrect_terminal"),
        ),
    ),
    Scenario(
        name="control-duplicate-dispatch-is-detected",
        profile_factory=BrokenInvariantProfile,
        seed=0,
        expected_violation="economic idempotence violated",
        steps=(
            Step("entry", values={"id": "L", "qty": 1.0, "limit": 90.0}),
            Step("sync"),
            Step("duplicate_active"),
        ),
    ),
    Scenario(
        name="control-off-grid-quantity-is-detected",
        profile_factory=BrokenInvariantProfile,
        seed=0,
        expected_violation="quantity-grid residual",
        steps=(
            Step("entry", values={"id": "L", "qty": 1.0, "limit": 90.0}),
            Step("sync"),
            Step("off_grid_quantity"),
        ),
    ),
    Scenario(
        name="control-cross-run-ownership-is-detected",
        profile_factory=BrokenInvariantProfile,
        seed=0,
        expected_violation="owned by unknown run foreign",
        steps=(
            Step("entry", values={"id": "L", "qty": 1.0, "limit": 90.0}),
            Step("sync"),
            Step("foreign_owner"),
        ),
    ),
    Scenario(
        name="control-unbounded-retry-is-detected",
        profile_factory=BrokenInvariantProfile,
        seed=0,
        expected_violation="bounded retry exceeded",
        steps=(Step("retry_storm"),),
    ),
    Scenario(
        name="control-economic-drift-is-detected",
        profile_factory=BrokenInvariantProfile,
        seed=0,
        expected_violation="account position ownership mismatch",
        steps=(Step("venue_position_drift"),),
    ),
)


def _extended(seed: int) -> list[Scenario]:
    scenarios: list[Scenario] = []
    axes = {
        "side": ("buy", "sell"),
        "order": ("market", "limit", "stop", "stop-limit", "bracket"),
        "fill": ("full", "partial"),
        "mode": ("netting", "hedged"),
        "runs": (1, 2),
        "restart": (False, True),
        "ordering": ("ack-push", "duplicate", "reordered", "delayed"),
        "fault": ("none", "read-before-dispatch"),
        "race": ("none", "cancel-fill"),
        "pine_ids": ("same", "different"),
    }
    for index, case in enumerate(pairwise_cases(axes, seed=seed)):
        qty = (
            2.0 if case["fill"] == "partial" or case["ordering"] == "reordered" else 1.0
        )
        order_values = {"id": "E", "side": case["side"], "qty": qty}
        if case["order"] == "limit":
            order_values["limit"] = 90.0 if case["side"] == "buy" else 110.0
        elif case["order"] == "stop":
            order_values["stop"] = 110.0 if case["side"] == "buy" else 90.0
        elif case["order"] in ("stop-limit", "bracket"):
            order_values["limit"] = 90.0 if case["side"] == "buy" else 110.0
            order_values["stop"] = 110.0 if case["side"] == "buy" else 90.0
        steps = [Step("entry", values=order_values)]
        if case["fault"] == "read-before-dispatch":
            steps.extend((Step("read_error"), Step("sync", check_invariants=False)))
        steps.append(Step("sync"))
        if case["restart"]:
            steps.extend(
                (Step("restart"), Step("entry", values=order_values), Step("sync"))
            )
        steps.append(
            Step(
                "fill",
                values={"qty": 1.0, "fill_id": f"extended-{index}"},
                check_invariants=False,
            )
        )
        if case["race"] == "cancel-fill":
            steps.append(Step("cancel", values={"id": "E"}, check_invariants=False))
        if case["ordering"] == "duplicate":
            steps.append(Step("duplicate_event", check_invariants=False))
        elif case["ordering"] == "reordered":
            steps.extend(
                (
                    Step(
                        "fill",
                        values={"qty": 1.0, "fill_id": f"extended-{index}-b"},
                        check_invariants=False,
                    ),
                    Step("reorder_events", check_invariants=False),
                )
            )
        if case["ordering"] == "delayed":
            steps.extend(
                (
                    Step(
                        "delayed_deliver",
                        values={"delay_ms": 10},
                        check_invariants=False,
                    ),
                    Step("advance", values={"ms": 10}),
                )
            )
        else:
            steps.append(Step("deliver"))
        runs = ("main", "peer") if case["runs"] == 2 else ("main",)
        if case["runs"] == 2:
            peer_values = dict(order_values)
            peer_values["id"] = "E" if case["pine_ids"] == "same" else "PEER"
            steps.extend(
                (
                    Step("entry", run="peer", values=peer_values),
                    Step("sync", run="peer"),
                    Step("fill", run="peer", check_invariants=False),
                    Step("deliver", run="peer"),
                )
            )
        scenarios.append(
            Scenario(
                name=f"pairwise-{index:03d}",
                profile_factory=(
                    HedgedReferenceVenueProfile
                    if case["mode"] == "hedged"
                    else ReferenceVenueProfile
                ),
                steps=tuple(steps),
                runs=runs,
                seed=seed,
                tags=frozenset({"extended"}),
            )
        )
    return scenarios


def build_suite(*, mode: str, seed: int):
    """Return the requested opt-in suite."""
    if mode == "smoke":
        return SMOKE
    return (*SMOKE, *_extended(seed))
