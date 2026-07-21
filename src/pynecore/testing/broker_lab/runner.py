"""Scenario execution over the real broker engine and SQLite store."""

import asyncio
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pynecore import lib
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.lib.strategy import Order, _order_type_close, _order_type_entry

from .model import Scenario, ScenarioInvariantError, ScenarioResult, Step, VenueProfile
from .scheduler import DeterministicScheduler


@dataclass
class RunRuntime:
    """Resources belonging to one logical bot run."""

    name: str
    store: BrokerStore
    store_ctx: RunContext
    broker: Any
    position: BrokerPosition
    engine: OrderSyncEngine


class _NetworkGuard:
    def __enter__(self):
        self._create_connection = socket.create_connection
        self._connect = socket.socket.connect
        self._connect_ex = socket.socket.connect_ex

        def blocked(*_args: Any, **_kwargs: Any):
            raise RuntimeError("offline broker lab blocked a network attempt")

        setattr(socket, "create_connection", blocked)
        setattr(socket.socket, "connect", blocked)
        setattr(socket.socket, "connect_ex", blocked)
        return self

    def __exit__(self, *_exc: Any) -> None:
        setattr(socket, "create_connection", self._create_connection)
        setattr(socket.socket, "connect", self._connect)
        setattr(socket.socket, "connect_ex", self._connect_ex)


class ScenarioRunner:
    """Execute, check, and minimize deterministic broker scenarios."""

    def __init__(self, *, artifact_root: Path | None = None) -> None:
        self.artifact_root = artifact_root
        self.profile: VenueProfile
        self.runs: dict[str, RunRuntime] = {}
        self.now_ms = 1_700_000_000_000
        self.scheduler = DeterministicScheduler(self.now_ms)
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._root: Path | None = None
        self._script_sentinel = object()
        self._old_script: Any = self._script_sentinel

    def run(self, scenario: Scenario, *, minimize: bool = True) -> ScenarioResult:
        result = self._execute(scenario)
        if result.passed or not minimize or len(scenario.steps) < 2:
            return result
        target = (result.violation or "").split(":", 1)[0]
        steps = list(scenario.steps)
        changed = True
        while changed:
            changed = False
            for index in range(len(steps)):
                candidate = steps[:index] + steps[index + 1 :]
                if not candidate:
                    continue
                trial = self._execute(
                    Scenario(
                        name=scenario.name,
                        profile_factory=scenario.profile_factory,
                        steps=tuple(candidate),
                        runs=scenario.runs,
                        seed=scenario.seed,
                        tags=scenario.tags,
                        expected_violation=scenario.expected_violation,
                    )
                )
                if not trial.passed and (trial.violation or "").split(":", 1)[0] == target:
                    steps = candidate
                    changed = True
                    break
        return ScenarioResult(
            name=result.name,
            passed=False,
            seed=result.seed,
            executed_steps=result.executed_steps,
            violation=result.violation,
            minimized_steps=tuple(steps),
            artifact_dir=result.artifact_dir,
        )

    def _execute(self, scenario: Scenario) -> ScenarioResult:
        executed: list[Step] = []
        violation: str | None = None
        artifact_dir: Path | None = None
        try:
            self._start(scenario)
            artifact_dir = self._root if self.artifact_root is not None else None
            with _NetworkGuard():
                for step in scenario.steps:
                    self.apply(step)
                    executed.append(step)
                    if step.check_invariants:
                        self.check_invariants()
        except Exception as exc:
            violation = f"{type(exc).__name__}: {exc}"
        finally:
            self.close()
        if scenario.expected_violation is not None:
            if violation is None:
                violation = f"expected violation was not raised: {scenario.expected_violation}"
            elif scenario.expected_violation in violation:
                violation = None
        return ScenarioResult(
            name=scenario.name,
            passed=violation is None,
            seed=scenario.seed,
            executed_steps=tuple(executed),
            violation=violation,
            artifact_dir=artifact_dir,
        )

    def _start(self, scenario: Scenario) -> None:
        self.profile = scenario.profile_factory()
        self.runs = {}
        self.now_ms = 1_700_000_000_000
        self.scheduler = DeterministicScheduler(self.now_ms)
        self._old_script = getattr(lib, "_script", self._script_sentinel)
        lib._script = SimpleNamespace(initial_capital=1_000_000.0)
        if self.artifact_root is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="pyne-broker-lab-")
            self._root = Path(self._tmp.name)
        else:
            self._root = self.artifact_root / scenario.name
            self._root.mkdir(parents=True, exist_ok=True)
        for run_name in scenario.runs:
            self._open_run(run_name)

    def _open_run(self, run_name: str) -> None:
        assert self._root is not None
        store = BrokerStore(
            self._root / "broker.sqlite",
            plugin_name=self.profile.plugin_name,
        )
        identity = RunIdentity(
            strategy_id="broker_lab",
            symbol=self.profile.symbol,
            timeframe=self.profile.timeframe,
            account_id=self.profile.account_id,
            label=run_name,
        )
        store_ctx = store.open_run(identity, script_source="// offline broker lab")
        broker = self.profile.create_broker(run_name, store_ctx)
        position = BrokerPosition()
        engine = OrderSyncEngine(
            broker=broker,
            position=position,
            symbol=self.profile.symbol,
            run_tag=store_ctx.run_tag,
            mintick=0.01,
            store_ctx=store_ctx,
            reconcile_every_n_syncs=1,
        )
        engine.reconcile()
        self.runs[run_name] = RunRuntime(
            name=run_name,
            store=store,
            store_ctx=store_ctx,
            broker=broker,
            position=position,
            engine=engine,
        )

    def apply(self, step: Step) -> None:
        if step.run not in self.runs:
            raise KeyError(f"unknown run {step.run!r}")
        runtime = self.runs[step.run]
        values = step.values
        if step.kind == "entry":
            pine_id = str(values.get("id", "L"))
            side = str(values.get("side", "buy"))
            qty = float(values.get("qty", 1.0))
            runtime.position.entry_orders[pine_id] = Order(
                pine_id,
                qty if side == "buy" else -qty,
                order_type=_order_type_entry,
                limit=values.get("limit"),
                stop=values.get("stop"),
            )
        elif step.kind == "amend":
            pine_id = str(values.get("id", "L"))
            old = runtime.position.entry_orders.get(pine_id)
            if old is None:
                raise ValueError(f"cannot amend unknown entry {pine_id!r}")
            runtime.position.entry_orders[pine_id] = Order(
                pine_id,
                old.size,
                order_type=_order_type_entry,
                limit=values.get("limit", old.limit),
                stop=values.get("stop", old.stop),
            )
        elif step.kind in ("exit", "close"):
            pine_id = str(values.get("id", "X" if step.kind == "exit" else "L"))
            exit_id = pine_id if step.kind == "exit" else f"Close entry(s) order {pine_id}"
            from_entry = values.get("from_entry")
            side = str(values.get("side", "sell"))
            qty = float(values.get("qty", abs(runtime.position.size) or 1.0))
            order = Order(
                from_entry if step.kind == "exit" else pine_id,
                qty if side == "buy" else -qty,
                order_type=_order_type_close,
                exit_id=exit_id,
                limit=values.get("limit"),
                stop=values.get("stop"),
            )
            order.from_entry_na = from_entry is None
            order.rest_leg = bool(values.get("rest_leg", False))
            key = (exit_id, from_entry if step.kind == "exit" else pine_id)
            runtime.position.exit_orders[key] = order
        elif step.kind == "cancel":
            cancel_id = values.get("id")
            runtime.position.entry_orders.pop(cancel_id, None)
            for key in list(runtime.position.exit_orders):
                if cancel_id in key:
                    runtime.position.exit_orders.pop(key)
        elif step.kind == "cancel_all":
            runtime.position.entry_orders.clear()
            runtime.position.exit_orders.clear()
        elif step.kind == "sync":
            self.scheduler.advance(int(values.get("advance_ms", 1_000)))
            self.now_ms = self.scheduler.now_ms
            runtime.engine.sync(self.now_ms, last_price=float(values.get("last_price", 100.0)))
        elif step.kind == "advance":
            self.scheduler.advance(int(values.get("ms", 1_000)))
            self.now_ms = self.scheduler.now_ms
        elif step.kind == "pump_watch":

            async def pump_one() -> None:
                stream = runtime.broker.watch_orders()
                try:
                    event = await stream.__anext__()
                finally:
                    await stream.aclose()
                runtime.engine.on_order_event(event)

            asyncio.run(pump_one())
            runtime.engine.apply_async_events()
        elif step.kind == "restart":
            runtime.store_ctx.close()
            runtime.store.close()
            del self.runs[step.run]
            self._open_run(step.run)
            self.runs[step.run].engine.settle_restart_state(self.now_ms)
        elif step.kind == "shutdown":
            runtime.store_ctx.close()
            runtime.store.close()
        elif not self.profile.handle_step(self, step):
            raise ValueError(f"unsupported scenario step {step.kind!r}")

    def check_invariants(self) -> None:
        violations = list(self.profile.check_invariants(self))
        if violations:
            raise ScenarioInvariantError("; ".join(violations))

    def close(self) -> None:
        for runtime in list(self.runs.values()):
            try:
                runtime.store_ctx.close()
            except Exception:
                pass
            runtime.store.close()
        self.runs.clear()
        if hasattr(self, "profile"):
            self.profile.close()
        if self._old_script is self._script_sentinel:
            if hasattr(lib, "_script"):
                delattr(lib, "_script")
        else:
            lib._script = self._old_script
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
