"""Tests for the both-set entry STOP-leg native fail-safe parent-ref resolver.

A both-set Pine entry (``strategy.entry(limit=, stop=)``) is a native LIMIT
working order plus a software price-watch on the STOP side. When the STOP leg
wins the OCO, the position opens via the stop-fired MARKET under the
deterministic ``KIND_ENTRY_STOP`` client-order-id — NOT the native LIMIT's
``KIND_ENTRY`` id.

The broker reconcile feed reports the live position under the id it opened
under, so every native fail-safe parent-ref resolver must key on that id.
:meth:`OrderSyncEngine._resolve_parent_opening_ref` is the single resolver:
it returns the ``KIND_ENTRY_STOP`` id iff a position row was persisted under it
(the durable, restart-surviving signal that the stop fired — the entry-stop
watch row is terminal-filtered once the stop wins), otherwise ``KIND_ENTRY``.

These tests pin that resolver and its use in the close-time retire path, both
live and across a simulated restart (only the persisted envelope anchor
survives).
"""
from __future__ import annotations

from pathlib import Path

from pynecore.core.broker.idempotency import KIND_ENTRY, KIND_ENTRY_STOP
from pynecore.core.broker.models import (
    CapabilityLevel,
    EntryIntent,
    ExchangeCapabilities,
    OrderType,
)
from pynecore.core.broker.native_failsafe_manager import FailsafeHealth
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.storage import (
    BrokerStore,
    EnvelopeRecord,
    RunContext,
    RunIdentity,
)
from pynecore.core.broker.store_helpers import (
    ENTRY_KIND_POSITION,
    create_entry_order_row,
)
from pynecore.core.broker.sync_engine import OrderSyncEngine


SYMBOL = "EURUSD"


class _FakeBroker:
    client_order_id_max_len = 30  # BrokerPlugin contract attribute
    def get_capabilities(self) -> ExchangeCapabilities:
        return ExchangeCapabilities(stop_order=CapabilityLevel.NATIVE)

    async def execute_entry(self, envelope):
        return []

    async def get_position(self, symbol):
        return None

    async def get_open_orders(self, symbol=None):
        return []

    async def get_balance(self):
        return {}


def _identity() -> RunIdentity:
    return RunIdentity(
        strategy_id="entry-stop-failsafe-test",
        symbol=SYMBOL,
        timeframe="60",
        account_id="acc-1",
    )


def _make_store(tmp_path: Path) -> tuple[BrokerStore, RunContext]:
    store = BrokerStore(tmp_path / "broker.db", plugin_name="capitalcom")
    ctx = store.open_run(_identity(), script_source="// test")
    return store, ctx


def _engine(ctx: RunContext | None) -> OrderSyncEngine:
    return OrderSyncEngine(
        _FakeBroker(), BrokerPosition(), SYMBOL,
        run_tag="tts0", mintick=0.0001, store_ctx=ctx,
    )


def _both_set_entry() -> EntryIntent:
    return EntryIntent(
        pine_id="Long", symbol=SYMBOL, side="buy", qty=1.0,
        order_type=OrderType.LIMIT, limit=1.16, stop=1.18,
    )


def _persist_stop_position(ctx: RunContext, stop_ref: str) -> None:
    """Persist the row the stop-fired MARKET would write when the STOP won."""
    create_entry_order_row(
        ctx,
        coid=stop_ref,
        symbol=SYMBOL,
        side="buy",
        qty=1.0,
        intent_key="Long",
        pine_entry_id="Long",
        kind=ENTRY_KIND_POSITION,
        order_type=OrderType.MARKET.value,
    )


# === _resolve_parent_opening_ref ========================================

def __test_resolve_returns_none_when_parent_untracked__():
    """Resolver returns ``None`` when no store context tracks the parent entry."""
    eng = _engine(None)
    assert eng._resolve_parent_opening_ref("Long") is None


def __test_resolve_returns_kind_entry_when_no_stop_row__(tmp_path):
    """Resolver returns the ``KIND_ENTRY`` id when no stop position row exists."""
    # Limit-won (or not-yet-resolved) entry: the position opened — if at all —
    # under the native LIMIT's KIND_ENTRY id; no KIND_ENTRY_STOP row exists.
    store, ctx = _make_store(tmp_path)
    eng = _engine(ctx)
    env = eng._build_envelope(_both_set_entry())
    assert eng._resolve_parent_opening_ref("Long") == env.client_order_id(KIND_ENTRY)
    store.close()


def __test_resolve_returns_kind_entry_stop_when_stop_row_present__(tmp_path):
    """Resolver returns the ``KIND_ENTRY_STOP`` id once a stop position row is persisted."""
    # Stop-won: the stop-fired MARKET persisted a position row under the
    # deterministic KIND_ENTRY_STOP id; the resolver must reflect it.
    store, ctx = _make_store(tmp_path)
    eng = _engine(ctx)
    env = eng._build_envelope(_both_set_entry())
    stop_ref = env.client_order_id(KIND_ENTRY_STOP)
    _persist_stop_position(ctx, stop_ref)
    assert eng._resolve_parent_opening_ref("Long") == stop_ref
    assert stop_ref != env.client_order_id(KIND_ENTRY)
    store.close()


def __test_resolve_kind_entry_stop_survives_restart__(tmp_path):
    """Resolver rebuilds the ``KIND_ENTRY_STOP`` id from the anchor alone after a restart."""
    # Restart-fragility guard: after a restart only the persisted envelope
    # anchor survives (the entry-stop watch row is terminal-filtered once the
    # stop won). The resolver must still rebuild the SAME KIND_ENTRY_STOP id
    # from the anchor and find the durable stop position row.
    store, ctx = _make_store(tmp_path)
    eng = _engine(ctx)
    env = eng._build_envelope(_both_set_entry())
    stop_ref = env.client_order_id(KIND_ENTRY_STOP)
    _persist_stop_position(ctx, stop_ref)

    # Simulate restart: the live envelope is gone, only the anchor remains.
    eng._envelopes.clear()
    eng._persisted_envelope_anchors["Long"] = EnvelopeRecord(
        key="Long", bar_ts_ms=env.bar_ts_ms, retry_seq=env.retry_seq,
    )
    assert eng._resolve_parent_opening_ref("Long") == stop_ref
    store.close()


def __test_resolve_kind_entry_on_restart_without_stop_row__(tmp_path):
    """Resolver rebuilds the ``KIND_ENTRY`` id from the anchor on restart with no stop row."""
    # Restart, limit-won: no KIND_ENTRY_STOP row -> rebuild the KIND_ENTRY id.
    store, ctx = _make_store(tmp_path)
    eng = _engine(ctx)
    env = eng._build_envelope(_both_set_entry())
    entry_ref = env.client_order_id(KIND_ENTRY)

    eng._envelopes.clear()
    eng._persisted_envelope_anchors["Long"] = EnvelopeRecord(
        key="Long", bar_ts_ms=env.bar_ts_ms, retry_seq=env.retry_seq,
    )
    assert eng._resolve_parent_opening_ref("Long") == entry_ref
    store.close()


# === retire path keyed on the opening ref ===============================

def __test_retire_clears_stop_won_state_under_kind_entry_stop_ref__(tmp_path):
    """Retire resolves the ``KIND_ENTRY_STOP`` ref and marks its fail-safe state RETIRED."""
    # End-to-end of the F3 bug: the reconcile feed registers the fail-safe
    # state under the KIND_ENTRY_STOP id (the id the position opened under).
    # On close, _retire_native_failsafe_for_entry must resolve THAT id and
    # clear the state — keying on KIND_ENTRY (the pre-fix behaviour) would
    # strand it.
    store, ctx = _make_store(tmp_path)
    eng = _engine(ctx)
    env = eng._build_envelope(_both_set_entry())
    stop_ref = env.client_order_id(KIND_ENTRY_STOP)
    _persist_stop_position(ctx, stop_ref)

    eng._native_failsafe_manager.register_parent(
        parent_entry_dispatch_ref=stop_ref,
        symbol=SYMBOL,
        parent_side="long",
        mintick=0.0001,
    )
    assert eng._native_failsafe_manager.get_state(stop_ref) is not None

    eng._retire_native_failsafe_for_entry("Long")
    # The retire keeps a one-shot RETIRED marker rather than dropping the state
    # (see NativeFailsafeManager.unregister_parent docstring). Resolving the
    # WRONG ref (KIND_ENTRY, the pre-fix behaviour) would leave this HEALTHY.
    settled = eng._native_failsafe_manager.get_state(stop_ref)
    assert settled is not None and settled.health == FailsafeHealth.RETIRED
    store.close()


def __test_retire_clears_limit_won_state_under_kind_entry_ref__(tmp_path):
    """Retire resolves the ``KIND_ENTRY`` ref and marks its fail-safe state RETIRED."""
    # Mirror for the limit-won case: no stop row, state registered under
    # KIND_ENTRY, retire resolves KIND_ENTRY and clears it.
    store, ctx = _make_store(tmp_path)
    eng = _engine(ctx)
    env = eng._build_envelope(_both_set_entry())
    entry_ref = env.client_order_id(KIND_ENTRY)

    eng._native_failsafe_manager.register_parent(
        parent_entry_dispatch_ref=entry_ref,
        symbol=SYMBOL,
        parent_side="long",
        mintick=0.0001,
    )
    assert eng._native_failsafe_manager.get_state(entry_ref) is not None

    eng._retire_native_failsafe_for_entry("Long")
    settled = eng._native_failsafe_manager.get_state(entry_ref)
    assert settled is not None and settled.health == FailsafeHealth.RETIRED
    store.close()
