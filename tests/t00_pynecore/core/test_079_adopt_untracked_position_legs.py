"""
Standalone unit tests for
:func:`pynecore.core.broker.store_helpers.adopt_untracked_position_legs`.

Core-layer only — no plugin, no exchange, no httpx. The helper seeds
confirmed ``position`` rows for untracked live venue legs so netting
venues with handle-addressed closes (Capital.com / IG-style
``DELETE /positions/{dealId}``) can route ``close_all`` through the
normal confirmed-row path after a restart under a new ``run_id``.

Coverage:

- seeds a confirmed row + ``ref_kind`` ref + audit event per leg
- skips legs already tracked by a live row's ``exchange_order_id``
- skips legs already reachable via a ``ref_kind`` ref
- skips legs owned by another run on the same account
- skips non-positive quantities
- raises on a foreign-symbol leg (structural symbol scoping)
- idempotent: a second pass adopts nothing
"""
import json
from pathlib import Path

import pytest

from pynecore.core.broker.models import PositionLeg
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.store_helpers import (
    ENTRY_KIND_POSITION,
    STATE_CONFIRMED,
    adopt_untracked_position_legs,
)
from pynecore.types.strategy import ADOPTED_STARTUP_EXTRA_KEY


PLUGIN = "TestBroker"
SCRIPT_SOURCE = "// adopt_untracked_position_legs test\n"
SYMBOL = "BTCUSD"


def _open_run(store: BrokerStore, *, label: str | None = None) -> RunContext:
    return store.open_run(
        RunIdentity(
            strategy_id="adopt_test", symbol=SYMBOL, timeframe="60",
            account_id="testbroker-demo", label=label,
        ),
        script_source=SCRIPT_SOURCE,
        script_path="strategies/adopt_test.py",
    )


def _leg(
        leg_id: str, *,
        symbol: str = SYMBOL,
        side: str = "buy",
        qty: float = 0.01,
) -> PositionLeg:
    return PositionLeg(
        leg_id=leg_id, symbol=symbol, side=side, qty=qty,
        entry_price=100.0, open_time=1_700_000_000.0,
    )


def __test_adopts_untracked_leg_into_confirmed_row__(tmp_path: Path) -> None:
    """An untracked leg gets a confirmed row, a ref, and an audit event."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)

        adopted = adopt_untracked_position_legs(
            ctx, [_leg("deal-1", side="sell", qty=0.02)], symbol=SYMBOL,
        )

        assert adopted == 1
        row = ctx.get_order(f"__pyne_adopted__{SYMBOL}__deal-1")
        assert row is not None
        assert row.state == STATE_CONFIRMED
        assert row.side == "sell"
        assert row.qty == 0.02 and row.filled_qty == 0.02
        assert row.exchange_order_id == "deal-1"
        extras = row.extras or {}
        assert extras.get("kind") == ENTRY_KIND_POSITION
        assert extras.get(ADOPTED_STARTUP_EXTRA_KEY) is True
        assert extras.get("entry_filled_at")
        assert ctx.find_by_ref("deal_id", "deal-1") is not None

        cur = ctx._store._conn.execute(
            "SELECT payload FROM events WHERE kind = 'startup_position_adopted'",
        )
        rows = cur.fetchall()
        assert len(rows) == 1
        payload = json.loads(rows[0][0])
        assert payload == {"symbol": SYMBOL, "side": "sell", "qty": 0.02}


def __test_skips_already_tracked_and_reffed_legs__(tmp_path: Path) -> None:
    """A leg matched by a live row's exchange id or a deal_id ref is skipped."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "coid-live", symbol=SYMBOL, side="buy", qty=0.01,
            state="confirmed", exchange_order_id="deal-tracked",
        )
        ctx.upsert_order(
            "coid-reffed", symbol=SYMBOL, side="buy", qty=0.01,
            state="confirmed",
        )
        ctx.add_ref("coid-reffed", "deal_id", "deal-reffed")

        adopted = adopt_untracked_position_legs(
            ctx,
            [_leg("deal-tracked"), _leg("deal-reffed"), _leg("deal-new")],
            symbol=SYMBOL,
        )

        assert adopted == 1
        assert ctx.get_order(f"__pyne_adopted__{SYMBOL}__deal-tracked") is None
        assert ctx.get_order(f"__pyne_adopted__{SYMBOL}__deal-reffed") is None
        assert ctx.get_order(f"__pyne_adopted__{SYMBOL}__deal-new") is not None


def __test_skips_legs_owned_by_another_run__(tmp_path: Path) -> None:
    """A leg live under a sibling run label on the same account is not ours."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        other = _open_run(store, label="other")
        other.upsert_order(
            "coid-other", symbol=SYMBOL, side="sell", qty=0.01,
            state="confirmed", exchange_order_id="deal-foreign-run",
        )
        ctx = _open_run(store)

        adopted = adopt_untracked_position_legs(
            ctx, [_leg("deal-foreign-run")], symbol=SYMBOL,
        )

        assert adopted == 0
        assert ctx.get_order(
            f"__pyne_adopted__{SYMBOL}__deal-foreign-run",
        ) is None


def __test_skips_non_positive_qty_and_is_idempotent__(tmp_path: Path) -> None:
    """Zero-size legs never seed; a second pass adopts nothing new."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        legs = [_leg("deal-zero", qty=0.0), _leg("deal-real")]

        assert adopt_untracked_position_legs(ctx, legs, symbol=SYMBOL) == 1
        assert ctx.get_order(f"__pyne_adopted__{SYMBOL}__deal-zero") is None

        assert adopt_untracked_position_legs(ctx, legs, symbol=SYMBOL) == 0
        cur = ctx._store._conn.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE kind = 'startup_position_adopted'",
        )
        assert cur.fetchone()[0] == 1


def __test_foreign_symbol_leg_raises__(tmp_path: Path) -> None:
    """Account-wide input is a contract violation, not a silent skip."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)

        with pytest.raises(ValueError, match="symbol-scoped"):
            adopt_untracked_position_legs(
                ctx, [_leg("deal-x", symbol="EURUSD")], symbol=SYMBOL,
            )

        assert ctx.get_order("__pyne_adopted__EURUSD__deal-x") is None
