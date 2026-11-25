"""
:class:`~pynecore.core.broker.parked_dispatch.ParkedDispatchResolver` against a real store.

The resolver awaits the venue lookup between reading a parked row and
rejecting it, so the row can change in between; and the rejection is several
writes that must land together, or a dispatch stays parked for the run.
"""
import asyncio
import logging
import sqlite3
from pathlib import Path

import pytest

from pynecore.core.broker import storage as storage_module
from pynecore.core.broker.parked_dispatch import (
    ParkedDispatchResolver,
    ParkedOutcome,
    ParkedVerdict,
    inherited_dispatch_in_flight,
    park_inherited_dispatches,
)
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, OrderRow, RunContext
from pynecore.core.broker.store_helpers import (
    ENTRY_KIND_POSITION,
    STATE_CONFIRMED,
    STATE_DISPOSITION_UNKNOWN,
    STATE_REJECTED,
    STATE_SUBMITTED,
    create_entry_order_row,
)

COID = 'coid-parked'
KEY = 'intent-entry'
LOG = logging.getLogger(__name__)


def _open_ctx(store: BrokerStore) -> RunContext:
    identity = RunIdentity(
        strategy_id='parked', symbol='BTCUSDT', timeframe='60',
        account_id='test-account', label=None,
    )
    return store.open_run(identity, script_source='// parked dispatch test')


def _park(ctx: RunContext) -> None:
    ctx.upsert_order(COID, symbol='BTCUSDT', side='buy', qty=1.0,
                     state=STATE_DISPOSITION_UNKNOWN, intent_key=KEY)
    ctx.record_park(COID, KEY)


def _resolver() -> ParkedDispatchResolver:
    # A clock far past the row's creation clears every grace window.
    return ParkedDispatchResolver(venue='Test', log=LOG, grace_s=0.0,
                                  misses_to_reject=1, clock=lambda: 4_000_000_000.0)


def _resolutions(ctx: RunContext) -> dict[str, str]:
    return {rec.coid: rec.resolution for rec in ctx.iter_pending_resolutions()}


def __test_a_row_confirmed_during_the_lookup_is_not_rejected__(tmp_path: Path) -> None:
    """A fill landing while the lookup is awaited wins over the stale verdict."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = _open_ctx(store)
        _park(ctx)

        async def lookup(_row: OrderRow) -> ParkedVerdict:
            ctx.upsert_order(COID, state=STATE_CONFIRMED,
                             exchange_order_id='venue-1', filled_qty=1.0)
            return ParkedVerdict(ParkedOutcome.ABSENT)

        rejected = asyncio.run(_resolver().resolve(ctx, lookup))

        assert rejected == []
        row = ctx.get_order(COID)
        assert row is not None
        assert row.state == STATE_CONFIRMED
        assert row.closed_ts_ms is None
        assert _resolutions(ctx) == {}


def __test_a_failed_resolution_write_leaves_the_row_parked_for_retry__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row's close and the engine's resolution commit or roll back together."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = _open_ctx(store)
        _park(ctx)
        resolver = _resolver()

        async def lookup(_row: OrderRow) -> ParkedVerdict:
            return ParkedVerdict(ParkedOutcome.ABSENT)

        def failing_record_resolution(_self: RunContext, _coid: str, _resolution: str) -> None:
            raise OSError('disk full')

        with monkeypatch.context() as patch:
            patch.setattr(storage_module.RunContext, 'record_resolution',
                          failing_record_resolution)
            with pytest.raises(OSError):
                asyncio.run(resolver.resolve(ctx, lookup))

        row = ctx.get_order(COID)
        assert row is not None
        assert row.state == STATE_DISPOSITION_UNKNOWN
        assert row.closed_ts_ms is None
        assert _resolutions(ctx) == {}

        assert asyncio.run(resolver.resolve(ctx, lookup)) == [COID]
        row = ctx.get_order(COID)
        assert row is not None
        assert row.state == STATE_REJECTED
        assert row.closed_ts_ms is not None
        assert _resolutions(ctx) == {COID: 'rejected'}


# === Dispatches an earlier process left pending ==========================

INHERITED = 'coid-inherited'
OWN = 'coid-own'


def _write_entry(ctx: RunContext, coid: str, *, key: str = KEY) -> None:
    create_entry_order_row(ctx, coid=coid, symbol='BTCUSDT', side='buy', qty=1.0,
                           intent_key=key, pine_entry_id=key, kind=ENTRY_KIND_POSITION,
                           order_type='market')


def _restarted_ctx(store: BrokerStore) -> RunContext:
    """An earlier process wrote :data:`INHERITED` and crashed before the answer."""
    earlier = _open_ctx(store)
    _write_entry(earlier, INHERITED)
    earlier.close()
    return _open_ctx(store)


def _absent_lookup(seen: list[str]):
    async def lookup(row: OrderRow) -> ParkedVerdict:
        seen.append(row.client_order_id)
        return ParkedVerdict(ParkedOutcome.ABSENT)
    return lookup


def __test_the_adopted_pending_rows_are_the_inherited_dispatches__(tmp_path: Path) -> None:
    """Only the rows adopted pending are inherited; this process's own rows are not."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = _restarted_ctx(store)
        _write_entry(ctx, OWN, key='other')

        assert ctx.is_inherited_dispatch(INHERITED)
        assert not ctx.is_inherited_dispatch(OWN)
        assert ctx.inherited_dispatch_ids() == frozenset({INHERITED})


def __test_an_inherited_submitted_row_is_looked_up_and_released__(tmp_path: Path) -> None:
    """The resolver asks the venue about an inherited ``submitted`` row, not about its own."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = _restarted_ctx(store)
        _write_entry(ctx, OWN, key='other')
        seen: list[str] = []

        rejected = asyncio.run(_resolver().resolve(ctx, _absent_lookup(seen)))

        assert seen == [INHERITED]
        assert rejected == [INHERITED]
        row = ctx.get_order(INHERITED)
        assert row is not None
        assert row.state == STATE_REJECTED
        assert row.closed_ts_ms is not None
        own = ctx.get_order(OWN)
        assert own is not None
        assert own.state == STATE_SUBMITTED


def __test_an_inherited_row_this_process_sends_again_is_its_own__(tmp_path: Path) -> None:
    """A send of this process under the inherited id makes the row mid-flight again."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = _restarted_ctx(store)
        _write_entry(ctx, INHERITED)
        seen: list[str] = []

        assert not ctx.is_inherited_dispatch(INHERITED)
        assert asyncio.run(_resolver().resolve(ctx, _absent_lookup(seen))) == []
        assert seen == []


def __test_an_inherited_row_sent_again_during_the_lookup_is_not_rejected__(
        tmp_path: Path,
) -> None:
    """The verdict was about the earlier send; a send of this process meanwhile wins."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = _restarted_ctx(store)

        async def lookup(_row: OrderRow) -> ParkedVerdict:
            _write_entry(ctx, INHERITED)
            return ParkedVerdict(ParkedOutcome.ABSENT)

        assert asyncio.run(_resolver().resolve(ctx, lookup)) == []
        row = ctx.get_order(INHERITED)
        assert row is not None
        assert row.state == STATE_SUBMITTED
        assert row.closed_ts_ms is None


def __test_inherited_entries_without_a_park_get_one__(tmp_path: Path) -> None:
    """A crash before the venue's answer left no park; startup gives the row one."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        earlier = _open_ctx(store)
        _write_entry(earlier, INHERITED)
        _write_entry(earlier, COID, key='parked')
        earlier.set_order_state(COID, STATE_DISPOSITION_UNKNOWN)
        earlier.record_park(COID, 'parked')
        earlier.record_resolution(COID, 'attached')
        earlier.upsert_order('coid-command', symbol='BTCUSDT', side='buy', qty=1.0,
                             state=STATE_SUBMITTED, intent_key='cmd',
                             extras={'kind': 'cancel'})
        earlier.close()
        ctx = _open_ctx(store)
        _write_entry(ctx, OWN, key='other')

        assert park_inherited_dispatches(ctx) == [INHERITED]

        _envelopes, parks = ctx.replay()
        assert set(parks) == {INHERITED, COID}
        assert parks[INHERITED].key == KEY
        assert parks[INHERITED].resolution is None
        inherited = ctx.get_order(INHERITED)
        assert inherited is not None
        assert parks[INHERITED].parked_ts_ms == inherited.created_ts_ms
        assert parks[COID].resolution == 'attached'
        assert park_inherited_dispatches(ctx) == []


def __test_an_inherited_dispatch_stays_in_flight_until_the_engine_retires_it__(
        tmp_path: Path,
) -> None:
    """Open while pending, and after the rejection until the engine consumed it."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = _restarted_ctx(store)
        park_inherited_dispatches(ctx)

        assert inherited_dispatch_in_flight(ctx, KEY) == INHERITED
        assert inherited_dispatch_in_flight(ctx, 'other') is None

        asyncio.run(_resolver().resolve(ctx, _absent_lookup([])))
        row = ctx.get_order(INHERITED)
        assert row is not None
        assert row.closed_ts_ms is not None
        assert _resolutions(ctx) == {INHERITED: 'rejected'}
        assert inherited_dispatch_in_flight(ctx, KEY) == INHERITED

        ctx.record_complete(KEY)
        assert inherited_dispatch_in_flight(ctx, KEY) is None


def __test_an_inherited_dispatch_that_landed_is_no_longer_in_flight__(tmp_path: Path) -> None:
    """A confirmed inherited row releases its intent, whatever its park says."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = _restarted_ctx(store)
        park_inherited_dispatches(ctx)

        ctx.upsert_order(INHERITED, state=STATE_CONFIRMED, exchange_order_id='venue-1',
                         filled_qty=1.0)

        assert inherited_dispatch_in_flight(ctx, KEY) is None


def __test_a_failed_inherited_park_parks_none_and_the_next_call_parks_all__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inherited parks are one write: none is left parked but unreported."""
    second = 'coid-inherited-2'
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        earlier = _open_ctx(store)
        _write_entry(earlier, INHERITED)
        _write_entry(earlier, second, key='second')
        earlier.close()
        ctx = _open_ctx(store)
        original = ctx.record_park
        calls = [0]

        def failing(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise sqlite3.OperationalError('disk I/O error')
            return original(*args, **kwargs)

        monkeypatch.setattr(ctx, 'record_park', failing)
        with pytest.raises(sqlite3.OperationalError):
            park_inherited_dispatches(ctx)
        assert ctx.replay()[1] == {}

        assert sorted(park_inherited_dispatches(ctx)) == sorted([INHERITED, second])
        assert park_inherited_dispatches(ctx) == []
