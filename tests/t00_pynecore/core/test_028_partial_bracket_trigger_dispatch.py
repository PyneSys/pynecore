"""Engine-isolated tests for the partial-bracket WATCH-phase settling API.

These tests exercise the three close-dispatch settling entry points
added on :class:`SoftwarePartialBracketEngine` to round-trip a
``triggering`` leg back to a terminal or re-armed state:

* :meth:`SoftwarePartialBracketEngine.confirm_trigger_dispatched`
* :meth:`SoftwarePartialBracketEngine.mark_trigger_dispatch_failed`
* :meth:`SoftwarePartialBracketEngine.mark_trigger_dispatch_unknown`

The sync-engine wiring that drives these methods
(:meth:`OrderSyncEngine._drive_partial_bracket_triggers` and
:meth:`OrderSyncEngine._dispatch_partial_bracket_close`) is covered
at the plugin-level integration layer (Capital.com plugin tests),
not here — these tests pin the state machine contract alone.
"""
import sqlite3

import pytest

from pynecore.core.broker import software_partial_bracket_engine as spbe_mod
from pynecore.core.broker.models import (
    ExchangePosition,
    OcaType,
)
from pynecore.core.broker.software_partial_bracket_engine import (
    PartialBracketLeg,
    SoftwarePartialBracketEngine,
)
from pynecore.core.broker.store_helpers import (
    LEG_KIND_SL_PARTIAL,
    LEG_KIND_TP_PARTIAL,
    LEG_KIND_TRAIL_PARTIAL,
    LEG_STATE_ARMED,
    LEG_STATE_TRIGGERED,
    LEG_STATE_TRIGGERING,
)


SYMBOL = "BTCUSDT"


def _mk_triggering_leg(
        *,
        coid: str = 'leg-tp',
        pine_id: str = 'exit_tp',
        from_entry: str = 'parent',
        leg_kind: str = LEG_KIND_TP_PARTIAL,
        oca_group: str | None = None,
        oca_type: str | None = None,
) -> PartialBracketLeg:
    intent_key = f"{pine_id}\x00{from_entry}"
    return PartialBracketLeg(
        coid=coid,
        symbol=SYMBOL,
        pine_id=pine_id,
        from_entry=from_entry,
        leg_kind=leg_kind,
        leg_state=LEG_STATE_TRIGGERING,
        side='sell',
        qty=0.5,
        intent_key=intent_key,
        parent_pine_entry_id=from_entry,
        parent_entry_dispatch_ref='parent-coid',
        intent_partial_qty=1.0,
        trigger_level=110.0,
        oca_group=oca_group,
        oca_type=oca_type,
    )


def __test_confirm_trigger_dispatched_flips_to_triggered_and_closes_row__():
    """``confirm_trigger_dispatched`` evicts the leg from the ledger."""
    engine = SoftwarePartialBracketEngine(store_ctx=None)
    leg = _mk_triggering_leg()
    # Hot-register a TRIGGERING leg directly (the public API only accepts
    # ACTIVE states; tests synthesise the state machine waypoint manually).
    engine._legs[leg.key] = leg
    engine._legs_by_parent.setdefault(
        (leg.symbol, leg.from_entry), set(),
    ).add(leg.key)

    cascaded = engine.confirm_trigger_dispatched(
        leg.key, close_pine_id='__pyne_partial_trigger__exit_tp\0parent\0tp_partial',
    )

    # Leg evicted from the in-memory ledger (terminal state, row closed).
    assert engine.get_leg(leg.key) is None
    # No OCA siblings → cascade returns empty.
    assert cascaded == []
    # The leg itself moved through TRIGGERED before eviction.
    assert leg.leg_state == LEG_STATE_TRIGGERED
    assert leg.extras['close_pine_id'] == (
        '__pyne_partial_trigger__exit_tp\0parent\0tp_partial'
    )


def __test_confirm_trigger_cascades_oca_siblings__():
    """A triggered TP cascades the sibling SL when both share an OCA-cancel group."""
    engine = SoftwarePartialBracketEngine(store_ctx=None)
    oca_group = '__partial_exit_exit_tp_parent__'
    tp = _mk_triggering_leg(
        coid='leg-tp', pine_id='exit_tp', leg_kind=LEG_KIND_TP_PARTIAL,
        oca_group=oca_group, oca_type=OcaType.CANCEL.value,
    )
    sl = _mk_triggering_leg(
        coid='leg-sl', pine_id='exit_sl', leg_kind=LEG_KIND_SL_PARTIAL,
        oca_group=oca_group, oca_type=OcaType.CANCEL.value,
    )
    # Sibling is ARMED (cascade target); TP is the one triggering.
    sl.leg_state = LEG_STATE_ARMED
    for leg in (tp, sl):
        engine._legs[leg.key] = leg
        engine._legs_by_parent.setdefault(
            (leg.symbol, leg.from_entry), set(),
        ).add(leg.key)
        engine._legs_by_oca_group.setdefault(oca_group, set()).add(leg.key)

    cascaded = engine.confirm_trigger_dispatched(
        tp.key, close_pine_id='__pyne_partial_trigger__exit_tp\0parent\0tp_partial',
    )

    assert {leg.pine_id for leg in cascaded} == {'exit_sl'}
    # Both evicted (TP via TRIGGERED, SL via CASCADED_CANCEL).
    assert engine.get_leg(tp.key) is None
    assert engine.get_leg(sl.key) is None


def __test_confirm_trigger_no_op_when_leg_missing__():
    """Idempotent: confirming a non-tracked key returns ``[]``."""
    engine = SoftwarePartialBracketEngine(store_ctx=None)
    cascaded = engine.confirm_trigger_dispatched(
        ('exit_tp', 'parent', LEG_KIND_TP_PARTIAL), close_pine_id='x',
    )
    assert cascaded == []


def __test_confirm_trigger_no_op_when_leg_not_in_triggering__():
    """Idempotent: confirm on an ARMED leg leaves it untouched."""
    engine = SoftwarePartialBracketEngine(store_ctx=None)
    leg = _mk_triggering_leg()
    leg.leg_state = LEG_STATE_ARMED
    engine._legs[leg.key] = leg
    engine._legs_by_parent.setdefault(
        (leg.symbol, leg.from_entry), set(),
    ).add(leg.key)

    cascaded = engine.confirm_trigger_dispatched(
        leg.key, close_pine_id='x',
    )

    assert cascaded == []
    # Still ARMED, still in ledger.
    assert engine.get_leg(leg.key) is leg
    assert leg.leg_state == LEG_STATE_ARMED


def __test_mark_trigger_dispatch_failed_rearms_leg_with_reason__():
    """A failed dispatch demotes TRIGGERING → ARMED with the reason recorded."""
    engine = SoftwarePartialBracketEngine(store_ctx=None)
    leg = _mk_triggering_leg()
    engine._legs[leg.key] = leg
    engine._legs_by_parent.setdefault(
        (leg.symbol, leg.from_entry), set(),
    ).add(leg.key)

    engine.mark_trigger_dispatch_failed(leg.key, reason='plugin_skipped:min_qty')

    assert engine.get_leg(leg.key) is leg
    assert leg.leg_state == LEG_STATE_ARMED
    assert leg.extras['trigger_failed_reason'] == 'plugin_skipped:min_qty'


def __test_mark_trigger_dispatch_unknown_rearms_leg_with_reason__():
    """A parked dispatch demotes TRIGGERING → ARMED with the reason recorded."""
    engine = SoftwarePartialBracketEngine(store_ctx=None)
    leg = _mk_triggering_leg()
    engine._legs[leg.key] = leg
    engine._legs_by_parent.setdefault(
        (leg.symbol, leg.from_entry), set(),
    ).add(leg.key)

    engine.mark_trigger_dispatch_unknown(leg.key, reason='dispatch_parked:timeout')

    assert engine.get_leg(leg.key) is leg
    assert leg.leg_state == LEG_STATE_ARMED
    assert leg.extras['trigger_unknown_reason'] == 'dispatch_parked:timeout'


def __test_mark_trigger_settling_idempotent_when_leg_already_armed__():
    """Calling either settling method on an ARMED leg is a no-op."""
    engine = SoftwarePartialBracketEngine(store_ctx=None)
    leg = _mk_triggering_leg()
    leg.leg_state = LEG_STATE_ARMED
    engine._legs[leg.key] = leg
    engine._legs_by_parent.setdefault(
        (leg.symbol, leg.from_entry), set(),
    ).add(leg.key)

    engine.mark_trigger_dispatch_failed(leg.key, reason='ignored')
    engine.mark_trigger_dispatch_unknown(leg.key, reason='ignored')

    assert leg.leg_state == LEG_STATE_ARMED
    assert 'trigger_failed_reason' not in leg.extras
    assert 'trigger_unknown_reason' not in leg.extras


@pytest.mark.parametrize('phase', ['activation', 'move'])
def __test_a_trail_write_failure_keeps_the_observed_high_for_the_next_tick__(
        monkeypatch, phase):
    """A tick whose trail write failed still moves the stop for the next tick.

    A long parent trails 5 below the high. The tick at 110 activates the trail
    (activation 110) or moves it (stop 90), and its store write fails. The
    next tick at 98 has crossed the 105 stop that tick set, so the leg
    triggers, exactly as without the failure, and the row gets the 105 stop.
    """
    engine = SoftwarePartialBracketEngine(store_ctx=None)
    leg = _mk_triggering_leg(leg_kind=LEG_KIND_TRAIL_PARTIAL)
    leg.leg_state = LEG_STATE_ARMED
    leg.trigger_offset = 5.0
    if phase == 'activation':
        leg.trigger_level = None
        leg.trail_activation_level = 110.0
    else:
        leg.trigger_level = 90.0
    engine.register_leg(leg)
    written: list[tuple[str, float | None]] = []
    failures = [sqlite3.OperationalError("disk I/O error")]

    def _write(_store_ctx, *, new_leg_state, trigger_level=None, **_kwargs):
        if failures:
            raise failures.pop()
        written.append((new_leg_state, trigger_level))

    engine._store_ctx = object()  # type: ignore[assignment]
    monkeypatch.setattr(spbe_mod, 'update_engine_trigger_partial_leg_state', _write)
    parent = ExchangePosition(
        symbol=SYMBOL, side='long', size=1.0, entry_price=100.0, unrealized_pnl=0.0,
        liquidation_price=None, leverage=1.0, margin_mode='cross')

    with pytest.raises(sqlite3.OperationalError):
        engine.on_price_tick(symbol=SYMBOL, last_price=110.0, bid=None, ask=None,
                             parent_snapshot=parent)
    triggering = engine.on_price_tick(symbol=SYMBOL, last_price=98.0, bid=None, ask=None,
                                      parent_snapshot=parent)

    assert triggering == [leg]
    assert leg.trigger_level == 105.0
    assert written == [(LEG_STATE_ARMED, 105.0), (LEG_STATE_TRIGGERING, None)]


@pytest.mark.parametrize('phase', ['activation', 'move'])
def __test_a_restart_after_a_failed_trail_write_trails_from_the_persisted_row__(
        tmp_path, monkeypatch, phase):
    """A restart after a failed trail write trails from what the row holds.

    The price the failed write kept (:attr:`PartialBracketLeg.unapplied_trail_price`)
    lives in memory only. The process that held it ends, and the restarted
    ledger rebuilds the leg from its row: the stop (or the activation level)
    of the last committed write and no kept price. Fresh ticks then trail it
    as they would any persisted leg, and it triggers on a tick that crosses
    the stop the row and those ticks set, never on a stop the lost price set.
    """
    from pynecore.core.broker.run_identity import RunIdentity
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import create_engine_trigger_partial_leg_row
    real_write = spbe_mod.update_engine_trigger_partial_leg_state
    failures = [sqlite3.OperationalError("disk I/O error")]

    def _write(store_ctx, **kwargs):
        if failures:
            raise failures.pop()
        real_write(store_ctx, **kwargs)

    parent = ExchangePosition(
        symbol=SYMBOL, side='long', size=1.0, entry_price=100.0, unrealized_pnl=0.0,
        liquidation_price=None, leverage=1.0, margin_mode='cross')
    identity = RunIdentity(strategy_id="t028", symbol=SYMBOL, timeframe="60",
                           account_id="testbroker-demo", label=None)
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(identity, script_source="src", script_path="t.py")
        create_engine_trigger_partial_leg_row(
            ctx, coid='leg-trail', symbol=SYMBOL, side='sell', qty=0.5,
            intent_key="exit_tr\x00parent", pine_entry_id='exit_tr', from_entry='parent',
            leg_kind=LEG_KIND_TRAIL_PARTIAL, leg_state=LEG_STATE_ARMED,
            parent_pine_entry_id='parent', parent_entry_dispatch_ref='parent-coid',
            intent_partial_qty=0.5, trigger_offset=5.0,
            trigger_level=None if phase == 'activation' else 90.0,
            trail_activation_level=110.0 if phase == 'activation' else None,
        )
        engine = SoftwarePartialBracketEngine(store_ctx=ctx)
        engine.restart_replay()
        monkeypatch.setattr(spbe_mod, 'update_engine_trigger_partial_leg_state', _write)
        with pytest.raises(sqlite3.OperationalError):
            engine.on_price_tick(symbol=SYMBOL, last_price=110.0, bid=None, ask=None,
                                 parent_snapshot=parent)
        assert next(iter(engine.iter_legs())).unapplied_trail_price == 110.0

        restarted = SoftwarePartialBracketEngine(store_ctx=ctx)
        restarted.restart_replay()
        leg = next(iter(restarted.iter_legs()))
        assert leg.unapplied_trail_price is None
        if phase == 'activation':
            assert (leg.trail_activation_level, leg.trigger_level) == (110.0, None)
        else:
            assert (leg.trail_activation_level, leg.trigger_level) == (None, 90.0)

        def tick(price: float) -> list[PartialBracketLeg]:
            return restarted.on_price_tick(symbol=SYMBOL, last_price=price, bid=None,
                                           ask=None, parent_snapshot=parent)

        # 98 crosses the 105 stop the lost tick would have set: no trigger.
        assert tick(98.0) == []
        if phase == 'activation':
            assert (leg.trail_activation_level, leg.trigger_level) == (110.0, None)
            assert tick(112.0) == []
            assert leg.trigger_level == 107.0
            assert tick(106.0) == [leg]
        else:
            assert leg.trigger_level == 93.0
            assert tick(92.0) == [leg]
        row = ctx.get_order('leg-trail')
        assert row is not None and row.extras['leg_state'] == LEG_STATE_TRIGGERING
        assert row.extras['trigger_level'] == leg.trigger_level
        ctx.close()


@pytest.mark.parametrize('phase', ['activation', 'move'])
def __test_a_trail_write_failure_keeps_the_observed_high_for_every_trail_of_the_tick__(
        monkeypatch, phase):
    """A failed trail write that aborts the tick still leaves its quote to the later trails.

    Two independent trails of a long parent follow 5 below the high. The tick
    at 110 activates (activation 110) or moves (stop 90) both, but the first
    leg's store write fails and ends the tick before the second leg sees the
    quote. The next tick at 98 has crossed the 105 stop the 110 quote set for
    both, so both trigger, exactly as without the failure.
    """
    engine = SoftwarePartialBracketEngine(store_ctx=None)
    legs = []
    for pine_id in ('exit_a', 'exit_b'):
        leg = _mk_triggering_leg(coid=f'leg-{pine_id}', pine_id=pine_id,
                                 leg_kind=LEG_KIND_TRAIL_PARTIAL)
        leg.leg_state = LEG_STATE_ARMED
        leg.trigger_offset = 5.0
        if phase == 'activation':
            leg.trigger_level = None
            leg.trail_activation_level = 110.0
        else:
            leg.trigger_level = 90.0
        engine.register_leg(leg)
        legs.append(leg)
    failures = [sqlite3.OperationalError("disk I/O error")]

    def _write(_store_ctx, **_kwargs):
        if failures:
            raise failures.pop()

    engine._store_ctx = object()  # type: ignore[assignment]
    monkeypatch.setattr(spbe_mod, 'update_engine_trigger_partial_leg_state', _write)
    parent = ExchangePosition(
        symbol=SYMBOL, side='long', size=2.0, entry_price=100.0, unrealized_pnl=0.0,
        liquidation_price=None, leverage=1.0, margin_mode='cross')

    with pytest.raises(sqlite3.OperationalError):
        engine.on_price_tick(symbol=SYMBOL, last_price=110.0, bid=None, ask=None,
                             parent_snapshot=parent)
    triggering = engine.on_price_tick(symbol=SYMBOL, last_price=98.0, bid=None, ask=None,
                                      parent_snapshot=parent)

    assert triggering == legs
    assert [leg.trigger_level for leg in legs] == [105.0, 105.0]


def __test_helper_held_bracket(tmp_path, *, sl_qty: float = 0.5):
    """A stored TP/SL bracket whose TP is held on close ``close-1``; returns ``(store, ctx, engine)``."""
    from pynecore.core.broker.run_identity import RunIdentity
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import create_engine_trigger_partial_leg_row
    identity = RunIdentity(strategy_id="t028", symbol=SYMBOL, timeframe="60",
                           account_id="testbroker-demo", label=None)
    store = BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker")
    ctx = store.open_run(identity, script_source="src", script_path="t.py")
    for coid, kind, qty, level in (('leg-tp', LEG_KIND_TP_PARTIAL, 0.5, 120.0),
                                   ('leg-sl', LEG_KIND_SL_PARTIAL, sl_qty, 90.0)):
        create_engine_trigger_partial_leg_row(
            ctx, coid=coid, symbol=SYMBOL, side='sell', qty=qty,
            intent_key="X\x00parent", pine_entry_id='X', from_entry='parent',
            leg_kind=kind, leg_state=LEG_STATE_ARMED, parent_pine_entry_id='parent',
            parent_entry_dispatch_ref='parent-coid', intent_partial_qty=0.5,
            trigger_level=level, oca_group='__partial_exit_X_parent__',
            oca_type=OcaType.CANCEL.value,
        )
    engine = SoftwarePartialBracketEngine(store_ctx=ctx)
    engine.restart_replay()
    tp_key = ('X', 'parent', LEG_KIND_TP_PARTIAL)
    engine._transition(engine.get_leg(tp_key), LEG_STATE_TRIGGERING, close_row=False)
    engine.record_trigger_close_attempt(tp_key, close_coid='close-1')
    return store, ctx, engine


def _leg_rows(ctx) -> dict:
    from pynecore.core.broker.store_helpers import STATE_PARTIAL_BRACKET_LEG
    return {row.client_order_id: (row.extras.get('leg_state'), row.qty, row.closed_ts_ms is None)
            for row in ctx.iter_orders_in_state_for_run_id(STATE_PARTIAL_BRACKET_LEG)}


def __test_a_partially_executed_close_re_arms_the_rest_and_sizes_its_sibling__(tmp_path):
    """The held TP's close ended after 0.2 of 0.5: the TP re-arms for 0.3 and
    its OCA sibling is sized down to 0.3 too, in the rows and in memory. The
    spent close id is kept on the leg's row and survives a restart."""
    store, ctx, engine = __test_helper_held_bracket(tmp_path)
    try:
        tp_key = ('X', 'parent', LEG_KIND_TP_PARTIAL)
        engine.release_held_trigger_partially(
            tp_key, executed_qty=0.2, reason='held_close_partially_executed')
        tp = engine.get_leg(tp_key)
        sl = engine.get_leg(('X', 'parent', LEG_KIND_SL_PARTIAL))
        assert tp is not None and sl is not None
        assert (tp.leg_state, tp.close_attempt_coid) == (LEG_STATE_ARMED, None)
        assert tp.qty == pytest.approx(0.3) and sl.qty == pytest.approx(0.3)
        assert _leg_rows(ctx) == {'leg-tp': (LEG_STATE_ARMED, pytest.approx(0.3), True),
                                  'leg-sl': (LEG_STATE_ARMED, pytest.approx(0.3), True)}
        assert engine.close_coid_spent(tp_key, 'close-1')
        restarted = SoftwarePartialBracketEngine(store_ctx=ctx)
        restarted.restart_replay()
        assert restarted.close_coid_spent(tp_key, 'close-1')
        assert not restarted.close_coid_spent(tp_key, 'close-2')
    finally:
        ctx.close()
        store.close()


def __test_a_partial_settlement_retried_after_a_failed_write_sizes_no_sibling_twice__(
        tmp_path, monkeypatch):
    """The leg's own row write fails after its sibling was sized: the leg stays
    held, and the retried settlement leaves the sibling at 0.3."""
    store, ctx, engine = __test_helper_held_bracket(tmp_path)
    try:
        tp_key = ('X', 'parent', LEG_KIND_TP_PARTIAL)
        real_write = spbe_mod.update_engine_trigger_partial_leg_state

        def _write(store_ctx, **kwargs):
            if kwargs['coid'] == 'leg-tp' and kwargs['new_leg_state'] == LEG_STATE_ARMED:
                raise sqlite3.OperationalError("disk I/O error")
            real_write(store_ctx, **kwargs)

        monkeypatch.setattr(spbe_mod, 'update_engine_trigger_partial_leg_state', _write)
        with pytest.raises(sqlite3.OperationalError):
            engine.release_held_trigger_partially(tp_key, executed_qty=0.2, reason='r')
        assert engine.get_leg(tp_key).leg_state == LEG_STATE_TRIGGERING
        monkeypatch.setattr(spbe_mod, 'update_engine_trigger_partial_leg_state', real_write)
        engine.release_held_trigger_partially(tp_key, executed_qty=0.2, reason='r')
        assert _leg_rows(ctx) == {'leg-tp': (LEG_STATE_ARMED, pytest.approx(0.3), True),
                                  'leg-sl': (LEG_STATE_ARMED, pytest.approx(0.3), True)}
    finally:
        ctx.close()
        store.close()


def __test_a_partial_close_that_used_up_a_sibling_cancels_it__(tmp_path):
    """A sibling holding no more than the executed quantity has nothing left to
    protect: it is cancelled, and the leg re-arms for its own rest."""
    store, ctx, engine = __test_helper_held_bracket(tmp_path, sl_qty=0.2)
    try:
        engine.release_held_trigger_partially(
            ('X', 'parent', LEG_KIND_TP_PARTIAL), executed_qty=0.2, reason='r')
        assert engine.get_leg(('X', 'parent', LEG_KIND_SL_PARTIAL)) is None
        rows = _leg_rows(ctx)
        assert rows['leg-sl'][0] == 'cascaded_cancel' and not rows['leg-sl'][2]
        assert rows['leg-tp'] == (LEG_STATE_ARMED, pytest.approx(0.3), True)
    finally:
        ctx.close()
        store.close()


def __test_a_refused_close_is_spent_for_the_leg_across_a_restart__(tmp_path):
    """A released close's id stays spent on the leg's row; a later send of the
    leg is refused that id even when a restart lost the in-memory re-anchor."""
    store, ctx, engine = __test_helper_held_bracket(tmp_path)
    try:
        tp_key = ('X', 'parent', LEG_KIND_TP_PARTIAL)
        engine.release_held_trigger(tp_key, reason='held_close_refused')
        restarted = SoftwarePartialBracketEngine(store_ctx=ctx)
        restarted.restart_replay()
        assert restarted.close_coid_spent(tp_key, 'close-1')
    finally:
        ctx.close()
        store.close()
