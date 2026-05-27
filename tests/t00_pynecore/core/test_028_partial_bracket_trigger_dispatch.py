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
from pynecore.core.broker.models import (
    OcaType,
)
from pynecore.core.broker.software_partial_bracket_engine import (
    PartialBracketLeg,
    SoftwarePartialBracketEngine,
)
from pynecore.core.broker.store_helpers import (
    LEG_KIND_SL_PARTIAL,
    LEG_KIND_TP_PARTIAL,
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
