"""
Behavior tests for the protection level guard
(:mod:`pynecore.core.broker.protection_guard`) and the plugin-side
:meth:`BrokerPlugin.report_external_intervention` it reports through.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore.core.broker.exceptions import UnexpectedCancelError
from pynecore.core.broker.models import QuarantineKind
from pynecore.core.broker.protection_guard import (
    PROTECTION_LEVELS_EXTRA,
    ProtectionLevelGuard,
    ProtectionLevels,
    rebaseline_after_resume,
)
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.plugin.broker import BrokerPlugin

TICK = 0.01


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _guard(store_ctx: RunContext | None = None):
    reports: list[tuple[str, dict]] = []
    clock = _Clock()
    guard = ProtectionLevelGuard(
        lambda reason, context: reports.append((reason, context)),
        store_ctx=store_ctx, clock=clock,
    )
    return guard, reports, clock


def _open_run(store: BrokerStore) -> RunContext:
    return store.open_run(
        RunIdentity(strategy_id="t161", symbol="BTCUSDT", timeframe="60",
                    account_id="testbroker-demo", label=None),
        script_source="src", script_path="t161.py",
    )


def _seed_row(ctx: RunContext, coid: str) -> None:
    ctx.upsert_order(coid, symbol="BTCUSDT", side="sell", qty=1.0,
                     state="confirmed", extras={})


def _place(guard, key: str, levels: ProtectionLevels, **kw) -> None:
    guard.expect(key, levels, **kw)
    guard.settle(key, levels)


BRACKET = ProtectionLevels(stop=95.0, limit=110.0)


def __test_own_levels_never_report__():
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    for _ in range(3):
        assert not guard.observe("p1", ProtectionLevels(stop=95.004, limit=110.0), tick=TICK)
    assert reports == []


def __test_a_moved_stop_reports_after_two_polls__():
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    moved = ProtectionLevels(stop=97.0, limit=110.0)
    assert not guard.observe("p1", moved, tick=TICK)
    assert guard.observe("p1", moved, tick=TICK)
    assert len(reports) == 1
    reason, context = reports[0]
    assert "p1" in reason
    assert context['expected'] == BRACKET.as_list()
    assert context['observed'] == moved.as_list()
    # Reported once per key.
    assert not guard.observe("p1", moved, tick=TICK)
    assert len(reports) == 1


def __test_a_push_reports_at_once_and_a_removed_level_counts__():
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    assert guard.observe("p1", ProtectionLevels(stop=None, limit=110.0),
                         tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_two_different_mismatches_do_not_confirm_each_other__():
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    assert not guard.observe("p1", ProtectionLevels(stop=97.0, limit=110.0), tick=TICK)
    assert not guard.observe("p1", ProtectionLevels(stop=98.0, limit=110.0), tick=TICK)
    assert guard.observe("p1", ProtectionLevels(stop=98.0, limit=110.0), tick=TICK)
    assert len(reports) == 1


def __test_an_own_write_in_flight_and_an_older_snapshot_are_not_judged__():
    guard, reports, clock = _guard()
    _place(guard, "p1", BRACKET)
    clock.now = 1_010.0
    read_started = clock.now
    clock.now = 1_011.0
    amended = ProtectionLevels(stop=97.0, limit=110.0)
    guard.expect("p1", amended)
    # In flight: nothing is judged, not even the old level.
    for _ in range(3):
        assert not guard.observe("p1", ProtectionLevels(stop=99.0, limit=110.0), tick=TICK)
    guard.settle("p1", amended)
    # A snapshot whose read started before the write is stale.
    for _ in range(3):
        assert not guard.observe("p1", BRACKET, tick=TICK, read_started_at=read_started)
    assert reports == []


def __test_a_late_push_of_the_previous_own_level_is_accepted__():
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    amended = ProtectionLevels(stop=97.0, limit=110.0)
    _place(guard, "p1", amended)
    # The venue's push for the earlier level arrives after the amend's ACK.
    assert not guard.observe("p1", BRACKET, tick=TICK, immediate=True)
    # Once the venue shows the new level, the old one is no longer accepted.
    assert not guard.observe("p1", amended, tick=TICK)
    assert guard.observe("p1", BRACKET, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_returning_to_an_earlier_own_level_retires_the_one_between__():
    guard, reports, _clock = _guard()
    amended = ProtectionLevels(stop=97.0, limit=110.0)
    _place(guard, "p1", BRACKET)
    _place(guard, "p1", amended)
    _place(guard, "p1", BRACKET)
    assert not guard.observe("p1", BRACKET, tick=TICK, immediate=True)
    assert guard.observe("p1", amended, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_an_unknown_write_stays_acceptable_until_the_venue_decides__():
    """A push never resolves an unknown write: it may predate a late landing."""
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    guard.expect("p1", ProtectionLevels(stop=97.0, limit=110.0))
    guard.settle("p1", None)
    for _ in range(3):
        assert not guard.observe("p1", BRACKET, tick=TICK)
    assert not guard.observe("p1", ProtectionLevels(stop=97.0, limit=110.0),
                             tick=TICK, immediate=True)
    assert reports == []


def __test_a_refused_write_stops_being_acceptable_at_once__():
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    refused = ProtectionLevels(stop=97.0, limit=110.0)
    guard.expect("p1", refused)
    guard.settle("p1", None, rejected=refused)
    assert not guard.observe("p1", BRACKET, tick=TICK, immediate=True)
    # Someone else moves the stop to the very level the venue refused.
    assert guard.observe("p1", refused, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_a_refused_first_write_leaves_the_key_unguarded__():
    guard, reports, _clock = _guard()
    guard.expect("p1", BRACKET)
    guard.settle("p1", None, rejected=BRACKET)
    assert not guard.known("p1")
    assert not guard.observe("p1", ProtectionLevels(stop=90.0), tick=TICK, immediate=True)
    assert reports == []


def __test_snapshots_without_an_unknown_write_retire_it__():
    """Two fresh snapshots of the confirmed level: the unknown write did not land."""
    guard, reports, clock = _guard()
    _place(guard, "p1", BRACKET)
    unknown = ProtectionLevels(stop=97.0, limit=110.0)
    guard.expect("p1", unknown)
    guard.settle("p1", None)
    clock.now += 1.0
    assert not guard.observe("p1", BRACKET, tick=TICK, read_started_at=clock.now)
    assert not guard.observe("p1", BRACKET, tick=TICK, read_started_at=clock.now)
    # A later manual move to that level is someone else's.
    assert not guard.observe("p1", unknown, tick=TICK, read_started_at=clock.now)
    assert guard.observe("p1", unknown, tick=TICK, read_started_at=clock.now)
    assert len(reports) == 1


def __test_an_unknown_write_seen_on_the_venue_landed__():
    guard, reports, clock = _guard()
    _place(guard, "p1", BRACKET)
    landed = ProtectionLevels(stop=97.0, limit=110.0)
    guard.expect("p1", landed)
    guard.settle("p1", None)
    clock.now += 1.0
    assert not guard.observe("p1", landed, tick=TICK, read_started_at=clock.now)
    # The level it replaced is history now.
    assert guard.observe("p1", BRACKET, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_a_late_answer_to_an_older_write_never_outranks_a_newer_one__():
    """Overlapping writes answered out of order keep their write order."""
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    first = ProtectionLevels(stop=97.0, limit=110.0)
    second = ProtectionLevels(stop=99.0, limit=110.0)
    guard.expect("p1", first)
    guard.expect("p1", second)
    guard.settle("p1", second)
    guard.settle("p1", first)
    # The venue's delayed push for the first write, then the second as polled.
    assert not guard.observe("p1", first, tick=TICK, immediate=True)
    for _ in range(3):
        assert not guard.observe("p1", second, tick=TICK)
    assert reports == []


def _settle_overlapping_repeat(guard, first, second) -> None:
    """Writes first, second, first again; answered first, second, a refusal of first.

    Without write ids the answers cannot say which write of ``first`` landed
    and which was refused: the venue may end up carrying either level.
    """
    _place(guard, "p1", BRACKET)
    guard.expect("p1", first)
    guard.expect("p1", second)
    guard.expect("p1", first)
    guard.settle("p1", first)
    guard.settle("p1", second)
    guard.settle("p1", None, rejected=first)


def __test_an_ambiguous_repeat_where_the_earlier_write_landed_never_reports__():
    """The earlier write of ``first`` landed, then ``second``: the venue carries ``second``."""
    guard, reports, clock = _guard()
    first = ProtectionLevels(stop=97.0, limit=110.0)
    second = ProtectionLevels(stop=99.0, limit=110.0)
    _settle_overlapping_repeat(guard, first, second)
    # The venue's delayed push for the earlier write, then ``second`` as polled.
    assert not guard.observe("p1", first, tick=TICK, immediate=True)
    clock.now += 1.0
    for _ in range(3):
        assert not guard.observe("p1", second, tick=TICK, read_started_at=clock.now)
    assert reports == []
    # Fresh snapshots of ``second`` decided it: ``first`` is no longer the bot's.
    assert guard.observe("p1", first, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_an_ambiguous_repeat_where_the_later_write_landed_never_reports__():
    """The earlier write of ``first`` was refused, the later one landed after
    ``second``: the venue carries ``first``."""
    guard, reports, clock = _guard()
    first = ProtectionLevels(stop=97.0, limit=110.0)
    second = ProtectionLevels(stop=99.0, limit=110.0)
    _settle_overlapping_repeat(guard, first, second)
    # The venue's delayed push for ``second``, then ``first`` as polled.
    assert not guard.observe("p1", second, tick=TICK, immediate=True)
    clock.now += 1.0
    for _ in range(3):
        assert not guard.observe("p1", first, tick=TICK, read_started_at=clock.now)
    assert reports == []
    # The venue carried the later ``first``: ``second`` is history.
    assert guard.observe("p1", second, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_repeated_echoes_of_an_ambiguous_level_resolve_only_one_write__():
    """The earlier write of ``first`` landed, then ``second``: the venue pushes the
    earlier write twice. Repeated views of one level prove one write, so the later,
    refused write of ``first`` never retires ``second``."""
    guard, reports, clock = _guard()
    first = ProtectionLevels(stop=97.0, limit=110.0)
    second = ProtectionLevels(stop=99.0, limit=110.0)
    _settle_overlapping_repeat(guard, first, second)
    assert not guard.observe("p1", first, tick=TICK, immediate=True)
    assert not guard.observe("p1", first, tick=TICK, immediate=True)
    clock.now += 1.0
    for _ in range(3):
        assert not guard.observe("p1", second, tick=TICK, read_started_at=clock.now)
    assert reports == []
    # Fresh snapshots of ``second`` decided it: ``first`` is no longer the bot's.
    assert guard.observe("p1", first, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_fresh_snapshots_of_an_ambiguous_repeat_retire_the_level_between__():
    """The earlier write of ``first`` was refused, the later one landed after
    ``second``, and only fresh snapshots of ``first`` follow: they settle the
    venue on the later ``first``, so a manual move back to ``second`` reports."""
    guard, reports, clock = _guard()
    first = ProtectionLevels(stop=97.0, limit=110.0)
    second = ProtectionLevels(stop=99.0, limit=110.0)
    _settle_overlapping_repeat(guard, first, second)
    clock.now += 1.0
    for _ in range(3):
        assert not guard.observe("p1", first, tick=TICK, read_started_at=clock.now)
    assert reports == []
    assert not guard.observe("p1", second, tick=TICK, read_started_at=clock.now)
    assert guard.observe("p1", second, tick=TICK, read_started_at=clock.now)
    assert len(reports) == 1


def __test_a_refused_revert_keeps_the_landed_write_of_that_level__():
    """Refusing a revert to an earlier level retires only the refused write: the
    earlier write of that level landed and its late push is still the bot's."""
    guard, reports, _clock = _guard()
    first = ProtectionLevels(stop=97.0, limit=110.0)
    second = ProtectionLevels(stop=99.0, limit=110.0)
    _place(guard, "p1", first)
    _place(guard, "p1", second)
    guard.expect("p1", first)
    guard.settle("p1", None, rejected=first)
    # The venue's delayed push for the first, landed write.
    assert not guard.observe("p1", first, tick=TICK, immediate=True)
    for _ in range(3):
        assert not guard.observe("p1", second, tick=TICK)
    assert reports == []
    # The venue carried the second write: the first one is history.
    assert guard.observe("p1", first, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_settle_by_write_id_closes_exactly_that_write__():
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    first = ProtectionLevels(stop=97.0, limit=110.0)
    second = ProtectionLevels(stop=99.0, limit=110.0)
    early = guard.expect("p1", first)
    guard.expect("p1", second)
    late = guard.expect("p1", first)
    guard.settle("p1", None, rejected=first, write=early)
    guard.settle("p1", second)
    guard.settle("p1", first, write=late)
    # The later write of 97 landed last: the venue carries it, 99 is history.
    assert not guard.observe("p1", first, tick=TICK)
    assert guard.observe("p1", second, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_unknown_writes_survive_a_restart__(tmp_path: Path):
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = _open_run(store)
        _seed_row(ctx, "leg-1")
        guard, _reports, _clock = _guard(ctx)
        _place(guard, "p1", BRACKET, row_coid="leg-1")
        unknown = ProtectionLevels(stop=97.0, limit=110.0)
        guard.expect("p1", unknown)
        guard.settle("p1", None)

        restarted, reports, clock = _guard(ctx)
        clock.now += 5.0
        assert not restarted.observe("p1", BRACKET, tick=TICK, read_started_at=clock.now)
        assert not restarted.observe("p1", BRACKET, tick=TICK, read_started_at=clock.now)
        assert restarted.observe("p1", unknown, tick=TICK, immediate=True)
        assert len(reports) == 1


def __test_a_trailing_stop_is_judged_by_flag_distance_and_take_profit__():
    guard, reports, _clock = _guard()
    trail = ProtectionLevels(stop=95.0, limit=110.0, trailing=True, trail_distance=5.0)
    _place(guard, "p1", trail)
    # The venue moves the stop by itself.
    assert not guard.observe("p1", ProtectionLevels(stop=99.0, limit=110.0, trailing=True,
                                                    trail_distance=5.0), tick=TICK, immediate=True)
    assert not guard.observe("p1", ProtectionLevels(stop=None, limit=110.0, trailing=True,
                                                    trail_distance=None), tick=TICK, immediate=True)
    assert reports == []
    assert guard.observe("p1", ProtectionLevels(stop=99.0, limit=110.0, trailing=False),
                         tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_a_changed_trail_distance_reports__():
    guard, reports, _clock = _guard()
    _place(guard, "p1", ProtectionLevels(limit=110.0, trailing=True,
                                                      trail_distance=5.0))
    assert guard.observe("p1", ProtectionLevels(limit=110.0, trailing=True, trail_distance=8.0),
                         tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_an_unknown_key_is_never_judged__():
    guard, reports, _clock = _guard()
    assert not guard.observe("nope", BRACKET, tick=TICK, immediate=True)
    assert reports == []


def __test_levels_survive_a_restart_and_forget_clears_them__(tmp_path: Path):
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = _open_run(store)
        _seed_row(ctx, "leg-1")
        guard, _reports, _clock = _guard(ctx)
        _place(guard, "p1", BRACKET, row_coid="leg-1")
        row = ctx.get_order("leg-1")
        assert row is not None and row.extras[PROTECTION_LEVELS_EXTRA]['key'] == "p1"

        restarted, reports, _clock = _guard(ctx)
        assert restarted.known("p1")
        moved = ProtectionLevels(stop=97.0, limit=110.0)
        assert not restarted.observe("p1", BRACKET, tick=TICK, immediate=True)
        assert restarted.observe("p1", moved, tick=TICK, immediate=True)
        assert len(reports) == 1

        restarted.forget("p1")
        row = ctx.get_order("leg-1")
        assert row is not None and PROTECTION_LEVELS_EXTRA not in row.extras


def __test_moving_a_key_to_another_row_drops_the_old_copy__(tmp_path: Path):
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = _open_run(store)
        _seed_row(ctx, "leg-1")
        _seed_row(ctx, "leg-2")
        guard, _reports, _clock = _guard(ctx)
        _place(guard, "p1", BRACKET, row_coid="leg-1")
        _place(guard, "p1", BRACKET, row_coid="leg-2")
        old = ctx.get_order("leg-1")
        new = ctx.get_order("leg-2")
        assert old is not None and PROTECTION_LEVELS_EXTRA not in old.extras
        assert new is not None and new.extras[PROTECTION_LEVELS_EXTRA]['key'] == "p1"


def __test_adopt_replaces_the_accepted_levels__():
    guard, reports, _clock = _guard()
    _place(guard, "p1", BRACKET)
    resized = ProtectionLevels(stop=96.0, limit=None)
    guard.adopt("p1", resized)
    assert not guard.observe("p1", resized, tick=TICK, immediate=True)
    assert guard.observe("p1", BRACKET, tick=TICK, immediate=True)
    assert len(reports) == 1


def __test_a_resume_adopts_the_venue_view_as_the_new_baseline__(tmp_path: Path):
    """After an operator resume the operator's levels are accepted, not re-reported."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = _open_run(store)
        _seed_row(ctx, "leg-1")
        guard, _reports, _clock = _guard(ctx)
        _place(guard, "p1", BRACKET, row_coid="leg-1")

        rebaseline_after_resume(ctx)
        restarted, reports, _clock = _guard(ctx)
        operator_levels = ProtectionLevels(stop=90.0, limit=None)
        assert not restarted.observe("p1", operator_levels, tick=TICK, immediate=True)
        assert not restarted.observe("p1", operator_levels, tick=TICK, immediate=True)
        assert reports == []
        # A later edit is judged against the adopted baseline.
        assert restarted.observe("p1", ProtectionLevels(stop=93.0, limit=None),
                                 tick=TICK, immediate=True)
        assert len(reports) == 1


# === BrokerPlugin.report_external_intervention =============================

def _plugin(policy: str, sink=None, store_ctx=None):
    return SimpleNamespace(on_unexpected_cancel=policy, quarantine_sink=sink,
                           store_ctx=store_ctx)


def __test_report_external_intervention_quarantines_under_stop__():
    latched: list = []
    plugin = _plugin(
        'stop', lambda reason, context, *, kind: latched.append((reason, context, kind)))
    BrokerPlugin.report_external_intervention(  # type: ignore[arg-type]
        plugin, "SL moved", {'protection_key': 'p1'})
    assert len(latched) == 1
    reason, context, kind = latched[0]
    assert reason == "SL moved"
    assert context == {'protection_key': 'p1', 'policy': 'stop'}
    assert kind is QuarantineKind.EXTERNAL_INTERVENTION


def __test_report_external_intervention_halts_without_a_sink_or_under_halt__():
    with pytest.raises(UnexpectedCancelError):
        BrokerPlugin.report_external_intervention(  # type: ignore[arg-type]
            _plugin('stop'), "SL moved")
    with pytest.raises(UnexpectedCancelError):
        BrokerPlugin.report_external_intervention(  # type: ignore[arg-type]
            _plugin('halt', lambda *a, **k: None), "SL moved")


def __test_report_external_intervention_only_audits_under_ignore__(tmp_path: Path):
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = _open_run(store)
        latched: list = []
        plugin = _plugin('ignore', lambda *a, **k: latched.append(a), ctx)
        BrokerPlugin.report_external_intervention(  # type: ignore[arg-type]
            plugin, "SL moved", client_order_id="leg-1")
        assert latched == []
        kinds = [r['kind'] for r in ctx._store._conn.execute(  # type: ignore[attr-defined]
            "SELECT kind FROM events WHERE run_instance_id = ?",
            (ctx.run_instance_id,)).fetchall()]
        assert 'external_intervention_ignored' in kinds
