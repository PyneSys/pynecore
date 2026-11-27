"""
The quarantine records a supervisor parses from the bot's log.

A supervisor (the live lab, an operator's alerting) reads the bot's stdout
line by line and must find, on one physical line each: the latch token with
the kind value and the quarantine id, and what ends the latch: the exact
resume command of an external intervention, or the open causes of an entry
block, which the bot ends by itself (no command is printed for it). The log
sink renders through :class:`~pynecore.lib.log.PineRichHandler` at the durable
width, which folds every long record onto continuation lines; the quarantine
records must survive that.
"""
import asyncio
import io
import logging
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.broker.disappearance import resolve_unexpected_cancel_policy
from pynecore.core.broker.models import (
    FENCE_CAUSE_PREFIX,
    ExchangeCapabilities,
    QuarantineKind,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.core.plugin.broker import BrokerPlugin
from pynecore.lib import log as log_module
from pynecore.lib.log import DURABLE_LOG_WIDTH

rich = pytest.importorskip("rich")
from rich.console import Console  # noqa: E402

PLUGIN = "Mock"
SYMBOL = "ETHUSDT"

#: What a line-by-line supervisor greps
_LATCH = re.compile(
    r"sync engine quarantined(?: \((?P<status>[^)]*)\))?: "
    r"kind=(?P<kind>[a-z_]+) quarantine_id=(?P<qid>[0-9a-f]+)"
)
_COMMAND = re.compile(
    r"sync engine quarantine (?P<qid>[0-9a-f]+) resume command: "
    r"(?P<command>pyne broker resume '(?P<run_id>[^']+)' (?P=qid))$"
)
_SELF_ENDING = re.compile(
    r"sync engine quarantine (?P<qid>[0-9a-f]+) ends by itself: open causes: "
    r"(?P<causes>\S+(?:, \S+)*) — "
)


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


@pytest.fixture
def durable_sink(monkeypatch: pytest.MonkeyPatch):
    """Render the pyne logger into a non-TTY console at the durable width.

    The same handler the CLI installs on a redirected stdout (the test session
    turns colored logging off, so it is attached here for the test).
    """
    buffer = io.StringIO()
    handler = log_module.PineRichHandler(
        console=Console(file=buffer, width=DURABLE_LOG_WIDTH, force_terminal=False,
                        color_system=None),
        show_time=True, show_level=True, omit_repeated_times=False, markup=False,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%Y-%m-%d %H:%M:%S%z]"))
    monkeypatch.setattr(lib, 'bar_index', 1234)
    log_module.logger.addHandler(handler)
    try:
        yield buffer
    finally:
        log_module.logger.removeHandler(handler)


class _Broker:
    on_unexpected_cancel = "stop"
    client_order_id_max_len = 30

    @staticmethod
    def get_capabilities() -> ExchangeCapabilities:
        return ExchangeCapabilities()


def _open_ctx(store: BrokerStore) -> RunContext:
    # A run id about as long as the live lab's labelled runs (~130 characters)
    identity = RunIdentity(
        strategy_id="live_lab_trend_bracket_bot_with_a_long_descriptive_strategy_id",
        symbol=SYMBOL, timeframe="1", account_id="bybit-demo-123456789",
        label="lane-bybit-cycle-rotation-trend",
    )
    return store.open_run(identity, script_source="// t162")


def _engine(ctx: RunContext | None) -> OrderSyncEngine:
    return OrderSyncEngine(
        broker=_Broker(),  # type: ignore[arg-type]
        position=BrokerPosition(),
        symbol=SYMBOL,
        run_tag=ctx.run_tag if ctx is not None else "test",
        mintick=0.01,
        store_ctx=ctx,
    )


def _lines(buffer: io.StringIO) -> list[str]:
    return [line.rstrip() for line in buffer.getvalue().splitlines()]


def _latches(buffer: io.StringIO) -> list[re.Match]:
    return [m for line in _lines(buffer) if (m := _LATCH.search(line))]


def _commands(buffer: io.StringIO) -> list[re.Match]:
    return [m for line in _lines(buffer) if (m := _COMMAND.search(line))]


def _self_ending(buffer: io.StringIO) -> list[re.Match]:
    return [m for line in _lines(buffer) if (m := _SELF_ENDING.search(line))]


def __test_a_latch_logs_its_kind_id_and_command_on_one_line_each__(
        tmp_path: Path, durable_sink: io.StringIO,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        assert len(ctx.run_id) > 120
        engine = _engine(ctx)
        engine.record_quarantine(
            "bot protection 'p1' changed on the venue by someone else:\n"
            "expected stop 95.0, found stop 90.0 " + "x" * 120,
            {'protection_key': 'p1'}, kind=QuarantineKind.EXTERNAL_INTERVENTION,
        )

        latches = _latches(durable_sink)
        assert len(latches) == 1
        assert latches[0]['kind'] == 'external_intervention'
        assert latches[0]['qid'] == engine.quarantine_id
        assert latches[0]['status'] is None
        commands = _commands(durable_sink)
        assert len(commands) == 1
        assert commands[0]['qid'] == engine.quarantine_id
        assert commands[0]['run_id'] == ctx.run_id
        assert commands[0]['command'] == (
            f"pyne broker resume '{ctx.run_id}' {engine.quarantine_id}")
        # The reason's line break never splits the record
        latch_line = next(line for line in _lines(durable_sink) if _LATCH.search(line))
        assert "expected stop 95.0, found stop 90.0" in latch_line
        assert len(_lines(durable_sink)) == 2
        ctx.close()


def __test_a_quote_in_the_run_id_keeps_the_resume_command_valid_shell__(
        tmp_path: Path, durable_sink: io.StringIO,
) -> None:
    """A run label (or a script name) may carry a quote: the command must still run."""
    import shlex

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = store.open_run(
            RunIdentity(strategy_id="bob's_trend", symbol=SYMBOL, timeframe="1",
                        account_id="bybit-demo-123456789", label="it's-lane-2"),
            script_source="// t162 quote",
        )
        engine = _engine(ctx)
        engine.record_quarantine("operator moved the stop",
                                 kind=QuarantineKind.EXTERNAL_INTERVENTION)

        command_lines = [line for line in _lines(durable_sink)
                         if "resume command: pyne broker resume '" in line]
        assert len(command_lines) == 1
        command = command_lines[0].split("resume command: ", 1)[1].strip()
        assert shlex.split(command) == [
            "pyne", "broker", "resume", ctx.run_id, engine.quarantine_id]
        assert len(_lines(durable_sink)) == 2
        ctx.close()


def __test_an_escalation_logs_the_same_stable_record__(
        tmp_path: Path, durable_sink: io.StringIO,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine = _engine(ctx)
        engine.record_quarantine("protective_exit_rejected", cause="protective_exit_rejected:L")
        engine.record_quarantine("operator moved the stop",
                                 kind=QuarantineKind.EXTERNAL_INTERVENTION)

        latches = _latches(durable_sink)
        assert [m['kind'] for m in latches] == ['entry_block', 'external_intervention']
        assert latches[1]['status'] == 'escalated from entry_block'
        assert latches[0]['qid'] == latches[1]['qid'] == engine.quarantine_id
        assert [m['qid'] for m in _commands(durable_sink)] == [engine.quarantine_id]
        assert [m['causes'] for m in _self_ending(durable_sink)] == [
            'protective_exit_rejected:L']
        ctx.close()


def __test_an_entry_block_names_its_causes_and_no_resume_command__(
        tmp_path: Path, durable_sink: io.StringIO,
) -> None:
    """The bot's own trouble asks nobody to act: its record names what the bot ends by itself."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine = _engine(ctx)
        engine.record_quarantine("protective_exit_rejected", cause="protective_exit_rejected:L")
        engine.record_quarantine("close_rejected_repeatedly", cause="close_rejected:S")
        engine._quarantine_reminded_at = 0.0  # noqa: SLF001
        engine._log_quarantine_record(" (reminder, entered x)", "still held",  # noqa: SLF001
                                      error=False)

        assert [m['kind'] for m in _latches(durable_sink)] == ['entry_block', 'entry_block']
        assert [m['causes'] for m in _self_ending(durable_sink)] == [
            'protective_exit_rejected:L', 'close_rejected:S, protective_exit_rejected:L']
        assert _commands(durable_sink) == []
        assert not any("pyne broker resume" in line for line in _lines(durable_sink))
        ctx.close()


def __test_a_fenced_process_is_named_as_fenced__(
        tmp_path: Path, durable_sink: io.StringIO,
) -> None:
    """A fence never ends in its process: the record says so instead of a self-ending block."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine = _engine(ctx)
        engine.record_quarantine("spot_lease_lost", cause=f"{FENCE_CAUSE_PREFIX}spot:BTC")

        fenced = [line for line in _lines(durable_sink)
                  if f"quarantine {engine.quarantine_id} fenced: fence:spot:BTC — " in line]
        assert len(fenced) == 1
        assert _self_ending(durable_sink) == [] and _commands(durable_sink) == []
        ctx.close()


def __test_a_restored_latch_is_logged_like_a_fresh_one__(
        tmp_path: Path, durable_sink: io.StringIO,
) -> None:
    """The next cycle of an unresumed intervention must be flagged by every consumer."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        first = _open_ctx(store)
        engine = _engine(first)
        engine.record_quarantine("operator closed the position",
                                 kind=QuarantineKind.EXTERNAL_INTERVENTION)
        quarantine_id = engine.quarantine_id
        first.close()
        durable_sink.truncate(0)
        durable_sink.seek(0)

        restarted = _open_ctx(store)
        engine = _engine(restarted)

        assert engine.venue_writes_blocked
        latches = _latches(durable_sink)
        assert len(latches) == 1
        assert latches[0]['kind'] == 'external_intervention'
        assert latches[0]['qid'] == quarantine_id
        assert re.fullmatch(r"restored at startup, entered \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ",
                            latches[0]['status'])
        assert [m['qid'] for m in _commands(durable_sink)] == [quarantine_id]
        restarted.close()


def __test_the_resume_watch_reminds_while_the_latch_holds__(
        tmp_path: Path, durable_sink: io.StringIO,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine = _engine(ctx)
        engine.record_quarantine("operator moved the stop",
                                 kind=QuarantineKind.EXTERNAL_INTERVENTION)

        async def _watch_briefly() -> None:
            task = asyncio.create_task(engine.run_quarantine_resume_watch(
                interval_s=0.0, reminder_interval_s=0.0))
            for _ in range(5):
                await asyncio.sleep(0)
            task.cancel()

        asyncio.run(_watch_briefly())
        statuses = [m['status'] for m in _latches(durable_sink)]
        assert statuses[0] is None
        assert statuses[1:], "the watch repeats the record while the latch holds"
        assert all(s is not None and s.startswith('reminder, entered ') for s in statuses[1:])
        ctx.close()


def _audit_kinds(ctx: RunContext) -> list[str]:
    return [r['kind'] for r in ctx._store._conn.execute(  # type: ignore[attr-defined]  # noqa: SLF001
        "SELECT kind FROM events ORDER BY id").fetchall()]


def __test_a_further_intervention_under_the_latch_is_logged_once_across_restarts__(
        tmp_path: Path, durable_sink: io.StringIO,
) -> None:
    def _report(engine: OrderSyncEngine, ctx: RunContext, key: str) -> None:
        resolve_unexpected_cancel_policy(
            'stop', reason=f"bot protection {key!r} changed on the venue",
            context={'protection_key': key}, request_quarantine=engine.record_quarantine,
            log_event=ctx.log_event, client_order_id=None, exchange_order_id=None,
            audit_prefix='external_intervention',
        )

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        first = _open_ctx(store)
        engine = _engine(first)
        _report(engine, first, 'p1')
        quarantine_id = engine.quarantine_id
        _report(engine, first, 'p2')
        _report(engine, first, 'p2')
        further = [line for line in _lines(durable_sink) if "further intervention" in line]
        assert len(further) == 1
        assert f"sync engine quarantine {quarantine_id} further intervention: " \
               f"kind=external_intervention" in further[0]
        assert "'p2'" in further[0]
        assert _audit_kinds(first).count('external_intervention_quarantine') == 2
        assert _audit_kinds(first).count('quarantine_further_intervention') == 1
        first.close()

        # A restarted plugin re-reports both edits: the restored latch knows them.
        restarted = _open_ctx(store)
        engine = _engine(restarted)
        durable_sink.truncate(0)
        durable_sink.seek(0)
        _report(engine, restarted, 'p1')
        _report(engine, restarted, 'p2')
        assert not [line for line in _lines(durable_sink) if "further intervention" in line]
        kinds = _audit_kinds(restarted)
        assert kinds.count('external_intervention_quarantine') == 2
        assert kinds.count('quarantine_further_intervention') == 1
        assert engine.quarantine_id == quarantine_id
        restarted.close()


def __test_a_failed_audit_of_a_further_intervention_never_stops_the_process__(
        tmp_path: Path, durable_sink: io.StringIO, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The held latch must not depend on the disk: a raise would read as a failed quarantine."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine = _engine(ctx)
        assert resolve_unexpected_cancel_policy(
            'stop', reason="bot protection 'p1' changed on the venue",
            context={'protection_key': 'p1'}, request_quarantine=engine.record_quarantine,
            log_event=ctx.log_event, client_order_id=None, exchange_order_id=None,
            audit_prefix='external_intervention',
        ) is None
        quarantine_id = engine.quarantine_id

        def _disk_full(kind: str, **_kwargs) -> None:
            if kind == 'quarantine_further_intervention':
                raise OSError("disk full")

        monkeypatch.setattr(ctx, 'log_event', _disk_full)
        halt = resolve_unexpected_cancel_policy(
            'stop', reason="bot protection 'p2' changed on the venue",
            context={'protection_key': 'p2'}, request_quarantine=engine.record_quarantine,
            log_event=None, client_order_id=None, exchange_order_id=None,
            audit_prefix='external_intervention',
        )

        assert halt is None, "no process-exiting halt"
        assert engine.venue_writes_blocked and engine.quarantine_id == quarantine_id
        assert any("could not be audited" in line for line in _lines(durable_sink))
        ctx.close()


def __test_a_durable_intervention_request_answers_whether_the_latch_survives_a_restart__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plugin-facing request reports ``False`` until the store holds the latch.

    The latch's store write fails: the engine holds the latch in memory and
    owes the write, and the request answers ``False``, so a plugin keeps the
    state only the latch guards. Once the owed write lands the same request
    answers ``True``. A latch whose resume the operator requested blocks
    nothing across a restart and answers ``False`` again.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine = _engine(ctx)
        plugin = SimpleNamespace(quarantine_sink=engine.record_quarantine, store_ctx=ctx)
        original = ctx.record_quarantine

        def _locked(*_args, **_kwargs) -> None:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(ctx, 'record_quarantine', _locked)
        assert BrokerPlugin.request_durable_intervention_quarantine(
            plugin, "bot position closed by someone else", {'deal_id': 'd1'}) is False  # type: ignore[arg-type]
        assert engine.venue_writes_blocked
        assert ctx.intervention_latch_stored() is False

        monkeypatch.setattr(ctx, 'record_quarantine', original)
        engine._persist_quarantine()
        assert BrokerPlugin.request_durable_intervention_quarantine(
            plugin, "bot position closed by someone else", {'deal_id': 'd1'}) is True  # type: ignore[arg-type]

        quarantine_id = engine.quarantine_id
        assert quarantine_id is not None
        store.request_quarantine_resume(ctx.run_id, quarantine_id)
        assert ctx.intervention_latch_stored() is False
        ctx.close()


def __test_a_durable_intervention_request_without_a_sink_latches_nothing__() -> None:
    """Without a wired sink nothing latches, so nothing may be retired on it."""
    plugin = SimpleNamespace(quarantine_sink=None, store_ctx=None)
    assert BrokerPlugin.request_durable_intervention_quarantine(
        plugin, "manual close", None) is False  # type: ignore[arg-type]
