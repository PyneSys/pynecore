from datetime import datetime, UTC
from typing import Annotated

from typer import Typer, Argument, Option, Exit, secho, colors

from ..app import app, app_state
from pynecore.core.broker.storage import BrokerStore, QuarantineRow

__all__ = []

app_broker = Typer(help="Live broker run management")
app.add_typer(app_broker, name="broker")


def _format_quarantine(row: QuarantineRow) -> str:
    entered = datetime.fromtimestamp(row.entered_ts_ms / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"  '{row.run_id}' {row.quarantine_id}  ({row.kind}, since {entered}): {row.reason}"


@app_broker.command()
def resume(
        run_id: str = Argument(..., help="The run id, as printed in the quarantine log line"),
        quarantine_id: str = Argument(..., help="The quarantine id, as printed in the quarantine log line"),
        close_executed: Annotated[list[str] | None, Option(
            "--close-executed", metavar="CLOSE_COID",
            help="A held partial-bracket close the venue shows executed (repeatable)",
        )] = None,
        close_not_executed: Annotated[list[str] | None, Option(
            "--close-not-executed", metavar="CLOSE_COID",
            help="A held partial-bracket close the venue shows did not execute (repeatable)",
        )] = None,
        entry_not_executed: Annotated[list[str] | None, Option(
            "--entry-not-executed", metavar="ENTRY_ID",
            help="An entry held for its execution evidence whose order did not execute (repeatable)",
        )] = None,
):
    """
    Resume a quarantined live run.

    The running bot picks the request up within seconds, stops, and restarts
    itself in place: it re-syncs with the venue exactly like a fresh start and
    restores its own orders. When the bot is not running, the next start
    resumes. Run it on the node where the bot runs, with the same workdir.

    A partial-bracket close no venue record settles keeps its leg held, and
    the bot quarantines again after the restart. Check the close on the venue
    and pass its client order id (printed in the quarantine ERROR line) with
    --close-executed or --close-not-executed: the restarted bot settles the
    leg with that decision.

    An entry whose order ended at the venue without a reported executed
    quantity stays held, and the bot quarantines again after the restart,
    until the venue's quantity arrives. Check the order on the venue, close
    anything it executed there (the bot does not own it), and pass the entry
    id (printed in the quarantine WARNING line) with --entry-not-executed:
    the restarted bot releases the entry, and the script may send it again.
    An entry the venue already reported executed is refused: it stays held
    until the plugin delivers its fill.
    """
    decisions: dict[str, bool] = {coid: True for coid in close_executed or ()}
    for coid in close_not_executed or ():
        if coid in decisions:
            secho(f"Close {coid} is given both as executed and as not executed.",
                  err=True, fg=colors.RED)
            raise Exit(1)
        decisions[coid] = False
    store_path = app_state.workdir / "output" / "logs" / "broker.sqlite"
    if not store_path.exists():
        secho(f"No broker store at {store_path}", err=True, fg=colors.RED)
        raise Exit(1)
    store = BrokerStore(store_path, plugin_name="pyne-cli")
    try:
        entries = sorted(set(entry_not_executed or ()))
        if entries:
            held_entries = store.held_fill_evidence_entries(run_id)
            unknown_entries = [key for key in entries if key not in held_entries]
            if unknown_entries:
                secho(f"No entry held for its execution evidence {', '.join(unknown_entries)} "
                      f"in run '{run_id}'.", err=True, fg=colors.RED)
                if held_entries:
                    secho("Held entries:", err=True)
                    for key in sorted(held_entries):
                        secho(f"  {key}", err=True)
                else:
                    secho("The run holds no entry for its execution evidence.", err=True)
                raise Exit(1)
            reported = store.fill_evidence_entries_reported_executed(run_id)
            executed = [key for key in entries if key in reported]
            if executed:
                for key in executed:
                    secho(f"Entry {key} did execute: the venue reported {reported[key]} "
                          f"executed. It stays held until the plugin delivers that fill; "
                          f"check the plugin's fill reporting.", err=True, fg=colors.RED)
                raise Exit(1)
        if decisions:
            held = store.held_trigger_close_coids(run_id)
            unknown = sorted(coid for coid in decisions if coid not in held)
            if unknown:
                secho(f"No held partial-bracket close {', '.join(unknown)} in run '{run_id}'.",
                      err=True, fg=colors.RED)
                if held:
                    secho("Held closes:", err=True)
                    for coid in sorted(held):
                        secho(f"  {coid}", err=True)
                else:
                    secho("The run holds no partial-bracket close.", err=True)
                raise Exit(1)
        row = store.request_quarantine_resume(run_id, quarantine_id, decisions, entries)
        if row is None:
            secho(f"No open quarantine {quarantine_id} for run '{run_id}'.", err=True, fg=colors.RED)
            open_rows = store.open_quarantines()
            if open_rows:
                secho("Open quarantines:", err=True)
                for open_row in open_rows:
                    secho(_format_quarantine(open_row), err=True)
            else:
                secho("There is no open quarantine.", err=True)
            raise Exit(1)
    finally:
        store.close()
    secho(f"Resume requested for quarantine {row.quarantine_id} of run '{row.run_id}'.", fg=colors.GREEN)
    for coid, executed in sorted(decisions.items()):
        secho(f"Recorded: close {coid} {'executed' if executed else 'did not execute'}.")
    for key in entries:
        secho(f"Recorded: entry {key} did not execute.")
    secho("A running bot restarts itself within seconds; a stopped bot resumes at its next start.")
