from datetime import datetime, UTC

from typer import Typer, Argument, Exit, secho, colors

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
):
    """
    Resume a quarantined live run.

    The running bot picks the request up within seconds, stops, and restarts
    itself in place: it re-syncs with the venue exactly like a fresh start and
    restores its own orders. When the bot is not running, the next start
    resumes. Run it on the node where the bot runs, with the same workdir.
    """
    store_path = app_state.workdir / "output" / "logs" / "broker.sqlite"
    if not store_path.exists():
        secho(f"No broker store at {store_path}", err=True, fg=colors.RED)
        raise Exit(1)
    store = BrokerStore(store_path, plugin_name="pyne-cli")
    try:
        row = store.request_quarantine_resume(run_id, quarantine_id)
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
    secho("A running bot restarts itself within seconds; a stopped bot resumes at its next start.")
