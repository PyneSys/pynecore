import json
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, cast

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from typer import Argument, Exit, Option, secho

from ..app import app, app_state
from .run import (backtest_security_path, check_data_period, resolve_chart_timeframe,
                  resolve_data, resolve_script, resolve_time_window, run_backtest_with_progress)
# noinspection PyProtectedMember
from .run import _pin_timenow
from ...analysis import (RobustnessConfig, RobustnessReport, TradeRecord, TradeRecorder, analyze,
                         load_trade_csv)
from ...analysis.bootstrap import Block
from ...analysis.coinflip import DEFAULT_RUNS, Mode, coinflip_test
from ...analysis.costs import DEFAULT_COST_BPS
from ...analysis.montecarlo import Method
from ...analysis.periods import Period
from ...analysis.trades import Unit
from ...core.ohlcv import ChartBarWindow, OHLCVReader
from ...core.script_runner import ScriptRunner
from ...core.syminfo import SymInfo

__all__ = []

console = Console()

_UNITS = ('auto', 'r', 'currency', 'percent')
_METHODS = ('bootstrap', 'shuffle')
_BLOCKS = ('week', 'month', 'trade')
_PERIODS = ('year', 'month', 'weekday')
_COINFLIPS = ('direction', 'timing', 'both', 'none')


def _parse_list(value: str, name: str) -> list[str]:
    items = [item.strip() for item in value.split(',') if item.strip()]
    if not items:
        secho(f"{name} needs at least one value", fg="red", err=True)
        raise Exit(1)
    return items


def _parse_bps(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in _parse_list(value, '--cost-bps'))
    except ValueError:
        secho(f"Invalid --cost-bps: '{value}' (expected numbers like 0,2,5,10)",
              fg="red", err=True)
        raise Exit(1)


def _choice(value: str, choices: tuple[str, ...], name: str) -> str:
    if value not in choices:
        secho(f"Invalid {name}: '{value}' (choose from {', '.join(choices)})",
              fg="red", err=True)
        raise Exit(1)
    return value


@contextmanager
def _scripts_lib_path() -> Iterator[None]:
    """Make the workdir's script libraries importable while the script loads."""
    lib_dir = app_state.scripts_dir / "lib"
    added = lib_dir.is_dir()
    if added:
        sys.path.insert(0, str(lib_dir))
    try:
        yield
    finally:
        if added:
            sys.path.remove(str(lib_dir))


@dataclass(slots=True)
class _Session:
    """
    One backtest set up for repeated runs.

    :ivar syminfo: Symbol information of the chart
    :ivar make_runner: Builds a fresh runner on the same data
    :ivar time_from: Start of the window, naive UTC
    :ivar time_to: End of the window, naive UTC
    """

    syminfo: SymInfo
    make_runner: Callable[[], ScriptRunner]
    time_from: datetime
    time_to: datetime

    @property
    def total_seconds(self) -> int:
        """:return: Length of the window"""
        return int((self.time_to - self.time_from).total_seconds())


@contextmanager
def _backtest_session(script: Path, data: str, time_from: str | None, time_to: str | None,
                      timeframe: str | None, security: list[str] | None) -> Iterator[_Session]:
    """
    Open the data of a backtest the way ``pyne run`` does, for one or more runs.

    :return: The session, valid while the context is open
    """
    data_path, syminfo, provider_data = resolve_data(data, time_from)
    data_file_period = syminfo.period
    magnifier_source_tf = resolve_chart_timeframe(syminfo, timeframe)

    security_data: dict[str, Any] = {}
    if security:
        for entry in security:
            if '=' not in entry:
                secho(f"Invalid --security format: '{entry}'. "
                      f"Expected 'TIMEFRAME=value' or 'SYMBOL:TIMEFRAME=value'",
                      fg="red", err=True)
                raise Exit(1)
            key, value = entry.split('=', 1)
            security_data[key] = backtest_security_path(value)

    with OHLCVReader(data_path) as reader:
        check_data_period(reader, data_file_period, data_path)
        time_from_dt, time_to_dt = resolve_time_window(
            reader, None if provider_data is not None else time_from, time_to, provider_data,
        )
        time_from_ts = int(time_from_dt.timestamp() * 1000)
        time_to_ts = int(time_to_dt.timestamp() * 1000)
        size = reader.get_size(time_from_ts, time_to_ts)
        last_bar_time = reader.get_last_bar_time(time_from_ts, time_to_ts)
        _pin_timenow(last_bar_time)

        def make_runner() -> ScriptRunner:
            chart_bar_window = ChartBarWindow(reader.path, time_from_ts, time_to_ts)
            if magnifier_source_tf is not None:
                magnifier_iter = reader.read_from(time_from_ts, time_to_ts)
                ohlcv_iter = iter([])
            else:
                magnifier_iter = None
                ohlcv_iter = chart_bar_window.bars()
            with _scripts_lib_path():
                return ScriptRunner(script, ohlcv_iter, syminfo, last_bar_index=size - 1,
                                    last_bar_time=last_bar_time,
                                    plot_path=None, strat_path=None, trade_path=None,
                                    security_data=security_data or None,
                                    magnifier_iter=magnifier_iter,
                                    magnifier_source_tf=magnifier_source_tf,
                                    chart_bar_window=chart_bar_window,
                                    time_from=time_from_dt,
                                    chart_data_path=data_path,
                                    lossless_volume=reader.lossless_volume,
                                    lossless_prices=reader.lossless_prices,
                                    config_dir=app_state.config_dir)

        yield _Session(syminfo=syminfo, make_runner=make_runner,
                       time_from=time_from_dt.astimezone(UTC).replace(tzinfo=None),
                       time_to=time_to_dt.astimezone(UTC).replace(tzinfo=None))


def _record_backtest(session: _Session, script: Path) -> list[TradeRecord]:
    """
    Run the backtest once and record its closed trades with their initial stops.

    :return: The closed trades in trade order
    """
    with Progress(SpinnerColumn(finished_text="[green]✓"), TextColumn("{task.description}"),
                  console=console) as loading:
        task = loading.add_task("Loading PyneCore...", total=1)
        runner = session.make_runner()
        loading.update(task, completed=1)

    position = getattr(runner.script, 'position', None)
    if position is None:
        secho(f"'{script.name}' is not a strategy: robustness tests need its trades.",
              fg="red", err=True)
        raise Exit(1)

    recorder = TradeRecorder(position, session.syminfo)
    run_backtest_with_progress(runner, session.time_from, session.time_to, session.total_seconds,
                               on_bar=recorder.on_bar)
    return recorder.trades


def _run_coinflips(session: _Session, report: RobustnessReport, trades: list[TradeRecord],
                   modes: tuple[Mode, ...], seed: int, draws: int, runs: int,
                   exact: bool) -> None:
    """Run the coin-flip tests and add them to the report."""
    for mode in modes:
        backtests = 1 if mode == 'direction' and not exact else runs
        label = f"Coin flip ({mode})..."
        with Progress(SpinnerColumn(finished_text="[green]✓"), TextColumn("{task.description}"),
                      BarColumn(), MofNCompleteColumn(), TextColumn("backtests"),
                      console=console) as progress:
            task = progress.add_task(label, total=backtests)

            def on_progress(done: int, total: int) -> None:
                progress.update(task, completed=done, total=total)

            result = coinflip_test(mode, session.make_runner, session.syminfo, trades,
                                   report.unit, seed=seed, draws=draws, exact=exact, runs=runs,
                                   on_progress=on_progress)
        report.coinflips.append(result)


def _number(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        if value != value:
            return ''
        if abs(value) >= 100.0:
            return f"{value:,.2f}"
        return f"{value:.4g}"
    return str(value)


def _print_report(report: RobustnessReport) -> None:
    table = Table(title=f"Robustness of {report.trades} trades", title_justify="left")
    table.add_column('Test', style='bold')
    table.add_column('Metric')
    for column in ('Value', 'Low / P5', 'High / P95'):
        table.add_column(column, justify='right')
    table.add_column('Verdict')
    last_test = None
    for row in report.summary_rows():
        test = row['Test']
        if last_test is not None and test != last_test:
            table.add_section()
        metric = f"{row['Metric']} ({row['Unit']})" if row['Unit'] else row['Metric']
        table.add_row(test if test != last_test else '', metric, _number(row['Value']),
                      _number(row['Low / P5']), _number(row['High / P95']), row['Verdict'])
        last_test = test
    console.print(table)
    for flip in report.coinflips:
        for note in flip.notes:
            console.print(f"[yellow]Coin flip ({flip.mode}): {note}.[/]")
    if report.skipped:
        console.print(f"[yellow]{report.skipped} of {report.trades} trades have no value in "
                      f"the unit '{report.unit}' and were left out.[/]")
    if not report.costs.rows:
        console.print(f"[yellow]No trade has a known cost in the unit '{report.unit}' (a trade "
                      f"CSV carries no point value): the cost test was skipped.[/]")


def _finish(report: RobustnessReport, output_dir: Path | None, stem: str,
            write_json: bool) -> None:
    """Print the report and write its files."""
    _print_report(report)
    directory = output_dir if output_dir is not None else app_state.output_dir
    written = report.write_csv(directory, stem)
    if write_json:
        json_path = directory / f"{stem}_robustness.json"
        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding='utf-8')
        written.append(json_path)
    for path in written:
        console.print(f"[dim]Written:[/] {path}")


@app.command(name="robustness")
def robustness(
        script: Path = Argument(..., dir_okay=False, file_okay=True,
                                help="Strategy to test (.py or .pine)"),
        data: str | None = Argument(None,
                                    help="Data file (*.ohlcv, *.csv) or provider string "
                                         "(e.g. ccxt:BYBIT:BTC/USDT:USDT@1D). "
                                         "Not needed with --from-trades."),
        time_from: str | None = Option(None, '--from', '-f', metavar="[DATE|DAYS|-BARS]",
                                       help="Start: date in UTC (2025-01-01), days back from now "
                                            "(30), or -N bars back (-500)"),
        time_to: str | None = Option(None, '--to', '-t', metavar="[DATE|DAYS]",
                                     help="End: date in UTC or days back from now"),
        timeframe: str | None = Option(None, "--timeframe", "-tf",
                                       help="Chart timeframe (TradingView format, e.g. '60', '1D')"),
        security: list[str] | None = Option(None, "--security", "-sec",
                                            help='Security data: "TIMEFRAME=data_name" or '
                                                 '"SYMBOL:TIMEFRAME=data_name"'),
        api_key: str | None = Option(None, "--api-key", "-a",
                                     help="PyneSys API key for compilation",
                                     envvar="PYNESYS_API_KEY"),
        from_trades: Path | None = Option(None, "--from-trades",
                                          help="Analyse a trade CSV written by 'pyne run' "
                                               "instead of running the backtest. It has no "
                                               "stops, so the results are in currency, not R, "
                                               "and no point value, so the cost test is skipped.",
                                          rich_help_panel="Input Options"),
        unit: str = Option('auto', "--unit",
                           help="Per-trade unit: auto (R when every trade has a stop), r, "
                                "currency or percent",
                           rich_help_panel="Test Options"),
        draws: int = Option(10_000, "--draws", min=100,
                            help="Resamples of every random test",
                            rich_help_panel="Test Options"),
        seed: int = Option(0, "--seed", help="Seed of every random test",
                           rich_help_panel="Test Options"),
        mc_method: str = Option('bootstrap', "--mc-method",
                                help="Monte Carlo: bootstrap (draw trades with replacement) or "
                                     "shuffle (reorder the trades)",
                                rich_help_panel="Test Options"),
        block: str = Option('week', "--block",
                            help="Bootstrap block: week, month or trade",
                            rich_help_panel="Test Options"),
        confidence: float = Option(95.0, "--confidence", min=50.0, max=99.9,
                                   help="Coverage of the intervals, percent",
                                   rich_help_panel="Test Options"),
        cost_bps: str = Option(','.join(f"{b:g}" for b in DEFAULT_COST_BPS), "--cost-bps",
                               help="Extra cost scenarios per side, basis points",
                               rich_help_panel="Test Options"),
        periods: str = Option('year', "--periods",
                              help="Calendar breakdowns: any of year, month, weekday",
                              rich_help_panel="Test Options"),
        capital: float | None = Option(None, "--capital", min=0.0,
                                       help="Starting account of the account Monte Carlo "
                                            "(default: the equity of the first trade)",
                                       rich_help_panel="Account Options"),
        risk: float = Option(1.0, "--risk", min=0.0, max=100.0,
                             help="Account risked per trade when R is compounded, percent",
                             rich_help_panel="Account Options"),
        ruin: float = Option(50.0, "--ruin", min=0.0, max=100.0,
                             help="Loss of the starting account that counts as ruin, percent",
                             rich_help_panel="Account Options"),
        max_leverage: float | None = Option(None, "--max-leverage", min=0.0,
                                            help="Largest position of the account Monte Carlo "
                                                 "as a multiple of the account (1 = no "
                                                 "leverage); a closer stop then risks less",
                                            rich_help_panel="Account Options"),
        target: float | None = Option(None, "--target", min=0.0,
                                      help="Multiple of the starting account counted as "
                                           "reaching the target (100 = $100 to $10,000)",
                                      rich_help_panel="Account Options"),
        coinflip: str = Option('direction', "--coinflip",
                               help="Coin-flip test: direction (a coin decides long or short on "
                                    "every real entry), timing (the real trades enter at random "
                                    "moments), both, or none",
                               rich_help_panel="Coin Flip Options"),
        coinflip_runs: int = Option(DEFAULT_RUNS, "--coinflip-runs", min=10,
                                    help="Backtests of the tests that run one per draw: timing, "
                                         "and direction with --exact",
                                    rich_help_panel="Coin Flip Options"),
        exact: bool = Option(False, "--exact",
                             help="Direction test: a full backtest per draw instead of pairing "
                                  "the trades of one flipped backtest",
                             rich_help_panel="Coin Flip Options"),
        output_dir: Path | None = Option(None, "--output", "-o", file_okay=False,
                                         help="Directory of the CSV files "
                                              "(default: workdir/output)",
                                         rich_help_panel="Out Path Options"),
        write_json: bool = Option(False, "--json",
                                  help="Also write the whole result as <script>_robustness.json",
                                  rich_help_panel="Out Path Options"),
):
    """
    Test how much of a strategy's backtest result could be luck

    Runs the backtest once, records its closed trades, and tests them:
    \b
      Monte Carlo     the trades resampled: spread of the final result and the drawdown
      Account         the same with a compounding account: ruin and target chances
      Bootstrap       interval of the mean trade, resampling calendar blocks
      Hit rate        Wilson interval of the hit rate
      Costs           the result under extra trading costs, and the break-even cost
      Periods         the result by year, month or weekday
      Coin flip       the real result against the same trades with a random direction
                      (default) or a random entry moment (--coinflip timing)

    A trade with a stop-loss is measured in R (profit / initial risk); without stops the
    trades are measured in account currency. Every random test is seeded: the same trades
    and seed give the same numbers.

    The results are written as CSV files next to the other outputs:
    [italic]<script>_robustness.csv[/], [italic]_montecarlo.csv[/], [italic]_costs.csv[/],
    [italic]_periods.csv[/], [italic]_coinflip_<mode>.csv[/].
    """  # noqa
    unit_choice = _choice(unit, _UNITS, '--unit')
    coinflip_choice = _choice(coinflip, _COINFLIPS, '--coinflip')
    modes: tuple[Mode, ...] = (('direction', 'timing') if coinflip_choice == 'both'
                               else () if coinflip_choice == 'none'
                               else (cast(Mode, coinflip_choice),))
    config = RobustnessConfig(
        unit=None if unit_choice == 'auto' else cast(Unit, unit_choice),
        seed=seed,
        draws=draws,
        mc_method=cast(Method, _choice(mc_method, _METHODS, '--mc-method')),
        block=cast(Block, _choice(block, _BLOCKS, '--block')),
        confidence=confidence / 100.0,
        cost_bps=_parse_bps(cost_bps),
        periods=tuple(cast(Period, _choice(p, _PERIODS, '--periods'))
                      for p in _parse_list(periods, '--periods')),
        initial_capital=capital,
        risk_fraction=risk / 100.0,
        ruin_fraction=ruin / 100.0,
        target_multiple=target,
        max_leverage=max_leverage or None,
    )

    def _analyze(closed: list[TradeRecord]) -> RobustnessReport:
        if not closed:
            secho("The backtest closed no trades: there is nothing to test.",
                  fg="yellow", err=True)
            raise Exit(1)
        with Progress(SpinnerColumn(finished_text="[green]✓"), TextColumn("{task.description}"),
                      console=console) as progress:
            task = progress.add_task(f"Testing {len(closed)} trades...", total=1)
            result = analyze(closed, config)
            progress.update(task, completed=1)
        return result

    if from_trades is not None:
        if data is not None:
            secho("Give either DATA or --from-trades, not both.", fg="red", err=True)
            raise Exit(1)
        if not from_trades.exists():
            secho(f"Trade file not found: {from_trades}", fg="red", err=True)
            raise Exit(1)
        stem = script.stem
        try:
            loaded = load_trade_csv(from_trades)
        except ValueError as e:
            secho(str(e), fg="red", err=True)
            raise Exit(1)
        report = _analyze(loaded)
        if modes and coinflip_choice != 'direction':
            console.print("[yellow]The coin-flip tests run the strategy and are skipped "
                          "with --from-trades.[/]")
        _finish(report, output_dir, stem, write_json)
    else:
        if data is None:
            secho("Missing argument DATA (or use --from-trades).", fg="red", err=True)
            raise Exit(1)
        resolved = resolve_script(script, api_key)
        stem = resolved.stem
        with _backtest_session(resolved, data, time_from, time_to, timeframe,
                               security) as session:
            trades = _record_backtest(session, resolved)
            report = _analyze(trades)
            _run_coinflips(session, report, trades, modes, seed, draws, coinflip_runs, exact)
        _finish(report, output_dir, stem, write_json)
