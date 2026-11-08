"""
All robustness tests of one backtest, and their CSV and JSON output.

:func:`analyze` runs the tests on a list of closed trades. The
:class:`RobustnessReport` it returns writes one CSV file per kind of result, sized
for a spreadsheet, and gives the whole result as a JSON-ready dict.
"""

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .bootstrap import (Block, BootstrapResult, WilsonInterval, block_bootstrap_mean,
                        wilson_interval, DEFAULT_CONFIDENCE)
from .costs import CostTable, cost_table, DEFAULT_COST_BPS
from .metrics import Summary, summarize
from .montecarlo import Method, MonteCarloResult, monte_carlo, PERCENTILES
from .periods import Period, PeriodRow, by_period
from .trades import TradeRecord, Unit, merge_fills, resolve_unit, trade_value

if TYPE_CHECKING:
    from .coinflip import CoinFlipResult

SCHEMA_VERSION = 1
"""Version of the :meth:`RobustnessReport.to_dict` layout."""

DEFAULT_SEED = 0
DEFAULT_DRAWS = 10_000
DEFAULT_RISK_FRACTION = 0.01
"""Account risked per trade when R multiples are compounded: 1%."""
DEFAULT_RUIN_FRACTION = 0.5
"""Loss of the starting account that counts as ruin: half of it."""
DEFAULT_ACCOUNT = 1000.0
"""Starting account when the trades carry no equity."""

_UNIT_LABELS: dict[Unit, str] = {'r': 'R', 'currency': 'currency', 'percent': '%'}


@dataclass(slots=True)
class RobustnessConfig:
    """
    Settings of the robustness tests.

    :ivar unit: Unit of the per-trade values, ``None`` to choose automatically (R
        when every trade had an initial stop, account currency otherwise)
    :ivar seed: Seed of every random test
    :ivar draws: Resamples per random test
    :ivar mc_method: Monte Carlo resampling method
    :ivar block: Bootstrap block
    :ivar confidence: Coverage of the intervals
    :ivar cost_bps: Extra cost scenarios per side, in basis points
    :ivar periods: Calendar breakdowns to report
    :ivar initial_capital: Starting account of the account Monte Carlo, ``None`` for
        the equity the first trade opened with
    :ivar risk_fraction: Account risked per trade when the account Monte Carlo
        compounds R multiples; with other units it compounds the strategy's own
        per-trade returns
    :ivar ruin_fraction: Loss of the starting account that counts as ruin
    :ivar target_multiple: Multiple of the starting account counted as reaching the
        target, ``None`` for no target
    :ivar max_leverage: Largest position of the account Monte Carlo of R multiples, as a
        multiple of the account: a trade whose stop is too close for
        :attr:`risk_fraction` at this leverage risks less. ``None`` for no limit.
    """

    unit: Unit | None = None
    seed: int = DEFAULT_SEED
    draws: int = DEFAULT_DRAWS
    mc_method: Method = 'bootstrap'
    block: Block = 'week'
    confidence: float = DEFAULT_CONFIDENCE
    cost_bps: tuple[float, ...] = DEFAULT_COST_BPS
    periods: tuple[Period, ...] = ('year',)
    initial_capital: float | None = None
    risk_fraction: float = DEFAULT_RISK_FRACTION
    ruin_fraction: float = DEFAULT_RUIN_FRACTION
    target_multiple: float | None = None
    max_leverage: float | None = None


@dataclass(slots=True)
class RobustnessReport:
    """
    The results of every robustness test of one backtest.

    :ivar config: The settings the tests ran with
    :ivar unit: Unit of the per-trade values
    :ivar trades: Closed trades analysed, the parts of a partially closed fill counted
        as one
    :ivar skipped: Trades left out because they have no value in the unit
    :ivar summary: Statistics of the backtest's own trade sequence
    :ivar hit_rate: Wilson interval of the hit rate
    :ivar bootstrap: Bootstrap interval of the mean trade
    :ivar monte_carlo: Monte Carlo of the running sum of the values
    :ivar account: Monte Carlo of a compounding account, ``None`` when the trades
        give no per-trade return
    :ivar account_sizing: How the account form sized its trades: ``risk`` (a fixed
        share of the account per R) or ``strategy`` (the backtest's own returns)
    :ivar account_capped: Trades the leverage limit made risk less than the set share
        of the account, ``None`` without a limit or with ``strategy`` sizing
    :ivar costs: The result under extra costs
    :ivar periods: Calendar breakdowns
    :ivar coinflips: Coin-flip tests; they run the strategy again, so the caller
        adds them (see :mod:`~pynecore.analysis.coinflip`)
    """

    config: RobustnessConfig
    unit: Unit
    trades: int
    skipped: int
    summary: Summary
    hit_rate: WilsonInterval
    bootstrap: BootstrapResult
    monte_carlo: MonteCarloResult
    account: MonteCarloResult | None
    account_sizing: str | None
    account_capped: int | None
    costs: CostTable
    periods: list[PeriodRow] = field(default_factory=list)
    coinflips: list['CoinFlipResult'] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """:return: The whole result, JSON-ready"""
        config = self.config
        return {
            'schema_version': SCHEMA_VERSION,
            'config': {
                'unit': config.unit, 'seed': config.seed, 'draws': config.draws,
                'mc_method': config.mc_method, 'block': config.block,
                'confidence': config.confidence, 'cost_bps': list(config.cost_bps),
                'periods': list(config.periods), 'initial_capital': config.initial_capital,
                'risk_fraction': config.risk_fraction, 'ruin_fraction': config.ruin_fraction,
                'target_multiple': config.target_multiple,
                'max_leverage': config.max_leverage,
            },
            'unit': self.unit,
            'trades': self.trades,
            'skipped': self.skipped,
            'summary': self.summary.to_dict(),
            'hit_rate': self.hit_rate.to_dict(),
            'bootstrap': self.bootstrap.to_dict(),
            'monte_carlo': self.monte_carlo.to_dict(),
            'account': self.account.to_dict() if self.account is not None else None,
            'account_sizing': self.account_sizing,
            'account_capped': self.account_capped,
            'costs': self.costs.to_dict(),
            'periods': [row.to_dict() for row in self.periods],
            'coinflips': [result.to_dict() for result in self.coinflips],
        }

    def summary_rows(self) -> list[dict[str, Any]]:
        """
        One row per measured figure, in the column layout of the summary CSV.

        :return: Rows with the keys of :data:`SUMMARY_COLUMNS`
        """
        unit = _UNIT_LABELS[self.unit]
        cfg = self.config
        s = self.summary
        rows: list[dict[str, Any]] = [
            _row('Backtest', 'Trades', '', s.trades),
            _row('Backtest', 'Total', unit, s.total),
            _row('Backtest', 'Mean per trade', unit, s.mean),
            _row('Backtest', 'Median per trade', unit, s.median),
            _row('Backtest', 'Profit factor', '', s.profit_factor),
            _row('Backtest', 'Max drawdown (closed trades)', unit, s.max_drawdown),
            _row('Backtest', 'Largest win', unit, s.largest_win),
            _row('Backtest', 'Largest loss', unit, s.largest_loss),
            _row('Backtest', 'Total without the 3 best trades', unit, s.total_without_top),
            _row('Backtest', 'Longest losing streak', 'trades', s.longest_losing_streak),
        ]
        h = self.hit_rate
        rows.append(_row('Hit rate', 'Hit rate', '%', _pct(h.rate), low=_pct(h.low),
                         high=_pct(h.high), method=f"Wilson {cfg.confidence:.0%}"))
        b = self.bootstrap
        rows.append(_row(
            'Bootstrap', 'Mean per trade', unit, b.mean, low=b.low, high=b.high,
            draws=b.used_draws, seed=b.seed,
            method=f"{b.block} blocks ({b.blocks}), {cfg.confidence:.0%} interval",
            verdict=('interval excludes zero' if b.excludes_zero else 'interval includes zero')))
        rows.append(_row('Bootstrap', 'Resampled means at or below zero', '%',
                         _pct(b.share_not_above_zero), draws=b.used_draws, seed=b.seed))
        rows.extend(_monte_carlo_rows('Monte Carlo', self.monte_carlo, unit))
        if self.account is not None:
            sizing = (f"{cfg.risk_fraction:.2%} of the account per R"
                      if self.account_sizing == 'risk' else 'the strategy\'s own returns')
            if self.account_sizing == 'risk' and cfg.max_leverage is not None:
                sizing += (f", positions at most {cfg.max_leverage:g}x the account"
                           f" ({self.account_capped} of {self.account.trades} trades capped)")
            rows.extend(_monte_carlo_rows('Account Monte Carlo', self.account, 'currency',
                                          note=f"compounding {sizing}"))
        for flip in self.coinflips:
            test = f"Coin flip ({flip.mode})"
            flip_unit = _UNIT_LABELS[flip.unit]
            d = flip.distribution
            rows.append(_row(test, 'Real total vs random totals', flip_unit, flip.real,
                             low=d['p5'], median=d['p50'], high=d['p95'], draws=flip.draws,
                             seed=flip.seed, method=flip.description, verdict=flip.verdict))
            rows.append(_row(test, 'Random draws that did better', '%',
                             _pct(flip.share_better), draws=flip.draws, seed=flip.seed))
        c = self.costs
        if c.break_even_bps is not None:
            rows.append(_row('Costs', 'Break-even extra cost per side', 'bps', c.break_even_bps,
                             method='extra cost on the traded value, fills unchanged'))
        return rows

    def write_csv(self, directory: Path, stem: str) -> list[Path]:
        """
        Write the results as CSV files.

        - ``<stem>_robustness.csv``: one row per measured figure
        - ``<stem>_montecarlo.csv``: the path percentiles by trade number
        - ``<stem>_costs.csv``: the result under each extra cost
        - ``<stem>_periods.csv``: the calendar breakdowns
        - ``<stem>_coinflip_<mode>.csv``: the random totals of each coin-flip test

        :param directory: Output directory
        :param stem: File name stem, usually the script's
        :return: The files written
        """
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        path = directory / f"{stem}_robustness.csv"
        _write(path, SUMMARY_COLUMNS, [[row[c] for c in SUMMARY_COLUMNS]
                                       for row in self.summary_rows()])
        written.append(path)

        path = directory / f"{stem}_montecarlo.csv"
        header = ['Trade'] + [f"Sum P{q:g}" for q in PERCENTILES]
        if self.account is not None:
            header += [f"Account P{q:g}" for q in PERCENTILES]
        rows: list[list[Any]] = []
        for i, (number, values) in enumerate(self.monte_carlo.bands):
            row: list[Any] = [number, *(values[f"p{q:g}"] for q in PERCENTILES)]
            if self.account is not None:
                account_values = self.account.bands[i][1]
                row += [account_values[f"p{q:g}"] for q in PERCENTILES]
            rows.append(row)
        _write(path, header, rows)
        written.append(path)

        path = directory / f"{stem}_costs.csv"
        unit = _UNIT_LABELS[self.unit]
        _write(path, ['Extra cost per side (bps)', 'Trades', f"Total ({unit})",
                      f"Mean per trade ({unit})", 'Hit rate %', 'Profit factor',
                      f"Max drawdown ({unit})"],
               [[r.bps, r.summary.trades, r.summary.total, r.summary.mean,
                 _pct(r.summary.hit_rate), r.summary.profit_factor, r.summary.max_drawdown]
                for r in self.costs.rows])
        written.append(path)

        if self.periods:
            path = directory / f"{stem}_periods.csv"
            _write(path, ['Period', 'Label', 'Trades', f"Total ({unit})",
                          f"Mean per trade ({unit})", 'Hit rate %', 'Profit factor',
                          f"Max drawdown ({unit})"],
                   [[p.period, p.label, p.summary.trades, p.summary.total, p.summary.mean,
                     _pct(p.summary.hit_rate), p.summary.profit_factor, p.summary.max_drawdown]
                    for p in self.periods])
            written.append(path)

        for flip in self.coinflips:
            path = directory / f"{stem}_coinflip_{flip.mode}.csv"
            label = _UNIT_LABELS[flip.unit]
            _write(path, ['Draw', f"Random total ({label})", f"Real total ({label})"],
                   [[i, total, flip.real] for i, total in enumerate(flip.totals, 1)])
            written.append(path)
        return written


SUMMARY_COLUMNS: tuple[str, ...] = (
    'Test', 'Metric', 'Unit', 'Value', 'Low / P5', 'Median / P50', 'High / P95',
    'Draws', 'Seed', 'Method', 'Verdict',
)
"""Columns of ``<stem>_robustness.csv``."""


def _pct(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def _row(test: str, metric: str, unit: str, value: Any, *, low: Any = None, median: Any = None,
         high: Any = None, draws: int | None = None, seed: int | None = None,
         method: str = '', verdict: str = '') -> dict[str, Any]:
    return {'Test': test, 'Metric': metric, 'Unit': unit, 'Value': value, 'Low / P5': low,
            'Median / P50': median, 'High / P95': high, 'Draws': draws, 'Seed': seed,
            'Method': method, 'Verdict': verdict}


def _monte_carlo_rows(test: str, result: MonteCarloResult, unit: str,
                      note: str = '') -> list[dict[str, Any]]:
    method = f"{result.method}" + (f", {note}" if note else '')
    dd_unit = '%' if result.compound else unit
    dd = result.max_drawdown
    scale = 100.0 if result.compound else 1.0

    def scaled(value: float | None) -> float | None:
        return None if value is None else value * scale

    rows = [
        _row(test, 'Final value', unit, result.final['p50'], low=result.final['p5'],
             median=result.final['p50'], high=result.final['p95'], draws=result.draws,
             seed=result.seed, method=method),
        _row(test, 'Max drawdown', dd_unit, scaled(dd['p50']), low=scaled(dd['p5']),
             median=scaled(dd['p50']), high=scaled(dd['p95']), draws=result.draws,
             seed=result.seed, method=method),
        _row(test, 'Paths ending at or below the start', '%', _pct(result.share_losing),
             draws=result.draws, seed=result.seed, method=method),
    ]
    if result.share_ruined is not None and result.ruin_level is not None:
        rows.append(_row(test, f"Paths ruined (account at or below {result.ruin_level:g})", '%',
                         _pct(result.share_ruined), draws=result.draws, seed=result.seed,
                         method=method))
    if result.share_target is not None and result.target_level is not None:
        rows.append(_row(test, f"Paths reaching {result.target_level:g}", '%',
                         _pct(result.share_target), draws=result.draws, seed=result.seed,
                         method=method))
    return rows


def _cell(value: Any) -> Any:
    if value is None:
        return ''
    if isinstance(value, float):
        if not math.isfinite(value):
            return '' if math.isnan(value) else ('inf' if value > 0 else '-inf')
        # ``+ 0.0`` turns a negative zero into zero
        return round(value, 10) + 0.0
    return value


def _write(path: Path, header: list[str] | tuple[str, ...], rows: list[list[Any]]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(v) for v in row])


def _risk_returns(trades: list[TradeRecord], risk_fraction: float,
                  max_leverage: float | None) -> tuple[list[float], int | None]:
    """
    Per-trade account returns of a fixed share of the account risked per R.

    Risking ``risk_fraction`` of the account with a stop ``d`` away from the entry
    takes a position of ``risk_fraction / d`` times the account. With a leverage limit
    the position stays at most ``max_leverage`` times the account, so a trade with a
    closer stop risks ``max_leverage * d`` instead.

    :param trades: The closed trades, one per fill
    :param risk_fraction: Account risked per R
    :param max_leverage: Largest position as a multiple of the account, ``None`` for
        no limit
    :return: The returns of the trades measured in R, and the number of trades the
        limit capped (``None`` without a limit)
    """
    returns: list[float] = []
    capped = 0
    for trade in trades:
        r = trade.r
        if r is None:
            continue
        fraction = risk_fraction
        if max_leverage is not None and trade.initial_stop is not None and trade.entry_price > 0.0:
            stop_distance = abs(trade.entry_price - trade.initial_stop) / trade.entry_price
            if max_leverage * stop_distance < risk_fraction:
                fraction = max_leverage * stop_distance
                capped += 1
        returns.append(fraction * r)
    return returns, capped if max_leverage is not None else None


def analyze(trades: list[TradeRecord], config: RobustnessConfig | None = None) -> RobustnessReport:
    """
    Run every robustness test on the closed trades of one backtest.

    The parts of a partially closed entry fill are merged first (see
    :func:`merge_fills`): every test sees one trade per fill.

    :param trades: The closed trades in trade order
    :param config: Settings, defaults when ``None``
    :return: The results
    """
    cfg = config if config is not None else RobustnessConfig()
    trades = merge_fills(trades)
    unit = resolve_unit(trades, cfg.unit)

    values: list[float] = []
    times: list[int] = []
    for trade in trades:
        value = trade_value(trade, unit)
        if value is not None:
            values.append(value)
            times.append(trade.exit_time)
    skipped = len(trades) - len(values)

    summary = summarize(values)

    account: MonteCarloResult | None = None
    sizing: str | None = None
    returns: list[float] | None = None
    capped: int | None = None
    if unit == 'r':
        returns, capped = _risk_returns(trades, cfg.risk_fraction, cfg.max_leverage)
        sizing = 'risk'
    else:
        fractions = [t.return_fraction for t in trades]
        if fractions and all(f is not None for f in fractions):
            returns = [f for f in fractions if f is not None]
            sizing = 'strategy'
    if returns is not None:
        capital = cfg.initial_capital
        if capital is None:
            first_equity = trades[0].entry_equity if trades else math.nan
            capital = first_equity if first_equity > 0.0 else DEFAULT_ACCOUNT
        account = monte_carlo(returns, method=cfg.mc_method, draws=cfg.draws, seed=cfg.seed,
                              compound=True, initial_capital=capital,
                              ruin_fraction=cfg.ruin_fraction,
                              target_multiple=cfg.target_multiple)

    return RobustnessReport(
        config=cfg,
        unit=unit,
        trades=len(trades),
        skipped=skipped,
        summary=summary,
        hit_rate=wilson_interval(summary.wins, summary.trades, cfg.confidence),
        bootstrap=block_bootstrap_mean(values, times, block=cfg.block, draws=cfg.draws,
                                       seed=cfg.seed, confidence=cfg.confidence),
        monte_carlo=monte_carlo(values, method=cfg.mc_method, draws=cfg.draws, seed=cfg.seed),
        account=account,
        account_sizing=sizing if account is not None else None,
        account_capped=capped if account is not None else None,
        costs=cost_table(trades, unit, cfg.cost_bps),
        periods=[row for period in cfg.periods for row in by_period(values, times, period)],
    )
