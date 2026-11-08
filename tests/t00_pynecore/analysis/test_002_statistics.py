"""
The robustness statistics on hand-made trade lists with known answers.
"""
import csv
import math
from datetime import datetime, UTC

from pynecore.analysis import (RobustnessConfig, TradeRecord, analyze, block_bootstrap_mean,
                               by_period, cost_table, load_trade_csv, merge_fills, monte_carlo,
                               summarize, wilson_interval)
from pynecore.analysis.metrics import max_drawdown, percentile

_DAY = 86_400_000
_START = int(datetime(2025, 1, 6, tzinfo=UTC).timestamp() * 1000)  # a Monday


def __test_helper_trade(i: int, r: float, *, risk: float | None = 100.0,
                        time_ms: int | None = None) -> TradeRecord:
    """A long trade of one unit risking ``risk`` that made ``r`` R."""
    entry = 1000.0
    profit = r * (risk if risk is not None else 100.0)
    exit_time = time_ms if time_ms is not None else _START + i * _DAY
    return TradeRecord(
        entry_id='L', direction=1, size=1.0, entry_bar_index=i, entry_time=exit_time - 3_600_000,
        entry_price=entry, exit_bar_index=i, exit_time=exit_time, exit_price=entry + profit,
        exit_comment='', profit=profit, commission=0.0, point_value=1.0, entry_equity=10_000.0,
        initial_stop=None if risk is None else entry - risk, risk=risk)


def __test_summary__():
    """Totals, hit rate, profit factor, drawdown, streak and concentration."""
    s = summarize([1.0, -1.0, -1.0, 2.0, -1.0, 3.0])
    assert s.trades == 6 and s.wins == 3 and s.losses == 3
    assert s.total == 3.0 and s.hit_rate == 0.5
    assert s.profit_factor == 2.0
    assert s.max_drawdown == 2.0
    assert s.longest_losing_streak == 2
    assert s.top_total == 6.0 and s.total_without_top == -3.0
    assert summarize([1.0]).profit_factor == math.inf
    empty = summarize([])
    assert empty.trades == 0 and math.isnan(empty.mean)
    assert empty.to_dict()['mean'] is None


def __test_drawdown_counts_the_start_as_a_peak__():
    """A sequence that starts with losses has drawn down from zero."""
    assert max_drawdown([-1.0, -2.0, 4.0, -1.0]) == 3.0


def __test_percentile_interpolates__():
    """Linear interpolation between the closest ranks."""
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 100.0) == 4.0
    assert math.isnan(percentile([], 50.0))


def __test_wilson_interval__():
    """Known values of the Wilson score interval."""
    w = wilson_interval(50, 100)
    assert abs(w.low - 0.4038) < 1e-4 and abs(w.high - 0.5962) < 1e-4
    zero = wilson_interval(0, 10)
    assert zero.low == 0.0 and 0.0 < zero.high < 0.35


def __test_bootstrap_is_seeded_and_counts_quiet_weeks__():
    """Same seed, same interval; the weeks without trades are blocks too."""
    values = [1.0, -0.5, 2.0, -1.0]
    times = [_START, _START + _DAY, _START + 21 * _DAY, _START + 22 * _DAY]
    a = block_bootstrap_mean(values, times, block='week', draws=2000, seed=7)
    b = block_bootstrap_mean(values, times, block='week', draws=2000, seed=7)
    assert a == b
    assert a.blocks == 4
    assert a.low <= a.mean <= a.high
    assert block_bootstrap_mean(values, times, block='trade', draws=10, seed=1).blocks == 4
    assert block_bootstrap_mean(values, times, block='month', draws=10, seed=1).blocks == 1


def __test_bootstrap_of_a_clear_edge_excludes_zero__():
    """Every trade a winner: the whole interval is above zero."""
    values = [1.0 + (i % 3) * 0.1 for i in range(60)]
    times = [_START + i * _DAY for i in range(60)]
    result = block_bootstrap_mean(values, times, draws=1000, seed=3)
    assert result.excludes_zero and result.share_not_above_zero == 0.0


def __test_shuffle_keeps_the_total__():
    """Reordering changes the drawdown, never the additive total."""
    values = [2.0, -1.0, -1.0, 3.0, -1.0, 0.5]
    result = monte_carlo(values, method='shuffle', draws=500, seed=11)
    assert result.final['p5'] == result.final['p95'] == sum(values)
    assert result.max_drawdown['p5'] <= result.max_drawdown['p95']
    assert result.bands[-1][0] == len(values)


def __test_bootstrap_monte_carlo_is_seeded__():
    """Same seed, same distribution."""
    values = [2.0, -1.0, -1.0, 3.0, -1.0, 0.5]
    assert (monte_carlo(values, draws=300, seed=5).to_dict()
            == monte_carlo(values, draws=300, seed=5).to_dict())


def __test_account_ruin_and_target__():
    """A sure loss ruins every account, a sure win reaches every target."""
    ruined = monte_carlo([-0.2] * 10, draws=50, seed=1, compound=True, initial_capital=100.0,
                         ruin_fraction=0.5)
    assert ruined.share_ruined == 1.0
    assert ruined.final['p50'] == 100.0 * 0.8 ** 4
    reached = monte_carlo([0.5] * 10, draws=50, seed=1, compound=True, initial_capital=100.0,
                          ruin_fraction=0.5, target_multiple=10.0)
    assert reached.share_target == 1.0 and reached.share_ruined == 0.0


def __test_costs_and_break_even__():
    """An extra cost is charged on the traded value of both sides."""
    trades = [__test_helper_trade(i, r) for i, r in enumerate([1.0, 1.0, -1.0])]
    table = cost_table(trades, 'currency', (0.0, 10.0))
    assert table.rows[0].summary.total == 100.0
    # Entry 1000 and exit 1100 / 1100 / 900 -> traded value 6100; 10 bps of it is 6.1
    assert abs(table.rows[1].summary.total - (100.0 - 6.1)) < 1e-9
    assert table.break_even_bps is not None
    assert abs(table.break_even_bps - 100.0 / 0.61) < 1e-9
    r_table = cost_table(trades, 'r', (10.0,))
    assert abs(r_table.rows[0].summary.total - (1.0 - 0.061)) < 1e-9


def __test_periods__():
    """Trades are grouped by the calendar period of their exit, in order."""
    jan = int(datetime(2024, 1, 3, tzinfo=UTC).timestamp() * 1000)
    feb = int(datetime(2025, 2, 3, tzinfo=UTC).timestamp() * 1000)
    rows = by_period([1.0, -1.0, 2.0], [feb, jan, feb], 'year')
    assert [(r.label, r.summary.total) for r in rows] == [('2024', -1.0), ('2025', 3.0)]
    assert [r.label for r in by_period([1.0], [feb], 'weekday')] == ['Mon']


def __test_analyze_chooses_the_unit__():
    """R with stops on every trade, currency when one is missing."""
    with_stops = [__test_helper_trade(i, r) for i, r in enumerate([1.0, -1.0, 2.0])]
    report = analyze(with_stops, RobustnessConfig(draws=200))
    assert report.unit == 'r' and report.summary.total == 2.0
    assert report.account is not None and report.account_sizing == 'risk'
    mixed = with_stops + [__test_helper_trade(3, 1.0, risk=None)]
    assert analyze(mixed, RobustnessConfig(draws=200)).unit == 'currency'


def __test_account_leverage_limit__():
    """A stop too close for the risk at the leverage limit risks less; others keep it."""
    tight = __test_helper_trade(0, -1.0, risk=1.0)  # stop 0.1% away: 1% risk is 10x
    wide = __test_helper_trade(1, 2.0, risk=100.0)  # stop 10% away: 1% risk is 0.1x
    free = analyze([tight, wide], RobustnessConfig(draws=50, mc_method='shuffle'))
    capped = analyze([tight, wide], RobustnessConfig(draws=50, mc_method='shuffle',
                                                     max_leverage=1.0))
    assert free.account is not None and capped.account is not None
    assert free.account_capped is None and capped.account_capped == 1
    assert abs(free.account.final['p50'] - 10_000.0 * 0.99 * 1.02) < 1e-6
    assert abs(capped.account.final['p50'] - 10_000.0 * 0.999 * 1.02) < 1e-6
    assert capped.to_dict()['config']['max_leverage'] == 1.0
    assert any('1 of 2 trades capped' in row['Method'] for row in capped.summary_rows())


def __test_csv_output__(tmp_path):
    """One file per kind of result; no negative zero reaches a cell."""
    trades = [__test_helper_trade(i, r) for i, r in enumerate([1.0, -1.0, 2.0, -1.0, 0.0])]
    report = analyze(trades, RobustnessConfig(draws=200, periods=('year', 'weekday')))
    written = report.write_csv(tmp_path, 'demo')
    assert sorted(p.name for p in written) == ['demo_costs.csv', 'demo_montecarlo.csv',
                                               'demo_periods.csv', 'demo_robustness.csv']
    for path in written:
        text = path.read_text(encoding='utf-8')
        assert '-0.0,' not in text and not text.rstrip().endswith('-0.0')
    with open(tmp_path / 'demo_robustness.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r['Test'] == 'Backtest' and r['Metric'] == 'Total')
    assert total['Unit'] == 'R' and float(total['Value']) == 1.0


def __test_load_trade_csv__(tmp_path):
    """The runner's trade CSV loads its closed trades and skips the open one."""
    path = tmp_path / 'demo_trade.csv'
    path.write_text(
        'Trade #,Bar Index,Type,Signal,Date/Time,Price USD,Contracts,Profit USD,Profit %,'
        'Cumulative profit USD,Cumulative profit %,Run-up USD,Run-up %,Drawdown USD,Drawdown %\n'
        '1,8,Entry long,L,2025-01-05T00:00:00+0000,100,1,0,0.00,0,0.00,0,0,0,0\n'
        '1,9,Exit long,LX,2025-01-05T04:00:00+0000,100,1,0,0.00,0,0.00,0,0,0,0\n'
        '2,10,Entry long,L,2025-01-06T00:00:00+0000,100,2,20,10.00,20,2.00,25,1,0,0\n'
        '2,12,Exit long,LX,2025-01-06T08:00:00+0000,110,2,20,10.00,20,2.00,25,1,0,0\n'
        '3,15,Entry short,S,2025-01-07T00:00:00+0000,110,1,-5,-4.55,15,1.50,0,0,5,1\n'
        '3,16,Exit short,SX,2025-01-07T04:00:00+0000,115,1,-5,-4.55,15,1.50,0,0,5,1\n'
        '4,20,Entry long,L,2025-01-08T00:00:00+0000,112,1,0,0.00,0,0.00,0,0,0,0\n'
        '4,25,Exit long,Open,2025-01-08T20:00:00+0000,113,1,1,0.89,16,1.60,0,0,0,0\n',
        encoding='utf-8')
    trades = load_trade_csv(path)
    assert [(t.direction, t.size, t.profit, t.exit_comment) for t in trades] == [
        (1, 1.0, 0.0, 'LX'), (1, 2.0, 20.0, 'LX'), (-1, 1.0, -5.0, 'SX')]
    assert trades[1].entry_time == _START
    # The capital comes from the largest cumulative percent and reaches the leading
    # break-even trade as well
    assert [t.entry_equity for t in trades] == [1_000.0, 1_000.0, 1_020.0]
    # The net profit hides the commission, so the file gives no point value
    assert all(math.isnan(t.point_value) for t in trades) and trades[0].r is None
    table = cost_table(trades, 'currency', (0.0, 10.0))
    assert table.rows == [] and table.break_even_bps is None


def __test_load_trade_csv_rounded_capital__(tmp_path):
    """A cumulative percent under 1% is too coarse to give the capital."""
    path = tmp_path / 'demo_trade.csv'
    path.write_text(
        'Trade #,Bar Index,Type,Signal,Date/Time,Price USD,Contracts,Profit USD,Profit %,'
        'Cumulative profit USD,Cumulative profit %,Run-up USD,Run-up %,Drawdown USD,Drawdown %\n'
        '1,10,Entry long,L,2025-01-06T00:00:00+0000,100,1,0.6,0.60,0.6,0.01,1,1,0,0\n'
        '1,12,Exit long,LX,2025-01-06T08:00:00+0000,100.6,1,0.6,0.60,0.6,0.01,1,1,0,0\n',
        encoding='utf-8')
    trades = load_trade_csv(path)
    assert len(trades) == 1 and math.isnan(trades[0].entry_equity)


def __test_partial_exits_merge_into_one_fill__():
    """The parts of one fill are one trade with one initial risk."""
    base = __test_helper_trade(0, 0.0)
    quarter = TradeRecord(
        entry_id='L', direction=1, size=0.25, entry_bar_index=0, entry_time=base.entry_time,
        entry_price=100.0, exit_bar_index=1, exit_time=base.exit_time, exit_price=120.0,
        exit_comment='TP', profit=5.0, commission=0.0, point_value=1.0, entry_equity=10_000.0,
        initial_stop=90.0, risk=2.5)
    rest = TradeRecord(
        entry_id='L', direction=1, size=0.75, entry_bar_index=0, entry_time=base.entry_time,
        entry_price=100.0, exit_bar_index=2, exit_time=base.exit_time + _DAY, exit_price=90.0,
        exit_comment='SL', profit=-7.5, commission=0.0, point_value=1.0, entry_equity=10_000.0,
        initial_stop=90.0, risk=7.5)
    merged = merge_fills([quarter, rest])
    assert len(merged) == 1
    fill = merged[0]
    assert fill.size == 1.0 and fill.risk == 10.0 and fill.r == -0.25
    assert fill.exit_price == 97.5 and fill.exit_comment == 'SL'
    report = analyze([quarter, rest], RobustnessConfig(draws=200))
    assert report.trades == 1 and report.summary.total == -0.25


def __test_merged_fills_keep_completion_order__():
    """A fill closed in parts takes the place of its last part among the other trades."""
    def piece(entry_bar, exit_bar, size, profit):
        return TradeRecord(
            entry_id='L', direction=1, size=size, entry_bar_index=entry_bar,
            entry_time=_START + entry_bar * _DAY, entry_price=100.0, exit_bar_index=exit_bar,
            exit_time=_START + exit_bar * _DAY, exit_price=100.0 + profit / size,
            exit_comment='', profit=profit, commission=0.0, point_value=1.0,
            entry_equity=10_000.0, initial_stop=90.0, risk=10.0 * size)

    a_first = piece(0, 1, 0.5, 50.0)
    b = piece(1, 2, 1.0, 200.0)
    c = piece(2, 3, 1.0, -100.0)
    a_rest = piece(0, 4, 0.5, -150.0)
    merged = merge_fills([a_first, b, c, a_rest])
    assert [(t.entry_bar_index, t.exit_bar_index) for t in merged] == [(1, 2), (2, 3), (0, 4)]
    assert max_drawdown([t.profit for t in merged]) == 200.0
