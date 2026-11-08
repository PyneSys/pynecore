"""
Robustness analysis of strategy backtests.

The tests take the closed trades of one backtest and ask how much of its result
could be luck: how the trade order shapes the drawdown, how wide the uncertainty of
the mean trade is, and how much cost the edge survives. Every test is seeded, so the
same trades and the same seed give the same numbers.

The trades come from a running backtest through :class:`TradeRecorder`, or from a
trade CSV the runner wrote earlier through :func:`load_trade_csv`.
"""

from .trades import (TradeRecord, TradeRecorder, Unit, load_trade_csv, merge_fills, resolve_unit,
                     trade_value, trade_values)
from .metrics import Summary, summarize
from .costs import CostRow, CostTable, cost_table
from .bootstrap import BootstrapResult, WilsonInterval, block_bootstrap_mean, wilson_interval
from .montecarlo import MonteCarloResult, monte_carlo
from .periods import PeriodRow, by_period
from .report import RobustnessConfig, RobustnessReport, analyze

__all__ = [
    'TradeRecord', 'TradeRecorder', 'Unit', 'load_trade_csv', 'merge_fills', 'resolve_unit',
    'trade_value', 'trade_values',
    'Summary', 'summarize',
    'CostRow', 'CostTable', 'cost_table',
    'BootstrapResult', 'WilsonInterval', 'block_bootstrap_mean', 'wilson_interval',
    'MonteCarloResult', 'monte_carlo',
    'PeriodRow', 'by_period',
    'RobustnessConfig', 'RobustnessReport', 'analyze',
]
