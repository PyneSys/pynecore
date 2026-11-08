"""
How much extra trading cost the result survives.

Each scenario charges every trade an extra cost in basis points of its traded value,
one side on the entry and one on the exit, on top of what the backtest already paid.
The fills stay as they were: this measures the cost, it does not re-run the strategy
under it, so a strategy whose signals depend on its own equity can react differently.
"""

import math
from dataclasses import dataclass
from typing import Any

from .metrics import Summary, summarize
from .trades import TradeRecord, Unit, trade_value

DEFAULT_COST_BPS: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)
"""Extra cost scenarios per side, in basis points."""


@dataclass(slots=True, frozen=True)
class CostRow:
    """
    The result under one extra cost.

    :ivar bps: Extra cost per side, in basis points of the traded value
    :ivar summary: Statistics of the per-trade values after the cost
    """

    bps: float
    summary: Summary

    def to_dict(self) -> dict[str, Any]:
        """:return: JSON-ready form"""
        return {'bps': self.bps, **self.summary.to_dict()}


@dataclass(slots=True, frozen=True)
class CostTable:
    """
    The result under each extra cost scenario.

    :ivar unit: Unit of the values
    :ivar rows: One row per scenario, in the order given; empty when no trade has a
        known cost (a trade CSV carries no point value)
    :ivar break_even_bps: Extra cost per side that brings the total to zero; ``0.0``
        when the total is not positive to begin with, ``None`` when the trades
        carry no traded value
    """

    unit: Unit
    rows: list[CostRow]
    break_even_bps: float | None

    def to_dict(self) -> dict[str, Any]:
        """:return: JSON-ready form"""
        return {
            'unit': self.unit,
            'break_even_bps': self.break_even_bps,
            'rows': [row.to_dict() for row in self.rows],
        }


def cost_per_bps(trade: TradeRecord, unit: Unit) -> float | None:
    """
    The cost of one basis point per side on one trade, in a unit.

    :param trade: The trade
    :param unit: The unit of the result
    :return: The cost, or ``None`` when the trade has no value in the unit or no
        known point value
    """
    money = trade.notional() * trade.point_value / 10_000.0
    if not math.isfinite(money):
        return None
    if unit == 'currency':
        return money
    if unit == 'r':
        return money / trade.risk if trade.risk is not None and trade.risk > 0.0 else None
    if trade.entry_equity > 0.0:
        return money / trade.entry_equity * 100.0
    return None


def cost_table(trades: list[TradeRecord], unit: Unit,
               bps_grid: tuple[float, ...] = DEFAULT_COST_BPS) -> CostTable:
    """
    The result under each extra cost scenario.

    :param trades: The closed trades
    :param unit: The unit to measure in
    :param bps_grid: Extra cost scenarios per side, in basis points
    :return: One row per scenario and the break-even cost
    """
    pairs: list[tuple[float, float]] = []
    for trade in trades:
        value = trade_value(trade, unit)
        unit_cost = cost_per_bps(trade, unit)
        if value is not None and unit_cost is not None:
            pairs.append((value, unit_cost))

    rows = [CostRow(bps=bps, summary=summarize([v - bps * c for v, c in pairs]))
            for bps in bps_grid] if pairs else []

    total = math.fsum(v for v, _ in pairs)
    total_cost = math.fsum(c for _, c in pairs)
    if total_cost <= 0.0:
        break_even = None
    else:
        break_even = max(total, 0.0) / total_cost
    return CostTable(unit=unit, rows=rows, break_even_bps=break_even)
