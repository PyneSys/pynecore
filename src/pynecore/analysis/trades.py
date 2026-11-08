"""
Closed trades in the form the robustness tests work on.

A :class:`TradeRecord` is a frozen copy of one closed trade. Besides the fields of
the strategy engine's own trade it carries the trade's *initial stop*: the first
protective stop level the trade had. The money that stop puts at risk turns the
profit into an R multiple (profit / initial risk), the unit that makes trades of
different size and volatility comparable.

The engine does not keep the initial stop on its trades, so :class:`TradeRecorder`
reads it from the exit orders bound to each entry while the backtest runs.
"""

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.syminfo import SymInfo
    from ..lib.strategy import Order, SimPosition, Trade

Unit = Literal['r', 'currency', 'percent']
"""Unit of the per-trade values: R multiples, account currency, or percent return."""

_TRADE_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S%z'

# Identifies the fill a trade piece belongs to: a partial exit splits one fill into
# several closed trades that share these fields.
_FillKey = tuple[str | None, int, int, float]

# ``(stop, loss_ticks, mirrored_stop)`` of one exit leg: an absolute stop, a tick
# offset from the fill, and a stop still waiting to be reflected through the fill
# of a flipped entry (see ``strategy._EntryFlip``)
_StopLevel = tuple[float | None, float | None, float | None]


@dataclass(slots=True, frozen=True)
class TradeRecord:
    """
    One closed trade.

    :ivar entry_id: Id of the entry order that opened the trade
    :ivar direction: ``1`` for long, ``-1`` for short
    :ivar size: Absolute quantity of the trade
    :ivar entry_bar_index: Bar index of the entry fill
    :ivar entry_time: Entry fill time, Unix milliseconds
    :ivar entry_price: Entry fill price
    :ivar exit_bar_index: Bar index of the exit fill
    :ivar exit_time: Exit fill time, Unix milliseconds
    :ivar exit_price: Exit fill price
    :ivar exit_comment: Exit comment, or the exit id when the exit had no comment
    :ivar profit: Net profit in account currency, commission included
    :ivar commission: Commission paid on the trade, account currency
    :ivar point_value: Account currency per one price unit of move per one unit of
        quantity: the point value together with any conversion between the symbol
        and the account currency; ``nan`` when unknown (a trade CSV)
    :ivar entry_equity: Account equity when the trade opened, ``nan`` when unknown
    :ivar initial_stop: The first protective stop level the trade had, or ``None``
    :ivar risk: Account currency lost if the trade had exited at its initial stop,
        ``None`` without an initial stop. Not positive when the stop sat on the
        profit side of the entry (a fill gapped past it), which leaves no R.
    """

    entry_id: str | None
    direction: int
    size: float
    entry_bar_index: int
    entry_time: int
    entry_price: float
    exit_bar_index: int
    exit_time: int
    exit_price: float
    exit_comment: str
    profit: float
    commission: float
    point_value: float
    entry_equity: float
    initial_stop: float | None
    risk: float | None

    @property
    def r(self) -> float | None:
        """
        Net profit in R multiples: profit divided by the initial risk.

        :return: The R multiple, or ``None`` when the trade has no usable initial risk
        """
        if self.risk is None or not self.risk > 0.0:
            return None
        return self.profit / self.risk

    @property
    def return_fraction(self) -> float | None:
        """
        Net profit as a fraction of the equity the trade opened with.

        :return: The return, or ``None`` when the entry equity is unknown
        """
        if not self.entry_equity > 0.0:
            return None
        return self.profit / self.entry_equity

    def notional(self) -> float:
        """
        Traded value of entry and exit together, in price units times quantity.

        Cost models in basis points apply to this amount: one side on the entry,
        one on the exit.

        :return: ``size * (entry_price + exit_price)``
        """
        return self.size * (self.entry_price + self.exit_price)


def resolve_unit(trades: list[TradeRecord], unit: Unit | None = None) -> Unit:
    """
    Choose the unit the tests run in.

    Without an explicit choice the trades are measured in R when every trade had an
    initial stop, otherwise in account currency.

    :param trades: The closed trades
    :param unit: Explicit unit, or ``None`` to choose automatically
    :return: The unit
    """
    if unit is not None:
        return unit
    if trades and all(t.risk is not None for t in trades):
        return 'r'
    return 'currency'


def trade_value(trade: TradeRecord, unit: Unit) -> float | None:
    """
    The value of one trade in a unit.

    :param trade: The trade
    :param unit: The unit to measure in
    :return: The value, or ``None`` when it does not exist in the unit (no usable
        initial risk for R, no known entry equity for percent)
    """
    if unit == 'currency':
        return trade.profit
    if unit == 'r':
        return trade.r
    fraction = trade.return_fraction
    return None if fraction is None else fraction * 100.0


def trade_values(trades: list[TradeRecord], unit: Unit) -> tuple[list[float], int]:
    """
    Per-trade values in one unit.

    A trade whose value does not exist in the unit is left out and counted.

    :param trades: The closed trades
    :param unit: The unit to measure in
    :return: The values in trade order, and the number of trades left out
    """
    values: list[float] = []
    skipped = 0
    for trade in trades:
        value = trade_value(trade, unit)
        if value is None:
            skipped += 1
        else:
            values.append(value)
    return values, skipped


def merge_fills(trades: list[TradeRecord]) -> list[TradeRecord]:
    """
    Merge the pieces a partial exit split one entry fill into.

    A fill closed in parts is one decision with one initial risk: its R multiple is
    the profit of every part over the risk of the whole fill, and it is one trade
    of the hit rate and of the resampled sequences.

    The merged record has the summed size, profit, commission and risk, the
    size-weighted exit price and point value, the exit moment and comment of the
    last part, and the entry equity of the first one.

    :param trades: Closed trades in the order they closed
    :return: One record per entry fill, in the order the fills completed (the
        position of their last part)
    """
    groups: dict[tuple[str | None, int, int, int, float], list[TradeRecord]] = {}
    completed: dict[tuple[str | None, int, int, int, float], int] = {}
    for index, trade in enumerate(trades):
        key = (trade.entry_id, trade.direction, trade.entry_bar_index, trade.entry_time,
               trade.entry_price)
        groups.setdefault(key, []).append(trade)
        completed[key] = index
    merged: list[TradeRecord] = []
    for key in sorted(groups, key=completed.__getitem__):
        pieces = groups[key]
        if len(pieces) == 1:
            merged.append(pieces[0])
            continue
        first = pieces[0]
        last = max(pieces, key=lambda t: (t.exit_time, t.exit_bar_index))
        size = math.fsum(t.size for t in pieces)
        risks = [t.risk for t in pieces]
        merged.append(TradeRecord(
            entry_id=first.entry_id,
            direction=first.direction,
            size=size,
            entry_bar_index=first.entry_bar_index,
            entry_time=first.entry_time,
            entry_price=first.entry_price,
            exit_bar_index=last.exit_bar_index,
            exit_time=last.exit_time,
            exit_price=math.fsum(t.exit_price * t.size for t in pieces) / size,
            exit_comment=last.exit_comment,
            profit=math.fsum(t.profit for t in pieces),
            commission=math.fsum(t.commission for t in pieces),
            point_value=math.fsum(t.point_value * t.size for t in pieces) / size,
            entry_equity=first.entry_equity,
            initial_stop=first.initial_stop,
            risk=(math.fsum(r for r in risks if r is not None)
                  if all(r is not None for r in risks) else None),
        ))
    return merged


def _fill_key(trade: 'Trade') -> _FillKey:
    return trade.entry_id, trade.entry_bar_index, trade.entry_time, trade.entry_price


def _stop_level(order: 'Order') -> _StopLevel | None:
    """
    :param order: An exit order
    :return: Its stop level, or ``None`` when it has no stop
    """
    mirrored = order.mirror_levels[1] if order.mirror_levels is not None else None
    if order.stop is None and order.loss_ticks is None and mirrored is None:
        return None
    return order.stop, order.loss_ticks, mirrored


class TradeRecorder:
    """
    Collect the closed trades of a running backtest, with their initial stops.

    Call :meth:`on_bar` once after every bar of the run. It reads the bar's closed
    trades and the exit orders resting at the end of the bar.

    The initial stop of a trade is the first stop level among the exit orders bound
    to its entry: the ones resting when the entry filled (placed together with the
    entry), or else the ones the first script execution after the fill placed, which
    is how a stop a ``calc_on_order_fills`` re-execution places and fills within the
    entry's bar is seen too. A trade without either gets the first stop placed for
    it later, also when that stop fills on the bar it was placed on. When an exit
    carries both an absolute ``stop`` and a ``loss`` tick offset, the nearer level
    counts, as it is the one the engine fills first. Exits placed without a
    ``from_entry`` apply to every entry.

    The engine's own closed-trade list is capped, so a long backtest must be
    recorded while it runs rather than read back at the end.
    """

    __slots__ = ('trades', '_position', '_pointvalue', '_mintick', '_stops', '_previous',
                 '_booked', '_last_closed')

    def __init__(self, position: 'SimPosition', syminfo: 'SymInfo'):
        """
        :param position: The strategy position of the run (``runner.script.position``)
        :param syminfo: Symbol information of the chart
        """
        self.trades: list[TradeRecord] = []
        self._position = position
        self._pointvalue = float(syminfo.pointvalue or 1.0)
        self._mintick = float(syminfo.mintick or 0.0)
        # Initial stop per fill; None while the fill has not had a stop yet
        self._stops: dict[_FillKey, float | None] = {}
        # Stop levels of the exit orders resting at the end of the previous bar
        self._previous: dict[str | None, list[_StopLevel]] = {}
        # ``(fills before the booking, entry id, stop level)`` of every exit order
        # booked during the current bar, in booking order
        self._booked: list[tuple[int, str | None, _StopLevel]] = []
        position.exit_listener = self._on_exit_booked
        # The closed trades recorded by the last call: a bar on which the script
        # did not run (a gap bar of a higher script timeframe) leaves the previous
        # bar's list in place
        self._last_closed: list['Trade'] = []

    def on_bar(self) -> None:
        """
        Record the trades the last bar closed and update the initial stops.
        """
        position = self._position
        current = self._stop_levels()

        closed = [t for t in position.new_closed_trades
                  if not any(t is seen for seen in self._last_closed)]
        if closed:
            self._last_closed = closed
        for trade in closed:
            key = _fill_key(trade)
            stop = self._stops.get(key)
            if stop is None:
                if key not in self._stops:
                    # Opened and closed within one bar: the exits resting before
                    # the fill protected it
                    stop = self._resolve(trade, self._previous)
                if stop is None:
                    # The first stop placed after the fill, filled on the bar it
                    # was placed on
                    stop = self._resolve(trade, self._booked_after(trade))
            self.trades.append(self._record(trade, stop))

        for trade in position.open_trades:
            key = _fill_key(trade)
            known = key in self._stops
            if known and self._stops[key] is not None:
                continue
            stop = None
            if not known:
                stop = self._resolve(trade, self._previous)
            if stop is None:
                stop = self._resolve(trade, self._booked_after(trade))
            if stop is None:
                stop = self._resolve(trade, current)
            self._stops[key] = stop

        self._previous = current
        self._booked.clear()

    def _on_exit_booked(self, order: 'Order') -> None:
        """
        Note the stop level of an exit order the moment the position books it.

        :param order: The exit order
        """
        level = _stop_level(order)
        if level is not None:
            key = None if order.from_entry_na else order.order_id
            self._booked.append((order.placed_fill_seq, key, level))

    def _booked_after(self, trade: 'Trade') -> dict[str | None, list[_StopLevel]]:
        """
        Stop levels of the exits the first script execution after a fill booked.

        :param trade: A trade that opened on the current bar
        :return: Stop levels by entry id (see :meth:`_stop_levels`), from the exits
            booked first after the trade's entry fill
        """
        first = None
        levels: dict[str | None, list[_StopLevel]] = {}
        for fills, key, level in self._booked:
            if fills < trade.entry_fill_seq:
                continue
            if first is None:
                first = fills
            elif fills != first:
                break
            levels.setdefault(key, []).append(level)
        return levels

    def _stop_levels(self) -> dict[str | None, list[_StopLevel]]:
        """
        Stop levels of the live exit orders, by the entry id they are bound to.

        :return: Stop levels per entry id; ``None`` collects the exits placed
            without a ``from_entry``
        """
        levels: dict[str | None, list[_StopLevel]] = {}
        for order in self._position.exit_orders.values():
            if order.cancelled or order.filled_by_type is not None:
                continue
            level = _stop_level(order)
            if level is not None:
                key = None if order.from_entry_na else order.order_id
                levels.setdefault(key, []).append(level)
        return levels

    def _resolve(self, trade: 'Trade',
                 levels: dict[str | None, list[_StopLevel]]) -> float | None:
        """
        The stop level protecting ``trade`` among ``levels``.

        :param trade: An open or just-closed trade
        :param levels: Stop levels by entry id (see :meth:`_stop_levels`)
        :return: The nearest stop level on the loss side of the entry; the nearest
            one on the wrong side when there is no other; ``None`` without any
        """
        candidates = levels.get(trade.entry_id, []) + levels.get(None, [])
        if not candidates:
            return None
        direction = 1.0 if trade.size > 0 else -1.0
        entry = float(trade.entry_price)
        stops: list[float] = []
        for stop, loss_ticks, mirrored in candidates:
            options: list[float] = []
            if stop is not None and stop == stop:
                options.append(float(stop))
            if mirrored is not None:
                options.append(2.0 * entry - float(mirrored))
            if loss_ticks is not None and self._mintick > 0.0:
                options.append(entry - direction * self._mintick * float(loss_ticks))
            if options:
                # Both levels of one exit are live: the nearer one fills first
                stops.append(max(options, key=lambda s: direction * s))
        if not stops:
            return None
        # Distance from the entry to the stop on the loss side; the smallest one
        # is the stop the price reaches first
        protective = [s for s in stops if direction * (entry - s) > 0.0]
        if protective:
            return min(protective, key=lambda s: direction * (entry - s))
        return max(stops, key=lambda s: direction * (entry - s))

    def _record(self, trade: 'Trade', stop: float | None) -> TradeRecord:
        """
        Freeze a closed trade.

        :param trade: The closed trade
        :param stop: Its initial stop, or ``None``
        :return: The record
        """
        direction = 1 if trade.size > 0 else -1
        size = abs(float(trade.size))
        entry_price = float(trade.entry_price)
        exit_price = float(trade.exit_price)
        profit = float(trade.profit)
        commission = float(trade.commission)
        # Read the rate back from the trade itself, so a currency conversion the
        # engine applied is included
        move = direction * (exit_price - entry_price) * size
        point_value = (profit + commission) / move if move != 0.0 else self._pointvalue
        risk = None if stop is None else direction * (entry_price - stop) * size * point_value
        entry_equity = float(trade.entry_equity)
        return TradeRecord(
            entry_id=trade.entry_id,
            direction=direction,
            size=size,
            entry_bar_index=int(trade.entry_bar_index),
            entry_time=int(trade.entry_time),
            entry_price=entry_price,
            exit_bar_index=int(trade.exit_bar_index),
            exit_time=int(trade.exit_time),
            exit_price=exit_price,
            exit_comment=str(trade.exit_comment or trade.exit_id or ''),
            profit=profit,
            commission=commission,
            point_value=point_value,
            entry_equity=entry_equity if entry_equity == entry_equity else math.nan,
            initial_stop=stop,
            risk=risk,
        )


def _parse_trade_time(value: str) -> int:
    return int(datetime.strptime(value, _TRADE_TIME_FORMAT).timestamp() * 1000)


def _column(header: list[str], prefix: str) -> int:
    for i, name in enumerate(header):
        if name == prefix or name.startswith(prefix + ' '):
            return i
    raise ValueError(f"Trade CSV has no '{prefix}' column")


def load_trade_csv(path: Path) -> list[TradeRecord]:
    """
    Read the closed trades of a trade CSV written by the script runner.

    The file holds an entry row and an exit row per trade. The trades still open at
    the end of the backtest (exit signal ``Open``) are not closed trades and are left
    out. The file carries no stops, no commission and no point value, so the records
    have no R, report a commission of zero and a point value of ``nan``. Their entry
    equity is derived from the cumulative profit columns: the initial capital comes
    from the row with the largest cumulative profit percent, the one its two-decimal
    rounding distorts least. Below 1% the rounding error could exceed 0.5% of the
    capital, so the capital and every entry equity stay unknown (``nan``).

    :param path: The trade CSV
    :return: The closed trades in trade order
    :raises ValueError: When the file is not a trade CSV
    """
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header = rows[0]
    c_num = _column(header, 'Trade #')
    c_bar = _column(header, 'Bar Index')
    c_type = _column(header, 'Type')
    c_signal = _column(header, 'Signal')
    c_time = _column(header, 'Date/Time')
    c_price = _column(header, 'Price')
    c_size = _column(header, 'Contracts')
    c_profit = _column(header, 'Profit')
    c_cum = _column(header, 'Cumulative profit')
    c_cum_pct = _column(header, 'Cumulative profit %')

    entries: dict[str, list[str]] = {}
    pairs: list[tuple[list[str], list[str]]] = []
    for row in rows[1:]:
        if not row:
            continue
        number = row[c_num]
        if row[c_type].startswith('Entry'):
            entries[number] = row
            continue
        entry = entries.pop(number, None)
        if entry is None or row[c_signal] == 'Open':
            continue
        pairs.append((entry, row))

    initial_capital = math.nan
    best_pct = 0.0
    for _, row in pairs:
        cum_pct = float(row[c_cum_pct])
        if abs(cum_pct) > best_pct:
            best_pct = abs(cum_pct)
            initial_capital = float(row[c_cum]) / (cum_pct / 100.0)
    if best_pct < 1.0:
        initial_capital = math.nan

    trades: list[TradeRecord] = []
    for entry, row in pairs:
        profit = float(row[c_profit])
        entry_equity = initial_capital + float(row[c_cum]) - profit
        direction = 1 if entry[c_type].endswith('long') else -1
        size = abs(float(entry[c_size]))
        entry_price = float(entry[c_price])
        exit_price = float(row[c_price])
        trades.append(TradeRecord(
            entry_id=entry[c_signal] or None,
            direction=direction,
            size=size,
            entry_bar_index=int(entry[c_bar]),
            entry_time=_parse_trade_time(entry[c_time]),
            entry_price=entry_price,
            exit_bar_index=int(row[c_bar]),
            exit_time=_parse_trade_time(row[c_time]),
            exit_price=exit_price,
            exit_comment=row[c_signal],
            profit=profit,
            commission=0.0,
            # The profit is net of a commission the file does not state, so it
            # gives no point value
            point_value=math.nan,
            entry_equity=entry_equity if entry_equity == entry_equity else math.nan,
            initial_stop=None,
            risk=None,
        ))
    return trades
