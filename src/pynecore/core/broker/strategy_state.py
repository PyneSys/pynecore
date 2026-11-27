"""
The strategy statistics a live run keeps across restarts and server moves.

A live run's :class:`~pynecore.core.broker.position.BrokerPosition` holds what
``strategy.netprofit``, ``strategy.closedtrades`` and the rest of the
``strategy.*`` namespace report. The sync engine stores it in the broker store
(``strategy_state`` and ``strategy_closed_trades``) in the transaction that
books the fills, and the next process of the run restores it before the first
live bar. This module holds the record formats and the bar-index remap.

Bar indexes are per process: a restarted run numbers its bars from the first
bar of its own warm-up history, so the same bar gets another index. Every
stored trade therefore carries the bar it was written on (its
``bar_anchor``: that bar's index and open time in the writer's numbering), and
:class:`BarIndexRemap` moves a restored trade's ``entry_bar_index`` and
``exit_bar_index`` by the anchor bar's index in the new numbering, which the
runner reports bar by bar (:meth:`BarIndexRemap.observe`). A trade without an
anchor, or one closed before the new process ran its first bar, takes the
index of the bar its fill time falls in instead.
"""
import heapq
import math
from typing import Any, Final

# noinspection PyProtectedMember
from pynecore.core.closed_trade_stats import ClosedTradeStats, _SideClosedTradeStats
from pynecore.lib.strategy import Trade
from pynecore.types.na import na_float

__all__ = [
    'STRATEGY_STATE_VERSION',
    'BarIndexRemap',
    'trade_to_payload',
    'trade_from_payload',
    'trade_anchor_of_payload',
    'closed_trade_stats_to_payload',
    'closed_trade_stats_from_payload',
]

STRATEGY_STATE_VERSION: Final[int] = 1

_TRADE_TEXT_FIELDS: Final = ('entry_id', 'entry_comment', 'exit_id', 'exit_comment')
_TRADE_INT_FIELDS: Final = ('entry_bar_index', 'entry_time', 'entry_fill_node',
                            'entry_fill_seq', 'exit_bar_index', 'exit_time')
_TRADE_FLOAT_FIELDS: Final = tuple(
    name for name in Trade.__slots__
    if name not in _TRADE_TEXT_FIELDS and name not in _TRADE_INT_FIELDS
)

_SIDE_INT_FIELDS: Final = ('count', 'winning_count', 'bar_length_sum', 'bar_length_count')
_SIDE_FLOAT_FIELDS: Final = tuple(
    name for name in _SideClosedTradeStats.__slots__ if name not in _SIDE_INT_FIELDS
)
_STATS_INT_FIELDS: Final = (
    'winning_count', 'losing_count', 'bar_length_sum', 'bar_length_count',
    'winning_bar_length_sum', 'winning_bar_length_count',
    'losing_bar_length_sum', 'losing_bar_length_count',
    'current_winning_streak', 'current_losing_streak',
    'max_winning_streak', 'max_losing_streak',
)
_STATS_FLOAT_FIELDS: Final = tuple(
    name for name in ClosedTradeStats.__slots__
    if name not in _STATS_INT_FIELDS
    and name not in ('month_profit', 'first_entry_time', 'long', 'short')
)


def encode_float(value: Any) -> float | None:
    """A float for a JSON record: ``None`` for a NaN or an infinity, which JSON cannot hold."""
    number = float(value)
    return number if math.isfinite(number) else None


def decode_float(value: Any, missing: float = na_float) -> float:
    """The float of a JSON record field; ``missing`` for ``None``.

    :raises ValueError: When the value is not a number.
    """
    if value is None:
        return missing
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"not a number: {value!r}")
    return float(value)


def decode_int(value: Any) -> int:
    """The int of a JSON record field.

    :raises ValueError: When the value is not a whole number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ValueError(f"not a whole number: {value!r}")
    return int(value)


def trade_to_payload(trade: Trade, anchor: tuple[int, int] | None) -> dict[str, Any]:
    """The JSON record of a trade.

    :param trade: An open or closed trade.
    :param anchor: The bar the trade's bar indexes are numbered against: its
        index and open time (epoch ms) in the same numbering; ``None`` when no
        bar ran in this numbering, so the indexes mean nothing on their own.
    :return: Every field of the trade, plus ``bar_anchor``.
    """
    payload: dict[str, Any] = {}
    for name in _TRADE_TEXT_FIELDS:
        value = getattr(trade, name)
        payload[name] = value if isinstance(value, str) else None
    for name in _TRADE_INT_FIELDS:
        payload[name] = int(getattr(trade, name))
    for name in _TRADE_FLOAT_FIELDS:
        payload[name] = encode_float(getattr(trade, name))
    payload['bar_anchor'] = None if anchor is None else [int(anchor[0]), int(anchor[1])]
    return payload


def trade_from_payload(payload: dict[str, Any]) -> Trade:
    """The trade of a JSON record written by :func:`trade_to_payload`.

    :raises ValueError: When a field is missing or malformed.
    """
    trade = Trade.__new__(Trade)
    try:
        for name in _TRADE_TEXT_FIELDS:
            value = payload[name]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"trade field {name} is not a string: {value!r}")
            setattr(trade, name, value)
        for name in _TRADE_INT_FIELDS:
            setattr(trade, name, decode_int(payload[name]))
        for name in _TRADE_FLOAT_FIELDS:
            setattr(trade, name, decode_float(payload[name]))
    except KeyError as exc:
        raise ValueError(f"trade field {exc} is missing") from exc
    if trade.exit_comment is None:
        trade.exit_comment = ''
    return trade


def trade_anchor_of_payload(payload: dict[str, Any]) -> tuple[int, int] | None:
    """The ``bar_anchor`` of a trade record, ``None`` when it has none.

    :raises ValueError: When the anchor is malformed.
    """
    anchor = payload.get('bar_anchor')
    if anchor is None:
        return None
    if not isinstance(anchor, list) or len(anchor) != 2:
        raise ValueError(f"malformed bar anchor: {anchor!r}")
    return decode_int(anchor[0]), decode_int(anchor[1])


def closed_trade_stats_to_payload(stats: ClosedTradeStats) -> dict[str, Any]:
    """The JSON record of the cumulative closed-trade statistics."""
    payload: dict[str, Any] = {
        'month_profit': {str(month): encode_float(profit)
                         for month, profit in stats.month_profit.items()},
        'first_entry_time': stats.first_entry_time,
    }
    for name in _STATS_INT_FIELDS:
        payload[name] = int(getattr(stats, name))
    for name in _STATS_FLOAT_FIELDS:
        payload[name] = encode_float(getattr(stats, name))
    for side_name in ('long', 'short'):
        side = getattr(stats, side_name)
        record: dict[str, Any] = {}
        for name in _SIDE_INT_FIELDS:
            record[name] = int(getattr(side, name))
        for name in _SIDE_FLOAT_FIELDS:
            record[name] = encode_float(getattr(side, name))
        payload[side_name] = record
    return payload


def closed_trade_stats_from_payload(payload: dict[str, Any]) -> ClosedTradeStats:
    """The statistics of a JSON record written by :func:`closed_trade_stats_to_payload`.

    :raises ValueError: When a field is missing or malformed.
    """
    stats = ClosedTradeStats()
    try:
        months = payload['month_profit']
        if not isinstance(months, dict):
            raise ValueError("month_profit is not an object")
        stats.month_profit = {decode_int(int(month)): decode_float(profit, 0.0)
                              for month, profit in months.items()}
        first = payload['first_entry_time']
        stats.first_entry_time = None if first is None else decode_int(first)
        for name in _STATS_INT_FIELDS:
            setattr(stats, name, decode_int(payload[name]))
        for name in _STATS_FLOAT_FIELDS:
            setattr(stats, name, decode_float(payload[name], 0.0))
        for side_name in ('long', 'short'):
            record = payload[side_name]
            if not isinstance(record, dict):
                raise ValueError(f"{side_name} is not an object")
            side = getattr(stats, side_name)
            for name in _SIDE_INT_FIELDS:
                setattr(side, name, decode_int(record[name]))
            for name in _SIDE_FLOAT_FIELDS:
                setattr(side, name, decode_float(record[name], 0.0))
    except KeyError as exc:
        raise ValueError(f"closed-trade statistics field {exc} is missing") from exc
    return stats


class BarIndexRemap:
    """Moves restored trades' bar indexes into the numbering of the process that restored them.

    Two kinds of trades wait here:

    - **Anchored** trades (:meth:`add_anchored`): their indexes count in the
      numbering of the anchor bar they were stored with. When the runner
      shows the bar that opened at the anchor's time, the trades move by that
      bar's new index less the anchor's old one, exactly. An anchor bar the new
      history does not hold (older than its first bar, or missing from it) is
      placed by the bar period: the first later bar's index less the whole
      periods between the two open times, an estimate wherever the market
      skips bars (sessions, weekends).
    - **Timed** trades (:meth:`add_timed`): their indexes mean nothing in any
      numbering (stored without an anchor, or closed before this process ran
      its first bar). Each index becomes the index of the bar its time (the
      fill's ``entry_time`` / ``exit_time``) falls in: the last bar that
      opened at or before it, known once a later bar arrives; a time before
      the first bar is placed by the bar period. A time of ``0`` (a trade a
      restart adopted without its entry time) keeps its index.

    A placement by the bar period needs the period, measured from the first
    two bars, so it waits for the second bar.
    """

    __slots__ = ('_anchored', '_anchor_times', '_timed', '_timed_queue', '_backlog',
                 '_owed', '_seq', '_first', '_prev', '_period', 'trade_anchors', 'approximate')

    def __init__(self) -> None:
        # Anchor time -> (anchor index, the trades numbered against it), and the
        # anchor times as a min-heap.
        self._anchored: dict[int, tuple[int, list[Trade]]] = {}
        self._anchor_times: list[int] = []
        # id(trade) -> (trade, the indexes it waits for: 'entry' / 'exit' -> the
        # wait's sequence number), and the waits as a min-heap of (time, sequence,
        # id, which). A wait the dict no longer holds is stale and skipped.
        self._timed: dict[int, tuple[Trade, dict[str, int]]] = {}
        self._timed_queue: list[tuple[int, int, int, str]] = []
        # Placements before the first bar that wait for the bar period:
        # ('anchor', time) and ('timed', time, sequence, id, which).
        self._backlog: list[tuple[Any, ...]] = []
        # id() of the closed trades whose bar length the closed-trade statistics
        # still lack (closed while their exit bar was unknown).
        self._owed: set[int] = set()
        self._seq = 0
        self._first: tuple[int, int] | None = None
        self._prev: tuple[int, int] | None = None
        self._period: int | None = None
        # id(trade) -> the anchor a waiting anchored trade is still numbered against,
        # for a write of it before the remap is done.
        self.trade_anchors: dict[int, tuple[int, int]] = {}
        # Whether a trade was placed by the bar period rather than by its own bar.
        self.approximate = False

    @property
    def done(self) -> bool:
        """Whether no trade waits any more."""
        return not self._anchored and not self._timed

    def add_anchored(self, trade: Trade, anchor: tuple[int, int]) -> None:
        """Let a trade numbered against ``anchor`` (index, open time) wait for the anchor bar."""
        index, time_ms = anchor
        entry = self._anchored.get(time_ms)
        if entry is None:
            self._anchored[time_ms] = (index, [trade])
            heapq.heappush(self._anchor_times, time_ms)
        elif entry[0] == index:
            entry[1].append(trade)
        else:
            # Two numberings anchored on one open time (a timeframe change
            # between processes): the later one is placed by its own times.
            self.add_timed(trade)
            return
        self.trade_anchors[id(trade)] = (index, time_ms)

    def add_timed(self, trade: Trade) -> None:
        """Let a trade take the indexes of the bars its entry and exit times fall in.

        A closed trade whose exit index is unknown (``exit_bar_index`` below
        0) lacks its bar length in the closed-trade statistics:
        :meth:`observe` returns it once its exit is placed. A known exit index
        stays until then, so the statistics rebuilt meanwhile keep the trade's
        bar length.
        """
        self.forget(trade)
        waits: dict[str, int] = {}
        if trade.entry_time > 0:
            waits['entry'] = self._queue(trade.entry_time, trade, 'entry')
        if trade.exit_time > 0:
            if trade.exit_bar_index < 0:
                self._owed.add(id(trade))
            waits['exit'] = self._queue(trade.exit_time, trade, 'exit')
        if waits:
            self._timed[id(trade)] = (trade, waits)

    def _queue(self, time_ms: int, trade: Trade, which: str) -> int:
        """Queue one wait; returns its sequence number."""
        self._seq += 1
        heapq.heappush(self._timed_queue, (time_ms, self._seq, id(trade), which))
        return self._seq

    def follow(self, source: Trade, piece: Trade) -> None:
        """Let ``piece``, split off ``source``, wait the way ``source`` waits."""
        anchor = self.trade_anchors.get(id(source))
        if anchor is not None:
            self.add_anchored(piece, anchor)
        elif id(source) in self._timed:
            self.add_timed(piece)

    def forget(self, trade: Trade) -> None:
        """Stop moving ``trade`` (it is placed by its times instead, or gone)."""
        anchor = self.trade_anchors.pop(id(trade), None)
        if anchor is not None:
            entry = self._anchored.get(anchor[1])
            if entry is not None:
                kept = [held for held in entry[1] if held is not trade]
                if kept:
                    self._anchored[anchor[1]] = (entry[0], kept)
                else:
                    del self._anchored[anchor[1]]
        self._timed.pop(id(trade), None)
        self._owed.discard(id(trade))

    def waits_timed(self, trade: Trade) -> bool:
        """Whether ``trade`` waits for an index of its own time."""
        return id(trade) in self._timed

    def observe(self, bar_index: int, bar_time: int) -> list[Trade]:
        """Show the remap the bar the runner is about to execute.

        :param bar_index: The bar's index in this process's numbering.
        :param bar_time: The bar's open time (epoch ms).
        :return: The closed trades whose bar length the closed-trade
            statistics lack and whose indexes are both placed now.
        """
        if self._first is None:
            self._first = (bar_index, bar_time)
        elif self._period is None and bar_time > self._first[1]:
            self._period = bar_time - self._first[1]
        placed: list[Trade] = []
        if self._period is not None and self._backlog:
            backlog, self._backlog = self._backlog, []
            for item in backlog:
                if item[0] == 'anchor':
                    self._move_anchor(item[1], bar_index, bar_time)
                else:
                    self._place_timed(item[1], item[2], item[3], item[4], bar_index, bar_time,
                                      placed)
        while self._anchor_times and self._anchor_times[0] <= bar_time:
            anchor_time = heapq.heappop(self._anchor_times)
            if anchor_time not in self._anchored:
                continue
            if anchor_time != bar_time and self._period is None:
                self._backlog.append(('anchor', anchor_time))
                continue
            self._move_anchor(anchor_time, bar_index, bar_time)
        while self._timed_queue and self._timed_queue[0][0] < bar_time:
            time_ms, seq, key, which = heapq.heappop(self._timed_queue)
            self._place_timed(time_ms, seq, key, which, bar_index, bar_time, placed)
        self._prev = (bar_index, bar_time)
        owed: list[Trade] = []
        for trade in placed:
            if id(trade) in self._owed and id(trade) not in self._timed:
                self._owed.discard(id(trade))
                owed.append(trade)
        return owed

    def _move_anchor(self, anchor_time: int, bar_index: int, bar_time: int) -> None:
        """Move the trades of one anchor: exactly on its own bar, else by the bar period."""
        entry = self._anchored.pop(anchor_time, None)
        if entry is None:
            return
        if anchor_time == bar_time:
            new_index = bar_index
        else:
            assert self._period is not None
            new_index = bar_index - max(1, round((bar_time - anchor_time) / self._period))
            self.approximate = True
        old_index, trades = entry
        shift = new_index - old_index
        for trade in trades:
            self.trade_anchors.pop(id(trade), None)
            trade.entry_bar_index += shift
            if trade.exit_bar_index >= 0:
                trade.exit_bar_index += shift

    def _place_timed(self, time_ms: int, seq: int, key: int, which: str, bar_index: int,
                     bar_time: int, placed: list[Trade]) -> None:
        """Give one waiting index the bar ``time_ms`` falls in, now a later bar opened."""
        held = self._timed.get(key)
        if held is None or held[1].get(which) != seq:
            return
        trade, waits = held
        if self._prev is not None and self._prev[1] <= time_ms < bar_time:
            index = self._prev[0]
        elif self._period is None:
            self._backlog.append(('timed', time_ms, seq, key, which))
            return
        else:
            assert self._first is not None
            first_index, first_time = self._first
            if time_ms >= first_time:
                # Between two bars the feed had no bar for: the earlier bar.
                index = bar_index - math.ceil((bar_time - time_ms) / self._period)
            else:
                index = first_index - math.ceil((first_time - time_ms) / self._period)
            self.approximate = True
        if which == 'entry':
            trade.entry_bar_index = index
        else:
            trade.exit_bar_index = index
        del waits[which]
        if not waits:
            del self._timed[key]
            placed.append(trade)
