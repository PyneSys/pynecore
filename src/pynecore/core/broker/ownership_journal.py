"""The row journal the one-way emulator keeps its per-leg bracket ownership in.

The :class:`~pynecore.core.broker.one_way_emulator.OneWayEmulator` records which
exit owns which hedging leg as order rows (see the bracket-ownership helpers in
:mod:`~pynecore.core.broker.store_helpers`). Those helpers and the emulator only
need the handful of row operations :class:`OwnershipJournal` names. A run with a
broker store keeps the rows in it (:class:`~pynecore.core.broker.storage.RunContext`
satisfies the protocol as it is), so they survive a restart. A run without one
keeps them in a :class:`MemoryOwnershipJournal`: the same bookkeeping, the same
owner merge and retirement, for the lifetime of the process only.
"""
import contextlib
import copy
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, Protocol

from pynecore.core.broker.storage import OrderRow

__all__ = [
    'OwnershipJournal',
    'MemoryOwnershipJournal',
]

#: Columns :meth:`MemoryOwnershipJournal.upsert_order` accepts besides ``extras``.
_COLUMNS = frozenset({
    'intent_key', 'exchange_order_id', 'symbol', 'side', 'qty', 'filled_qty', 'state',
    'from_entry', 'pine_entry_id', 'sl_level', 'tp_level', 'trailing_stop',
    'trailing_distance',
})

#: Columns a new row must be given.
_REQUIRED = ('symbol', 'side', 'qty', 'state')


class OwnershipJournal(Protocol):
    """The order-row operations the bracket-ownership bookkeeping runs on.

    Each method has the meaning of its
    :class:`~pynecore.core.broker.storage.RunContext` namesake; a row returned is
    a snapshot, a later write does not change it.
    """

    def get_order(self, client_order_id: str) -> OrderRow | None:
        """The row ``client_order_id``, live or closed, or ``None``."""
        ...

    def upsert_order(self, client_order_id: str, **fields: Any) -> None:
        """Insert the row, or write the given fields of the existing one."""
        ...

    def reopen_order(self, client_order_id: str) -> None:
        """Return a closed row to the live set; no-op for a live or absent row."""
        ...

    def close_order(self, client_order_id: str) -> None:
        """Take the row out of the live set."""
        ...

    def iter_live_orders(self, *, from_entry: str | None = None) -> Iterator[OrderRow]:
        """The live rows, optionally only those of one parent entry."""
        ...

    def transaction(self) -> contextlib.AbstractContextManager[Any]:
        """One all-or-nothing span over the writes made inside it; nestable."""
        ...


class MemoryOwnershipJournal:
    """An :class:`OwnershipJournal` held in process memory.

    Used when the engine runs without a broker store. It mirrors the row
    semantics of :class:`~pynecore.core.broker.storage.RunContext`: a new row
    needs ``symbol``, ``side``, ``qty`` and ``state``, an update writes only the
    fields passed (``extras`` replaced whole), reads return copies, and the live
    rows are iterated in insertion order. ``updated_ts_ms`` grows strictly with
    every write, so the most recently written row is always the newest one.
    """

    def __init__(self) -> None:
        self._rows: dict[str, OrderRow] = {}
        self._last_ts_ms = 0
        self._span_depth = 0

    def _stamp(self) -> int:
        """Epoch milliseconds of a write, strictly after the previous write's."""
        self._last_ts_ms = max(int(time.time() * 1000), self._last_ts_ms + 1)
        return self._last_ts_ms

    @staticmethod
    def _copy(row: OrderRow) -> OrderRow:
        return replace(row, extras=copy.deepcopy(row.extras))

    def get_order(self, client_order_id: str) -> OrderRow | None:
        """The row ``client_order_id``, live or closed, or ``None``."""
        row = self._rows.get(client_order_id)
        return self._copy(row) if row is not None else None

    def upsert_order(self, client_order_id: str, **fields: Any) -> None:
        """Insert the row, or write the given fields of the existing one.

        :raises ValueError: On an unknown field, or a new row missing a
            required field.
        """
        extras = fields.pop('extras', None)
        unknown = set(fields) - _COLUMNS
        if unknown:
            raise ValueError(f"upsert_order({client_order_id!r}): unknown fields {sorted(unknown)}")
        now = self._stamp()
        existing = self._rows.get(client_order_id)
        if existing is None:
            missing = [name for name in _REQUIRED if name not in fields]
            if missing:
                raise ValueError(
                    f"upsert_order({client_order_id!r}) new row, "
                    f"missing required fields: {missing}"
                )
            self._rows[client_order_id] = OrderRow(
                client_order_id=client_order_id,
                plugin_name='',
                intent_key=fields.get('intent_key'),
                exchange_order_id=fields.get('exchange_order_id'),
                symbol=fields['symbol'],
                side=fields['side'],
                qty=fields['qty'],
                filled_qty=fields.get('filled_qty', 0.0),
                state=fields['state'],
                from_entry=fields.get('from_entry'),
                pine_entry_id=fields.get('pine_entry_id'),
                sl_level=fields.get('sl_level'),
                tp_level=fields.get('tp_level'),
                trailing_stop=bool(fields.get('trailing_stop', False)),
                trailing_distance=fields.get('trailing_distance'),
                created_ts_ms=now,
                updated_ts_ms=now,
                closed_ts_ms=None,
                extras=copy.deepcopy(extras) if extras is not None else {},
            )
            return
        if 'trailing_stop' in fields:
            fields['trailing_stop'] = bool(fields['trailing_stop'])
        if extras is not None:
            fields['extras'] = copy.deepcopy(extras)
        self._rows[client_order_id] = replace(existing, updated_ts_ms=now, **fields)

    def reopen_order(self, client_order_id: str) -> None:
        """Return a closed row to the live set; no-op for a live or absent row."""
        row = self._rows.get(client_order_id)
        if row is None or row.closed_ts_ms is None:
            return
        now = self._stamp()
        self._rows[client_order_id] = replace(
            row, closed_ts_ms=None, created_ts_ms=now, updated_ts_ms=now)

    def close_order(self, client_order_id: str) -> None:
        """Take the row out of the live set; no-op for an absent row."""
        row = self._rows.get(client_order_id)
        if row is None:
            return
        now = self._stamp()
        self._rows[client_order_id] = replace(row, closed_ts_ms=now, updated_ts_ms=now)

    def iter_live_orders(self, *, from_entry: str | None = None) -> Iterator[OrderRow]:
        """The live rows in insertion order, optionally only those of one parent entry."""
        rows = [
            row for row in self._rows.values()
            if row.closed_ts_ms is None and (from_entry is None or row.from_entry == from_entry)
        ]
        for row in rows:
            yield self._copy(row)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """One all-or-nothing span over the writes made inside it.

        Mirrors :meth:`~pynecore.core.broker.storage.RunContext.transaction`: a
        nested span joins the outermost one, and an exception leaving the
        outermost span restores every row to its state at the span's start.
        """
        if self._span_depth:
            self._span_depth += 1
            try:
                yield
            finally:
                self._span_depth -= 1
            return
        saved = dict(self._rows)
        self._span_depth = 1
        try:
            yield
        except BaseException:
            self._rows = saved
            raise
        finally:
            self._span_depth = 0
