"""
Per-chart-period accumulator for live ``request.security_lower_tf``.

In a live run the chart bar is *developing*: its lower-timeframe (LTF) array
grows as intrabars arrive and only becomes the full period ``[T, T+tf)`` once
the chart bar closes. TradingView includes the currently-developing LTF bar as
the live *last* element of the array (verified against TV: array size is
``floor(elapsed / ltf_span) + 1`` — the closed intrabars plus the one
developing intrabar, whose value tracks the live price tick-by-tick).

This container holds exactly that shape in three ordered segments:

* ``closed`` — finalized (confirmed-close) intrabar values;
* ``provisional`` — intrabars that have rolled over (a newer developing intrabar
  started) but whose confirmed close has not arrived yet, so their value is the
  last developing tick and is *replaced* in place when the late close lands;
* ``developing`` — the single currently-forming intrabar as the live tail.

The provisional segment exists only because a live feed can deliver the forming
tick of intrabar ``N+1`` ahead of the late close of intrabar ``N`` (e.g.
cTrader closes a slot on the next slot's open, or plain stream reordering). The
array must still grow by one when ``N+1`` starts — ``N`` is held provisionally
at its last developing value until its close replaces it at the same position.

It is pure value bookkeeping — it does NOT run the script or touch shared
memory; the security subprocess decides which intrabars to feed it, re-runs the
developing intrabar per tick, and replays the provisional chain when a late
close shifts the baseline. Keeping the accumulation logic here (rather than as
raw list surgery on the flush buffer) keeps the "closed + provisional +
developing" invariant in one place.

The child must still *execute* every LTF bar it needs to maintain expression
history (e.g. a ``ta.sma`` inside the LTF expression); only array *membership*
is bounded by the chart period, so the child appends to this accumulator only
for intrabars inside ``[period_start, period_end)``.
"""

from typing import Any

__all__ = ["LiveLtfWindow"]


class LiveLtfWindow:
    """Accumulates one chart period's LTF intrabar values for the live path.

    The array published to the script is ``closed + provisional + [developing]``.
    A developing chart bar reports ``len(closed) + len(provisional) + 1`` elements
    while a tail is forming and exactly ``len(closed)`` once every tail has been
    finalized into the closed list at the chart period's end. The provisional
    segment carries intrabars that have rolled over but whose confirmed close has
    not yet arrived; their values are revised in place by a replay and become
    closed (front-to-back) as their late closes land.
    """

    def __init__(self) -> None:
        self._closed: list[Any] = []
        self._provisional: list[Any] = []
        self._developing: Any = None
        self._has_developing: bool = False

    def reset(self) -> None:
        """Drop all state for the start of a new chart period."""
        self._closed.clear()
        self._provisional.clear()
        self._developing = None
        self._has_developing = False

    def append_closed(self, value: Any) -> None:
        """Append a finalized (confirmed) intrabar value.

        Used for closed intrabars that are neither the finalization of the
        current developing tail (:meth:`finalize_developing`) nor of a pending
        provisional (:meth:`finalize_first_provisional`).
        """
        self._closed.append(value)

    def set_developing(self, value: Any) -> None:
        """Set or replace the developing tail value.

        Called on every tick of the currently-developing LTF intrabar; the
        previous tick's value is discarded (the developing bar occupies a
        single array slot whose value is revised in place).
        """
        self._developing = value
        self._has_developing = True

    def finalize_developing(self, value: Any) -> None:
        """Finalize the developing tail into the closed list.

        Called when the developing LTF intrabar closes in order (a confirmed bar
        arrives for its timestamp with no provisional chain ahead of it); the
        confirmed ``value`` may differ from the last developing tick, so it is
        supplied explicitly rather than reusing the in-flight tail. The array
        length is unchanged — the tail slot simply becomes a closed element.
        """
        self._closed.append(value)
        self._developing = None
        self._has_developing = False

    def promote_developing_to_provisional(self) -> None:
        """Roll the developing tail into the provisional segment.

        Called when a newer developing intrabar starts before the current one's
        close has arrived: the current tail keeps its (last developing) value
        and position, becoming provisional, so the array still grows by one when
        the new developing tail is set. A no-op if there is no developing tail.
        """
        if self._has_developing:
            self._provisional.append(self._developing)
            self._developing = None
            self._has_developing = False

    def append_provisional(self, value: Any) -> None:
        """Append a new provisional entry behind the existing provisional chain.

        Called when a brand-new closed intrabar arrives while an older provisional
        is still unresolved (a provider gap left the chain head open). Its
        confirmed value cannot join the closed segment yet — that sits ahead of
        the provisional chain, so appending there would place the newer bar before
        the older, still-pending one. It is held provisionally at the back instead,
        in timestamp order, and finalized once the chain ahead of it drains.
        """
        self._provisional.append(value)

    def finalize_first_provisional(self, value: Any) -> None:
        """Promote the earliest provisional entry to a confirmed closed value.

        Called when the late close of the oldest pending provisional arrives.
        The provisional slot at the front becomes a closed element at the SAME
        array position (closed entries sit ahead of provisional ones); the
        confirmed ``value`` replaces the last-developing value it was holding.
        """
        del self._provisional[0]
        self._closed.append(value)

    def replace_provisional(self, index: int, value: Any) -> None:
        """Revise a still-pending provisional value in place (replay).

        After a late close shifts the baseline, the remaining provisional
        intrabars are re-run on the corrected state; their values are written
        back at their existing positions without changing the array length.
        """
        self._provisional[index] = value

    def drop_developing(self) -> None:
        """Discard the developing tail without finalizing it into the closed list.

        Used when a chart bar confirms but the developing intrabar's close never
        arrived (a provider gap): the unconfirmed tail must not remain in the
        confirmed period array, and unlike :meth:`finalize_developing` there is no
        confirmed value to keep — the array length shrinks by one.
        """
        self._developing = None
        self._has_developing = False

    def drop_provisionals(self) -> None:
        """Discard every still-pending provisional entry.

        Used at chart confirm when one or more rolled-over intrabars never
        received their close (a provider gap or grace timeout): like an
        unconfirmed developing tail, an unconfirmed provisional must not remain
        in the confirmed period array. Already-finalized provisionals live in the
        closed segment and are unaffected.
        """
        self._provisional.clear()

    @property
    def has_developing(self) -> bool:
        """Whether a developing tail is currently present."""
        return self._has_developing

    @property
    def has_provisional(self) -> bool:
        """Whether any pending (rolled-over, not-yet-closed) provisional remains."""
        return bool(self._provisional)

    def publish(self) -> list[Any]:
        """Return a copy of the current array (closed + provisional + developing).

        A fresh list is returned each call so the consumer may hand it across
        the result block without the accumulator's later mutations leaking in.
        """
        out = [*self._closed, *self._provisional]
        if self._has_developing:
            out.append(self._developing)
        return out

    def __len__(self) -> int:
        return len(self._closed) + len(self._provisional) + (1 if self._has_developing else 0)
