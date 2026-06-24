"""
Per-chart-period accumulator for live ``request.security_lower_tf``.

In a live run the chart bar is *developing*: its lower-timeframe (LTF) array
grows as intrabars arrive and only becomes the full period ``[T, T+tf)`` once
the chart bar closes. TradingView includes the currently-developing LTF bar as
the live *last* element of the array (verified against TV: array size is
``floor(elapsed / ltf_span) + 1`` — the closed intrabars plus the one
developing intrabar, whose value tracks the live price tick-by-tick).

This container holds exactly that shape: the finalized intrabar expression
values plus an optional developing tail. It is pure value bookkeeping — it does
NOT run the script or touch shared memory; the security subprocess decides which
intrabars to feed it and re-runs the developing intrabar per tick. Keeping the
accumulation logic here (rather than as raw list surgery on the flush buffer)
keeps the "closed values + developing tail" invariant in one place.

The child must still *execute* every LTF bar it needs to maintain expression
history (e.g. a ``ta.sma`` inside the LTF expression); only array *membership*
is bounded by the chart period, so the child appends to this accumulator only
for intrabars inside ``[period_start, period_end)``.
"""

from typing import Any

__all__ = ["LiveLtfWindow"]


class LiveLtfWindow:
    """Accumulates one chart period's LTF intrabar values for the live path.

    The array published to the script is ``closed values + [developing tail]``,
    so a developing chart bar reports ``len(closed) + 1`` elements while it is
    forming and exactly ``len(closed)`` once the developing tail has been
    finalized into the closed list at the chart period's end.
    """

    def __init__(self) -> None:
        self._closed: list[Any] = []
        self._developing: Any = None
        self._has_developing: bool = False

    def reset(self) -> None:
        """Drop all state for the start of a new chart period."""
        self._closed.clear()
        self._developing = None
        self._has_developing = False

    def append_closed(self, value: Any) -> None:
        """Append a finalized (confirmed) intrabar value.

        Used for closed intrabars that are not the finalization of the current
        developing tail (those go through :meth:`finalize_developing`).
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

        Called when the developing LTF intrabar closes (a confirmed bar arrives
        for its timestamp); the confirmed ``value`` may differ from the last
        developing tick, so it is supplied explicitly rather than reusing the
        in-flight tail. The array length is unchanged — the tail slot simply
        becomes a closed element.
        """
        self._closed.append(value)
        self._developing = None
        self._has_developing = False

    def drop_developing(self) -> None:
        """Discard the developing tail without finalizing it into the closed list.

        Used when a chart bar confirms but the developing intrabar's close never
        arrived (a provider gap): the unconfirmed tail must not remain in the
        confirmed period array, and unlike :meth:`finalize_developing` there is no
        confirmed value to keep — the array length shrinks by one.
        """
        self._developing = None
        self._has_developing = False

    @property
    def has_developing(self) -> bool:
        """Whether a developing tail is currently present."""
        return self._has_developing

    def publish(self) -> list[Any]:
        """Return a copy of the current array (closed values + developing tail).

        A fresh list is returned each call so the consumer may hand it across
        the result block without the accumulator's later mutations leaking in.
        """
        if self._has_developing:
            return [*self._closed, self._developing]
        return list(self._closed)

    def __len__(self) -> int:
        return len(self._closed) + (1 if self._has_developing else 0)
