"""
Unit tests for ``LiveLtfWindow`` — the per-chart-period accumulator for live
``request.security_lower_tf``.

The oracle is TradingView's live developing-bar behaviour (verified by a
realtime probe on BINANCE:BTCUSDT, 30m chart / 3m LTF): the array is the closed
intrabars of the current chart period PLUS the currently-developing intrabar as
the live last element, so its size is ``floor(elapsed / ltf_span) + 1`` and it
steps up by one exactly when an intrabar closes and the next one opens.
"""
from pynecore.core.live_ltf_window import LiveLtfWindow


def __test_closed_only_publishes_copy_of_values__():
    """With no developing tail the array is just the closed values, copied out."""
    w = LiveLtfWindow()
    w.append_closed(10.0)
    w.append_closed(20.0)
    assert w.publish() == [10.0, 20.0]
    assert len(w) == 2
    assert not w.has_developing


def __test_developing_tail_is_live_last_element__():
    """The developing intrabar is appended as the last array element.

    This is the TV-probe shape mid-period: 1 closed intrabar + 1 developing ->
    size 2, the developing value last.
    """
    w = LiveLtfWindow()
    w.append_closed(100.0)          # the closed [T, T+ltf) intrabar
    w.set_developing(101.5)         # the developing [T+ltf, T+2ltf) intrabar
    assert w.publish() == [100.0, 101.5]
    assert len(w) == 2
    assert w.has_developing


def __test_set_developing_revises_in_place__():
    """Each tick replaces the developing tail; it never grows the array."""
    w = LiveLtfWindow()
    w.append_closed(100.0)
    w.set_developing(101.0)
    w.set_developing(102.0)
    w.set_developing(101.5)
    assert w.publish() == [100.0, 101.5]   # only the latest tick survives
    assert len(w) == 2


def __test_finalize_developing_uses_confirmed_value__():
    """Finalizing moves the tail into the closed list with the confirmed value.

    The confirmed close may differ from the last developing tick, so it is
    supplied explicitly; the array length is unchanged.
    """
    w = LiveLtfWindow()
    w.append_closed(100.0)
    w.set_developing(101.9)            # last developing tick
    w.finalize_developing(102.0)       # confirmed close differs from the tick
    assert w.publish() == [100.0, 102.0]
    assert len(w) == 2
    assert not w.has_developing


def __test_probe_boundary_increment_2_to_3__():
    """Reproduce the probe's n: 2 -> 3 step at an intrabar close.

    Mid-period there is 1 closed + 1 developing (size 2). When that developing
    intrabar closes and the next one opens, the array becomes 2 closed + 1
    developing (size 3) — exactly the observed BTCUSDT 30/3 transition.
    """
    w = LiveLtfWindow()
    w.append_closed(1.0)              # intrabar 0 closed
    w.set_developing(2.0)            # intrabar 1 developing
    assert len(w) == 2

    w.finalize_developing(2.5)       # intrabar 1 closes
    w.set_developing(3.0)            # intrabar 2 opens developing
    assert w.publish() == [1.0, 2.5, 3.0]
    assert len(w) == 3


def __test_reset_clears_all_state__():
    """A new chart period starts from empty."""
    w = LiveLtfWindow()
    w.append_closed(1.0)
    w.set_developing(2.0)
    w.reset()
    assert w.publish() == []
    assert len(w) == 0
    assert not w.has_developing


def __test_publish_returns_independent_copy__():
    """Mutating a published array must not corrupt the accumulator."""
    w = LiveLtfWindow()
    w.append_closed(1.0)
    w.set_developing(2.0)
    out = w.publish()
    out.append(999.0)
    out[0] = -1.0
    assert w.publish() == [1.0, 2.0]


def __test_none_developing_value_is_distinct_from_no_developing__():
    """A developing tail whose value is ``None`` still counts as present.

    LTF expression values can legitimately be ``na``/``None``; the presence of a
    developing tail is tracked by a flag, not by a sentinel value.
    """
    w = LiveLtfWindow()
    w.append_closed(1.0)
    assert len(w) == 1
    w.set_developing(None)
    assert w.has_developing
    assert len(w) == 2
    assert w.publish() == [1.0, None]
