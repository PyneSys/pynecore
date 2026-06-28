"""Unit tests for the live LTF grace-wait collect loop.

``security_process._collect_in_period_intrabars`` collects a chart period's
closed intrabars and, on a confirmed chart bar, blocks (bounded) for a lagging
final intrabar. The loop's clock and the streamer wait/ingest are injected, so
these tests drive every control-flow branch deterministically — no real time,
no ``sleep``. Each test asserts both the returned ``collected`` list and the
advanced cursor, so a cursor-advancement regression is caught too.
"""
import logging
from typing import Callable

from pynecore.types.ohlcv import OHLCV
# noinspection PyProtectedMember
from pynecore.core.security_process import _collect_in_period_intrabars

# ── Geometry ─────────────────────────────────────────────────────────────────
# A 5-minute chart period holding five 60s LTF intrabars: opens at T, T+60s ...
# T+240s; the period ends (exclusive) at T+300s. The final in-period intrabar
# opens at ``final_open = period_end - span = T+240s``.
_T = 1_700_000_000_000          # period start, ms, aligned to the 60s grid
_SPAN_MS = 60_000
_PERIOD_END = _T + 5 * _SPAN_MS  # T + 300s (exclusive)
_GRACE = 15.0


def _bar(open_ms: int) -> OHLCV:
    """An OHLCV at ``open_ms`` (ms); the loop only reads ``timestamp`` (seconds)."""
    return OHLCV(timestamp=open_ms // 1000, open=0.0, high=0.0, low=0.0,
                 close=0.0, volume=0.0)


# The five in-period intrabars and the first next-period bar.
_B = [_bar(_T + i * _SPAN_MS) for i in range(5)]   # opens T .. T+240s
_B_NEXT = _bar(_PERIOD_END)                         # opens T+300s (next period)


class _Clock:
    """Deterministic monotonic clock returning successive programmed ticks.

    Exhausting the programmed ticks raises rather than returning a sentinel, so a
    test that triggers more ``now()`` calls than it accounted for fails loudly
    (catching a now()-call-count regression) instead of silently expiring.
    """

    def __init__(self, ticks: list[float]):
        self._ticks = ticks
        self._i = 0

    def __call__(self) -> float:
        if self._i >= len(self._ticks):
            raise AssertionError(
                "clock exhausted: the loop made more now() calls than programmed"
            )
        v = self._ticks[self._i]
        self._i += 1
        return v


class _Waiter:
    """Deterministic streamer wait: returns one programmed batch per call.

    Records every call and the timeout it was handed; empty once the programmed
    batches run out.
    """

    def __init__(self, batches: list[list[OHLCV]] | None = None):
        self._batches = batches or []
        self._i = 0
        self.calls = 0
        self.timeouts: list[float] = []

    def __call__(self, timeout: float) -> list[OHLCV]:
        self.calls += 1
        self.timeouts.append(timeout)
        if self._i < len(self._batches):
            b = self._batches[self._i]
            self._i += 1
            return b
        return []


def _ingest_into(buf: list[OHLCV]) -> Callable[[OHLCV], None]:
    """An ingest callback appending into ``buf`` (the same list the loop reads)."""
    def _ingest(bar: OHLCV) -> None:
        buf.append(bar)
    return _ingest


def _collect(buf: list[OHLCV], *, chart_developing: bool, clock_ticks: list[float],
             grace: float = _GRACE, waiter: '_Waiter | None' = None):
    """Drive the loop with a fresh clock/waiter and return its full context."""
    clock = _Clock(clock_ticks)
    wait = waiter if waiter is not None else _Waiter()
    collected, next_idx = _collect_in_period_intrabars(
        buf, 0, _PERIOD_END, _SPAN_MS, chart_developing, grace,
        wait, _ingest_into(buf), now=clock,
    )
    return collected, next_idx, wait


def __test_confirmed_full_period_present_returns_without_waiting__():
    """All five in-period intrabars already buffered: collect and return, no wait."""
    buf = list(_B)
    collected, next_idx, wait = _collect(
        buf, chart_developing=False, clock_ticks=[0.0],
    )
    assert collected == _B
    assert next_idx == 5
    assert wait.calls == 0


def __test_developing_returns_partial_without_waiting__():
    """A forming chart bar returns the partial collection at once, even with the
    final intrabar still absent — it never blocks on the grace wait."""
    buf = [_B[0], _B[1], _B[2]]
    collected, next_idx, wait = _collect(
        buf, chart_developing=True, clock_ticks=[0.0],
    )
    assert collected == [_B[0], _B[1], _B[2]]
    assert next_idx == 3
    assert wait.calls == 0


def __test_future_bar_buffered_publishes_gap_without_waiting__(caplog):
    """A next-period bar already sits behind the missing final intrabar: the loop
    detects the gap, warns, and publishes the partial period without waiting."""
    buf = [_B[0], _B[1], _B[2], _B[3], _B_NEXT]   # B[4] (final) missing
    with caplog.at_level(logging.WARNING, logger="pynecore.core.security_process"):
        collected, next_idx, wait = _collect(
            buf, chart_developing=False, clock_ticks=[0.0],
        )
    assert collected == [_B[0], _B[1], _B[2], _B[3]]
    assert next_idx == 4                  # the future bar is left for the next round
    assert buf[next_idx] is _B_NEXT
    assert wait.calls == 0
    assert "missing its final intrabar" in caplog.text


def __test_deadline_expiry_publishes_partial_after_grace__(caplog):
    """The final intrabar never arrives and no later bar is buffered: the loop
    blocks once, the deadline expires, and it warns and publishes the partial."""
    buf = [_B[0], _B[1], _B[2], _B[3]]
    # ticks: deadline=0+15; remaining=15-5=10 (>0 -> wait); remaining=15-20=-5 (expire).
    waiter = _Waiter(batches=[[]])         # one wait, no bar delivered
    with caplog.at_level(logging.WARNING, logger="pynecore.core.security_process"):
        collected, next_idx, wait = _collect(
            buf, chart_developing=False, clock_ticks=[0.0, 5.0, 20.0], waiter=waiter,
        )
    assert collected == [_B[0], _B[1], _B[2], _B[3]]
    assert next_idx == 4
    assert wait.calls == 1
    assert "incomplete after" in caplog.text


def __test_wait_delivers_final_intrabar_then_collects__():
    """The lagging final intrabar arrives via the bounded wait: it is ingested and
    the next pass collects it, completing the full period."""
    buf = [_B[0], _B[1], _B[2], _B[3]]
    waiter = _Waiter(batches=[[_B[4]]])    # the final intrabar arrives on the wait
    collected, next_idx, wait = _collect(
        buf, chart_developing=False, clock_ticks=[0.0, 5.0], waiter=waiter,
    )
    assert collected == _B                 # all five, including the awaited one
    assert next_idx == 5
    assert wait.calls == 1
    assert wait.timeouts == [10.0]         # remaining grace handed to the wait


def __test_zero_grace_times_out_immediately_without_waiting__():
    """``grace_seconds == 0`` with the final intrabar missing: the deadline is
    already reached, so the loop never blocks and publishes the partial at once."""
    buf = [_B[0], _B[1], _B[2], _B[3]]
    # deadline=0+0=0; remaining=0-0=0 (<=0 -> expire), no wait.
    collected, next_idx, wait = _collect(
        buf, chart_developing=False, grace=0.0, clock_ticks=[0.0, 0.0],
    )
    assert collected == [_B[0], _B[1], _B[2], _B[3]]
    assert next_idx == 4
    assert wait.calls == 0


def __test_wait_delivers_next_period_bar_falls_through_to_gap__(caplog):
    """The wait returns a NEXT-period bar instead of the final in-period one: it is
    ingested, the next pass cannot collect it (it is out of period), and the loop
    takes the future-bar gap branch rather than waiting again."""
    buf = [_B[0], _B[1], _B[2], _B[3]]
    waiter = _Waiter(batches=[[_B_NEXT]])  # streamer skipped the final, sent next period
    with caplog.at_level(logging.WARNING, logger="pynecore.core.security_process"):
        collected, next_idx, wait = _collect(
            buf, chart_developing=False, clock_ticks=[0.0, 5.0], waiter=waiter,
        )
    assert collected == [_B[0], _B[1], _B[2], _B[3]]
    assert next_idx == 4
    assert buf[next_idx] is _B_NEXT        # the next-period bar stays buffered
    assert wait.calls == 1
    assert "missing its final intrabar" in caplog.text


# ── Empty-collected confirmed bar — the canonical "all intrabars lag" case ─────
# A confirmed chart bar can reach the wait machinery with NOTHING collected yet
# (the new period's intrabars all lag the WS feed). The `collected and ...` guard
# must short-circuit on the empty list instead of indexing ``collected[-1]``.


def __test_empty_collected_waits_then_collects_full_period__():
    """Confirmed bar, nothing buffered in-period: the loop enters the wait with an
    empty collection and accumulates the lagging intrabars across successive
    waits, completing the full period."""
    buf: list[OHLCV] = []
    # The five intrabars arrive in two late batches.
    waiter = _Waiter(batches=[[_B[0], _B[1]], [_B[2], _B[3], _B[4]]])
    # deadline=0+15; remaining=15-1 (wait); remaining=15-2 (wait); break at final_open.
    collected, next_idx, wait = _collect(
        buf, chart_developing=False, clock_ticks=[0.0, 1.0, 2.0], waiter=waiter,
    )
    assert collected == _B
    assert next_idx == 5
    assert wait.calls == 2


def __test_empty_collected_deadline_expiry_publishes_empty__(caplog):
    """Confirmed bar whose intrabars never arrive: the loop blocks once with an
    empty collection, the deadline expires, and it publishes an empty array
    (the `collected and` guard never indexes the empty list)."""
    buf: list[OHLCV] = []
    waiter = _Waiter(batches=[[]])         # one wait, nothing delivered
    with caplog.at_level(logging.WARNING, logger="pynecore.core.security_process"):
        collected, next_idx, wait = _collect(
            buf, chart_developing=False, clock_ticks=[0.0, 5.0, 20.0], waiter=waiter,
        )
    assert collected == []
    assert next_idx == 0
    assert wait.calls == 1
    assert "incomplete after" in caplog.text


def __test_empty_collected_future_bar_only_publishes_gap__(caplog):
    """Confirmed bar with only a NEXT-period bar buffered: collection is empty, the
    gap branch fires immediately (no wait), and an empty array is published."""
    buf = [_B_NEXT]
    with caplog.at_level(logging.WARNING, logger="pynecore.core.security_process"):
        collected, next_idx, wait = _collect(
            buf, chart_developing=False, clock_ticks=[0.0],
        )
    assert collected == []
    assert next_idx == 0                    # the future bar is left for the next round
    assert buf[next_idx] is _B_NEXT
    assert wait.calls == 0
    assert "missing its final intrabar" in caplog.text


def __test_default_clock_used_when_now_not_injected__():
    """The shipped call site passes no ``now``: the default ``monotonic`` clock is
    bound and an already-complete buffer returns at once (no wait, no real sleep)."""
    buf = list(_B)
    wait = _Waiter()
    collected, next_idx = _collect_in_period_intrabars(
        buf, 0, _PERIOD_END, _SPAN_MS, False, _GRACE,
        wait, _ingest_into(buf),           # no now= -> defaults to time.monotonic
    )
    assert collected == _B
    assert next_idx == 5
    assert wait.calls == 0
