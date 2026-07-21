"""Deterministic virtual-time scheduling for offline broker scenarios."""

from collections.abc import Callable
from dataclasses import dataclass, field
from heapq import heappop, heappush


@dataclass(order=True)
class ScheduledEvent:
    """One callback ordered by virtual deadline and insertion sequence."""

    deadline_ms: int
    sequence: int
    callback: Callable[[], None] = field(compare=False)


class DeterministicScheduler:
    """Advance callbacks without sleeping or consulting wall-clock time."""

    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        self.now_ms = start_ms
        self._sequence = 0
        self._events: list[ScheduledEvent] = []

    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> int:
        if delay_ms < 0:
            raise ValueError("delay_ms must not be negative")
        self._sequence += 1
        heappush(
            self._events,
            ScheduledEvent(self.now_ms + delay_ms, self._sequence, callback),
        )
        return self._sequence

    def advance(self, milliseconds: int) -> None:
        if milliseconds < 0:
            raise ValueError("milliseconds must not be negative")
        target = self.now_ms + milliseconds
        while self._events and self._events[0].deadline_ms <= target:
            event = heappop(self._events)
            self.now_ms = event.deadline_ms
            event.callback()
        self.now_ms = target

    def run_ready(self) -> None:
        self.advance(0)

    @property
    def pending(self) -> int:
        return len(self._events)
