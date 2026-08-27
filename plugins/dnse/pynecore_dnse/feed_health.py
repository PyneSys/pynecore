"""Pure fill-feed health ladder (#54).

A persistently failing order-book poll must not leave ``watch_orders``
permanently silent (measured: 18,936 DEBUG-only polls under a dead
credential). Nothing here does I/O — the watch loop feeds one observation per
book (or crash channel) per cycle, and the ladder answers two questions:
which feed-attributed WARNINGs are due, and whether the designed halt is due.

Panel-adopted rules (card #54):

- **Per-book counters** — a healthy NORMAL book must never mask a permanently
  dead STOP book; only a book's own success resets that book.
- **Warn ladder** — a book warns after ``warn_after`` consecutive failures and
  re-warns every ``rewarn_every`` thereafter (bounded log volume under a
  multi-hour outage).
- **Halt** — only when EVERY real book has been AUTH-classified for
  ``halt_after`` consecutive cycles (~60 s at the 0.5 s cadence): a latched
  halt is irreversible, so a venue auth blip or one weird book must warn, not
  halt. Crash channels (row scan, drain, poll) warn but never halt.
"""
from dataclasses import dataclass, field

#: Channels that represent code-path crashes rather than venue books —
#: they ride the same warn ladder but can never satisfy the halt condition.
CRASH_CHANNELS = ("poll", "scan", "drain")


@dataclass
class _ChannelState:
    consecutive: int = 0
    auth_streak: int = 0
    last_warned_at: int = 0        # value of ``consecutive`` when last warned
    last_kind: str = ""


@dataclass
class FeedHealth:
    """Consecutive-failure ladder over the watch loop's observation channels."""
    warn_after: int
    rewarn_every: int
    halt_after: int
    books: tuple[str, ...]
    _channels: dict = field(default_factory=dict)

    def _state(self, channel: str) -> _ChannelState:
        return self._channels.setdefault(channel, _ChannelState())

    def record_success(self, channel: str) -> None:
        """One healthy observation resets ONLY this channel."""
        self._channels[channel] = _ChannelState()

    def record_failure(self, channel: str, kind: str, *, is_auth: bool = False) -> None:
        state = self._state(channel)
        state.consecutive += 1
        state.auth_streak = state.auth_streak + 1 if is_auth else 0
        state.last_kind = kind

    def warnings_due(self) -> list[str]:
        """Feed-attributed warning lines due THIS cycle (throttled)."""
        due = []
        for channel, state in self._channels.items():
            if state.consecutive < self.warn_after:
                continue
            first_due = state.last_warned_at == 0
            rewarn_due = (state.consecutive - state.last_warned_at
                          >= self.rewarn_every)
            if first_due or rewarn_due:
                state.last_warned_at = state.consecutive
                due.append(
                    f"fill feed degraded: {channel} has failed "
                    f"{state.consecutive} consecutive cycles "
                    f"(last: {state.last_kind}) — fills may be going "
                    f"undetected")
        return due

    def halt_due(self) -> str | None:
        """The designed-halt message, or None.

        Fires only when EVERY real book shows a PERSISTENT AUTH-classified
        failure — a dead credential kills all books identically, while a
        blip, a single-book outage, or any transient never satisfies this.
        """
        if not self.books:
            return None
        streaks = []
        for book in self.books:
            state = self._channels.get(book)
            if state is None or state.auth_streak < self.halt_after:
                return None
            streaks.append(f"{book}={state.auth_streak}")
        return (f"order-book reads AUTH-refused on every book for "
                f"{', '.join(streaks)} consecutive cycles — the credential "
                f"is dead and the fill feed is blind; manual intervention "
                f"required")
