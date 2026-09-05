"""Pure external-cancel RESIDUE tracker (#74, Phase B3 — #69 riding).

The venue reports an external cancel as a STATUS TRANSITION on a row the day
book retains — that channel is the PRIMARY and already works live. This
module owns only the residue: an id that vanished from every READABLE book
without a terminal transition ever observed. Contract (card #74 panel,
leader-adjudicated):

- an absence clock starts only on a cycle where ALL books were readable
  (rows|None, #54/#62) — a failed read must degrade to non-detection,
  never to a false cancel;
- after ``grace_s`` (>= 30 s flat in production — 3x the measured ~10 s
  stale-replica lag, NEVER a cadence formula) the id is due for ONE
  confirm; further confirms wait out a cooldown, so a stuck residue costs
  one rate-limited history call per cooldown, not one per 0.5 s cycle;
- the tracker never concludes anything itself: the caller runs the confirm
  (positive, ``_classify_readback``-graded ``/orders/history`` row ->
  CANCELLED; everything else INCONCLUSIVE) and reports back;
- INCONCLUSIVE keeps the stamp forever and gets LOUD after
  ``warn_after`` consecutive inconclusive confirms (throttled: once per
  threshold crossing) — silent non-detection is the #54 lesson;
- the POPULATION is the caller's job and encodes the exclusions: exposure
  rows (``filled_qty>0`` / terminal extras — the #73 keep-live ledger) and
  #41 Activated shells (tracked by their CHILD ref, the shell id never)
  must not appear in ``tracked_ids``.

Pure state, injected clock; no I/O.
"""
from dataclasses import dataclass, field


@dataclass
class _ResidueState:
    missing_since: float
    last_confirm_at: "float | None" = None
    inconclusive_count: int = 0


@dataclass
class ResidueTracker:
    grace_s: float = 30.0
    confirm_cooldown_s: "float | None" = None   # default: grace_s
    warn_after: int = 3
    _states: "dict[str, _ResidueState]" = field(default_factory=dict)

    def _cooldown(self) -> float:
        return (self.confirm_cooldown_s if self.confirm_cooldown_s is not None
                else self.grace_s)

    def observe(self, *, tracked_ids, present_ids, all_books_readable,
                now: float) -> "str | None":
        """One watch cycle. Returns AT MOST ONE id due for a confirm.

        Presence clears the stamp unconditionally; absence stamps only on a
        fully-readable cycle. Ids that left ``tracked_ids`` (terminal seen,
        exposure row, closed) are dropped entirely.
        """
        for order_id in list(self._states):
            if order_id not in tracked_ids or order_id in present_ids:
                del self._states[order_id]
        if not all_books_readable:
            return None                 # absence unprovable: neither stamp nor age
        due: "str | None" = None
        for order_id in tracked_ids:
            if order_id in present_ids:
                continue
            state = self._states.get(order_id)
            if state is None:
                self._states[order_id] = _ResidueState(missing_since=now)
                continue
            if due is not None:
                continue                # one confirm per cycle (budget guard)
            if (now - state.missing_since) < self.grace_s:
                continue
            if (state.last_confirm_at is not None
                    and (now - state.last_confirm_at) < self._cooldown()):
                continue
            due = order_id
        if due is not None:
            self._states[due].last_confirm_at = now
        return due

    def record_confirm(self, order_id: str, *, concluded: bool
                       ) -> "str | None":
        """Report a confirm outcome; returns a WARNING line when the
        inconclusive streak crosses ``warn_after`` (then every multiple)."""
        state = self._states.get(order_id)
        if state is None:
            return None
        if concluded:
            del self._states[order_id]
            return None
        state.inconclusive_count += 1
        count = state.inconclusive_count
        if count >= self.warn_after and count % self.warn_after == 0:
            return (
                f"residue: order {order_id} has been missing from every "
                f"readable book for {count} inconclusive confirms — no "
                f"positive /orders/history record either way. Touching "
                f"NOTHING (absence never concludes). Operator: check "
                f"`venue.py order {order_id}` if this persists.")
        return None
