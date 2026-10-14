"""Detect a bot SL/TP someone else moved or removed on the venue.

A bot's protection (stop-loss, take-profit, trailing stop) can change on the
venue for three reasons: the bot amended it, the venue changed it as part of
its own mechanics (a trailing stop following the price), or someone else — the
operator, another bot — edited it. Only the last one is an external
intervention, and it must quarantine the bot instead of being overridden.

The guard keeps, per protection key, the levels the bot itself put on the
venue and judges every venue observation against them:

- **Record before the write.** The plugin calls :meth:`ProtectionLevelGuard.expect`
  before every request that sets the protection, and
  :meth:`ProtectionLevelGuard.settle` once the venue answered. Until then the
  write is in flight and no observation is judged. The previous levels stay
  acceptable until the venue is seen carrying the new ones, so a late push of
  an earlier own level is never mistaken for an edit. Every write is tracked
  on its own, numbered in write order, so a late answer to an older write
  never outranks a newer confirmed one, even when two writes carry the same
  levels. :meth:`~ProtectionLevelGuard.settle` closes the write
  :meth:`~ProtectionLevelGuard.expect` returned (``write=``); without it an
  answer closes the one open write of those levels. While several writes of
  the same levels are open, an answer without ``write=`` cannot tell them
  apart: they all stay unknown and the venue decides, as below.
- **Unconfirmed writes resolve.** A write the venue refused
  (``settle(key, None, rejected=levels)``) stops being acceptable at once. A
  write whose outcome is unknown stays acceptable until the venue decides
  it: seen carrying it, it landed; two consecutive snapshots carrying
  another own level, it did not. When several unresolved writes share the
  levels two consecutive snapshots carry, the newest of them is taken as the
  venue's current protection.
- **Persisted.** The accepted levels live in the extras of a journal row the
  plugin names, so a restart restores them (on first use) and a crash in the
  middle of a write leaves both the old and the new levels acceptable. An
  operator resume (:func:`rebaseline_after_resume`) accepts the venue as the
  operator left it: the first observation after it becomes the new baseline.
- **Tolerant.** Levels match within half a price tick. Under a trailing stop
  the stop level is the venue's to move: only the trailing flag, the trail
  distance and the take-profit are compared.
- **Confirmed.** A snapshot that started before the last own write is
  skipped, and a polled mismatch must show up on two consecutive
  observations. A push the venue sends for this exact change may be judged
  at once (``immediate=True``).
- **Reported once.** A confirmed mismatch — a level moved, a level removed,
  the trailing switched — calls the ``report`` hook once per key, normally
  :meth:`~pynecore.core.plugin.broker.BrokerPlugin.report_external_intervention`.
"""
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pynecore.core.broker.storage import RunContext

__all__ = [
    'PROTECTION_LEVELS_EXTRA',
    'ProtectionLevels',
    'ProtectionLevelGuard',
    'rebaseline_after_resume',
]

logger = logging.getLogger(__name__)

#: Journal-row extras key holding a protection key's accepted levels.
PROTECTION_LEVELS_EXTRA = 'protection_levels'

#: Consecutive mismatching observations a polled verdict needs.
_CONFIRMATIONS = 2


@dataclass(frozen=True, slots=True)
class ProtectionLevels:
    """The protection levels of one key, as the bot set them or the venue shows them.

    ``None`` means the level is absent. For an order-based leg only the field
    the order carries is set (a stop order: ``stop``; a take-profit limit:
    ``limit``).

    :ivar stop: Stop-loss trigger level.
    :ivar limit: Take-profit level.
    :ivar trailing: Whether the stop is a venue-side trailing stop.
    :ivar trail_distance: Trailing distance in price units, when the venue
        reports it.
    """
    stop: float | None = None
    limit: float | None = None
    trailing: bool = False
    trail_distance: float | None = None

    def as_list(self) -> list[Any]:
        """Serialize for the journal extras."""
        return [self.stop, self.limit, self.trailing, self.trail_distance]

    @classmethod
    def from_list(cls, values: list[Any]) -> 'ProtectionLevels':
        """Rebuild from :meth:`as_list` output."""
        stop, limit, trailing, trail_distance = values
        return cls(stop=stop, limit=limit, trailing=bool(trailing),
                   trail_distance=trail_distance)

    def matches(self, venue: 'ProtectionLevels', tolerance: float) -> bool:
        """Whether the venue view carries these levels, within ``tolerance``."""
        if venue.trailing != self.trailing:
            return False
        if not _same_level(self.limit, venue.limit, tolerance):
            return False
        if self.trailing:
            # The venue moves a trailing stop by itself; only its distance is ours.
            if self.trail_distance is None or venue.trail_distance is None:
                return True
            return _same_level(self.trail_distance, venue.trail_distance, tolerance)
        return _same_level(self.stop, venue.stop, tolerance)


def _same_level(ours: float | None, venue: float | None, tolerance: float) -> bool:
    if ours is None or venue is None:
        return ours is None and venue is None
    return abs(ours - venue) <= tolerance


@dataclass(frozen=True, slots=True)
class _Write:
    """One own write of a key's protection; ``gen`` grows in write order."""
    gen: int
    levels: ProtectionLevels


@dataclass
class _Entry:
    row_coid: str | None
    writes: list[_Write] = field(default_factory=list)
    """The acceptable own writes, oldest first."""
    confirmed: int | None = None
    """``gen`` of the newest write known to have landed."""
    unresolved: list[int] = field(default_factory=list)
    """``gen`` of every write whose outcome is not known yet."""
    next_gen: int = 0
    resolve_count: int = 0
    resolve_on: tuple[int, ...] = ()
    """The unresolved writes the snapshots ``resolve_count`` counts carry."""
    written_at: float = 0.0
    in_flight: int = 0
    mismatch: ProtectionLevels | None = None
    mismatch_count: int = 0
    reported: bool = False
    rebaseline: bool = False

    def add(self, levels: ProtectionLevels) -> int:
        """Append a new write of ``levels`` and return its ``gen``."""
        gen = self.next_gen
        self.next_gen += 1
        self.writes.append(_Write(gen, levels))
        return gen

    def get(self, gen: int | None) -> _Write | None:
        for write in self.writes:
            if write.gen == gen:
                return write
        return None

    def confirmed_levels(self) -> ProtectionLevels | None:
        write = self.get(self.confirmed)
        return write.levels if write is not None else None

    def promote(self, gen: int) -> None:
        """Make write ``gen`` the confirmed one unless a later-written one already is."""
        if self.confirmed is None or self.get(self.confirmed) is None or gen >= self.confirmed:
            self.confirmed = gen

    def open_writes(self, levels: ProtectionLevels) -> list[int]:
        """``gen`` of every unresolved write of exactly ``levels``."""
        return [g for g in self.unresolved if (w := self.get(g)) is not None and w.levels == levels]


class ProtectionLevelGuard:
    """Judges venue protection levels against the levels the bot set.

    One instance per plugin, used from the plugin's event-loop thread.

    :param report: Called once per key with ``(reason, context)`` when an
        external edit is confirmed. Normally the plugin's
        :meth:`~pynecore.core.plugin.broker.BrokerPlugin.report_external_intervention`;
        whatever it raises propagates to the caller of :meth:`observe`.
    :param store_ctx: The run's journal; ``None`` keeps the levels in memory only.
    :param clock: Wall clock in seconds, comparable with the ``read_started_at``
        the plugin passes to :meth:`observe`.
    """

    def __init__(
            self,
            report: Callable[[str, dict], None],
            *,
            store_ctx: 'RunContext | None' = None,
            clock: Callable[[], float] = time.time,
    ) -> None:
        self._report = report
        self._store = store_ctx
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._restored = False

    def _restore(self) -> None:
        """Reload the accepted levels persisted on the live journal rows, once.

        Runs on first use rather than at construction, so an operator resume
        the engine applies at startup has already re-flagged the rows.
        """
        if self._restored:
            return
        self._restored = True
        if self._store is None:
            return
        for row in self._store.iter_live_orders():
            saved = (row.extras or {}).get(PROTECTION_LEVELS_EXTRA)
            if not isinstance(saved, dict) or not isinstance(saved.get('key'), str):
                continue
            try:
                writes = [_Write(int(gen), ProtectionLevels.from_list(levels))
                          for gen, levels in saved.get('writes') or []]
                confirmed_raw = saved.get('confirmed')
                confirmed = int(confirmed_raw) if confirmed_raw is not None else None
                unresolved = [int(gen) for gen in saved.get('unresolved') or []]
                next_gen = int(saved.get('next_gen') or 0)
            except (TypeError, ValueError):
                logger.warning("unreadable protection levels on row %r", row.client_order_id)
                continue
            self._entries.setdefault(saved['key'], _Entry(
                row_coid=row.client_order_id, writes=writes, confirmed=confirmed,
                unresolved=unresolved,
                next_gen=max([next_gen, *(w.gen + 1 for w in writes)]),
                written_at=float(saved.get('written_at') or 0.0),
                rebaseline=bool(saved.get('rebaseline')),
            ))

    def known(self, key: str) -> bool:
        """Whether the guard holds levels for ``key``."""
        self._restore()
        return key in self._entries

    def expect(self, key: str, levels: ProtectionLevels, *, row_coid: str | None = None) -> int:
        """Record levels the bot is about to send; call before the request.

        The key counts as in flight until the matching :meth:`settle`.

        :param key: The plugin's protection key (a leg row, a position, ...).
        :param levels: The complete protection the request sets.
        :param row_coid: Journal row the levels are persisted on; defaults to
            the row named earlier for this key. Naming another row moves the
            levels there.
        :return: The write's id, for :meth:`settle`'s ``write``.
        """
        self._restore()
        entry = self._entry_on_row(key, row_coid)
        entry.rebaseline = False
        gen = entry.add(levels)
        entry.unresolved.append(gen)
        entry.resolve_count = 0
        entry.in_flight += 1
        entry.written_at = self._clock()
        entry.mismatch = None
        entry.mismatch_count = 0
        self._persist(key, entry)
        return gen

    def adopt(self, key: str, levels: ProtectionLevels, *, row_coid: str | None = None) -> None:
        """Accept ``levels`` as the bot's own, replacing everything recorded.

        For a change the bot caused without writing the protection itself:
        its own fill resized or reversed the position and the venue adjusted
        the protection with it.

        :param key: The protection key.
        :param levels: What the venue carries now.
        :param row_coid: As for :meth:`expect`.
        """
        self._restore()
        entry = self._entry_on_row(key, row_coid)
        entry.writes = []
        entry.confirmed = entry.add(levels)
        entry.unresolved = []
        entry.resolve_count = 0
        entry.rebaseline = False
        entry.mismatch = None
        entry.mismatch_count = 0
        self._persist(key, entry)

    def settle(
            self,
            key: str,
            confirmed: ProtectionLevels | None,
            *,
            rejected: ProtectionLevels | None = None,
            write: int | None = None,
    ) -> None:
        """Close one write recorded by :meth:`expect`.

        :param key: The protection key.
        :param confirmed: The levels the venue accepted. ``None`` when the
            write was rejected or its outcome is unknown: an unknown write stays
            acceptable until a venue observation resolves it.
        :param rejected: With ``confirmed=None``: the levels of a write the
            venue definitively refused. That write stops being acceptable at
            once; another write of the same levels is not affected.
        :param write: The id :meth:`expect` returned for this write. Without
            it the answer closes the one unresolved write of those levels; when
            several are unresolved, which one it answers is unknown, so all of
            them stay unresolved until a venue observation decides.
        """
        self._restore()
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.in_flight = max(0, entry.in_flight - 1)
        entry.written_at = self._clock()
        if confirmed is not None:
            open_gens = [write] if entry.get(write) is not None else entry.open_writes(confirmed)
            if not open_gens:
                entry.promote(entry.add(confirmed))
            elif len(open_gens) == 1:
                gen = open_gens[0]
                if gen in entry.unresolved:
                    entry.unresolved.remove(gen)
                # Keep the levels the venue reports for this write
                entry.writes = [_Write(gen, confirmed) if w.gen == gen else w
                                for w in entry.writes]
                entry.promote(gen)
        elif rejected is not None:
            open_gens = [write] if entry.get(write) is not None else entry.open_writes(rejected)
            gen = open_gens[0] if len(open_gens) == 1 else None
            if gen is not None and gen != entry.confirmed:
                if gen in entry.unresolved:
                    entry.unresolved.remove(gen)
                entry.writes = [w for w in entry.writes if w.gen != gen]
            if not entry.writes:
                # The refused write was the key's first: the guard knows no
                # protection of the bot's on the venue.
                self.forget(key)
                return
        self._persist(key, entry)

    def forget(self, key: str) -> None:
        """Stop guarding ``key``: its protection or position is gone."""
        self._restore()
        entry = self._entries.pop(key, None)
        if entry is not None and entry.row_coid is not None and self._store is not None:
            self._write_extra(entry.row_coid, None)

    def observe(
            self,
            key: str,
            venue: ProtectionLevels,
            *,
            tick: float,
            read_started_at: float | None = None,
            immediate: bool = False,
    ) -> bool:
        """Judge one venue observation of ``key``'s protection.

        :param key: The protection key.
        :param venue: What the venue carries now (``None`` fields: absent).
        :param tick: The instrument's price tick; levels match within half of it.
        :param read_started_at: Wall-clock time the snapshot read started, so
            a snapshot older than the last own write is skipped. ``None`` for
            a push.
        :param immediate: Report on this observation alone (a push for this
            exact change) instead of waiting for a second one.
        :return: ``True`` when this observation reported an external edit.
        """
        self._restore()
        entry = self._entries.get(key)
        if entry is None or entry.reported or entry.in_flight > 0:
            return False
        if read_started_at is not None and read_started_at < entry.written_at:
            return False
        if entry.rebaseline:
            # The operator resumed the run: what the venue carries now is the
            # accepted state until the bot writes its own levels again.
            entry.rebaseline = False
            entry.writes = []
            entry.confirmed = entry.add(venue)
            entry.unresolved = []
            entry.resolve_count = 0
            self._persist(key, entry)
            return False
        tolerance = 0.5 * tick if tick > 0.0 and math.isfinite(tick) else 0.0
        tolerance += 1e-12 * max(1.0, abs(venue.stop or 0.0), abs(venue.limit or 0.0))
        matching = [w.gen for w in entry.writes if w.levels.matches(venue, tolerance)]
        if matching:
            entry.mismatch = None
            entry.mismatch_count = 0
            if self._resolve(entry, matching, snapshot=read_started_at is not None):
                self._persist(key, entry)
            return False
        entry.resolve_count = 0
        if entry.mismatch is not None and entry.mismatch.matches(venue, tolerance):
            entry.mismatch_count += 1
        else:
            entry.mismatch = venue
            entry.mismatch_count = 1
        if not immediate and entry.mismatch_count < _CONFIRMATIONS:
            return False
        entry.reported = True
        expected = entry.confirmed_levels() or entry.writes[-1].levels
        reason = (
            f"bot protection {key!r} changed on the venue by someone else: "
            f"expected {_describe(expected)}, found {_describe(venue)}"
        )
        self._report(reason, {
            'protection_key': key,
            'expected': expected.as_list(),
            'observed': venue.as_list(),
            'client_order_id': entry.row_coid,
        })
        return True

    @staticmethod
    def _resolve(entry: _Entry, matching: list[int], *, snapshot: bool) -> bool:
        """Update ``entry`` for a venue view that carries the writes ``matching``.

        No write is in flight here, so every unresolved write is one whose
        outcome is unknown. Seen on the venue, it landed; when several match,
        only the oldest is taken as landed, so a write is never ranked above
        one it may have trailed, and the newer ones stay acceptable until a
        later observation decides them. A view an older landed write already
        explains proves nothing about a newer write of the same levels, so a
        single or pushed repeat never resolves another write.

        Two consecutive snapshots settle what the venue carries now: none of
        the unresolved writes of other levels landed, and when the view is one
        only the unresolved writes of its levels explain, the venue carries
        the newest of them, whichever one actually landed. A push is not
        enough, it may predate a write that landed late.

        :param matching: ``gen`` of every accepted write the venue view matches.
        :return: Whether the entry changed.
        """
        changed = False
        candidates = [gen for gen in matching if gen in entry.unresolved]
        explained_up_to = min((gen for gen in matching if gen not in entry.unresolved),
                              default=None)
        landed = [gen for gen in candidates if explained_up_to is None or gen < explained_up_to]
        if landed:
            entry.unresolved.remove(min(landed))
            entry.resolve_count = 0
            entry.promote(min(landed))
            changed = True
        elif entry.unresolved and snapshot:
            carried = tuple(candidates)
            if entry.resolve_count == 0 or entry.resolve_on != carried:
                entry.resolve_on = carried
                entry.resolve_count = 1
            else:
                entry.resolve_count += 1
            if entry.resolve_count >= _CONFIRMATIONS:
                if candidates:
                    entry.promote(max(candidates))
                dropped = [gen for gen in entry.unresolved if gen not in candidates]
                entry.writes = [w for w in entry.writes
                                if w.gen not in dropped or w.gen == entry.confirmed]
                entry.unresolved = []
                entry.resolve_count = 0
                changed = True
        confirmed = entry.confirmed
        if confirmed is not None and confirmed in matching and entry.writes[0].gen < confirmed:
            # The venue carries the confirmed write: the writes before it are
            # history. Later, unresolved ones stay.
            entry.writes = [w for w in entry.writes if w.gen >= confirmed]
            entry.unresolved = [gen for gen in entry.unresolved if gen > confirmed]
            changed = True
        return changed

    def _entry_on_row(self, key: str, row_coid: str | None) -> _Entry:
        """The key's entry, created or moved onto ``row_coid`` when one is named."""
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(row_coid=row_coid)
            self._entries[key] = entry
        elif row_coid is not None and row_coid != entry.row_coid:
            if entry.row_coid is not None and self._store is not None:
                self._write_extra(entry.row_coid, None)
            entry.row_coid = row_coid
        return entry

    def _persist(self, key: str, entry: _Entry) -> None:
        if entry.row_coid is None or self._store is None:
            return
        self._write_extra(entry.row_coid, {
            'key': key,
            'writes': [[w.gen, w.levels.as_list()] for w in entry.writes],
            'confirmed': entry.confirmed,
            'unresolved': list(entry.unresolved),
            'next_gen': entry.next_gen,
            'written_at': entry.written_at,
            'rebaseline': entry.rebaseline,
        })

    def _write_extra(self, coid: str, value: dict | None) -> None:
        assert self._store is not None
        with self._store.transaction():
            row = self._store.get_order(coid)
            if row is None or row.closed_ts_ms is not None:
                return
            extras = dict(row.extras or {})
            if value is None:
                if PROTECTION_LEVELS_EXTRA not in extras:
                    return
                del extras[PROTECTION_LEVELS_EXTRA]
            else:
                extras[PROTECTION_LEVELS_EXTRA] = value
            self._store.upsert_order(coid, extras=extras)


def rebaseline_after_resume(store_ctx: 'RunContext') -> None:
    """Make every persisted protection baseline adopt the venue's next view.

    An operator resume accepts the venue as the operator left it. Without this
    the restarted bot would judge the operator's levels against its own
    pre-quarantine ones and quarantine again at once.

    :param store_ctx: The resumed run's journal.
    """
    with store_ctx.transaction():
        for row in store_ctx.iter_live_orders():
            saved = (row.extras or {}).get(PROTECTION_LEVELS_EXTRA)
            if not isinstance(saved, dict) or saved.get('rebaseline'):
                continue
            extras = dict(row.extras or {})
            extras[PROTECTION_LEVELS_EXTRA] = {**saved, 'rebaseline': True}
            store_ctx.upsert_order(row.client_order_id, extras=extras)


def _describe(levels: ProtectionLevels) -> str:
    stop = (f"trailing stop (distance {levels.trail_distance})" if levels.trailing
            else f"stop {levels.stop}")
    return f"{stop}, take-profit {levels.limit}"
