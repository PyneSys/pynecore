"""
Append-only JSONL state store for cross-restart broker recovery.

The :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine` keeps two pieces
of state that are required for the canonical ``client_order_id`` scheme to
survive a process restart:

- ``_envelopes`` — the **first** :class:`DispatchEnvelope` ever built for a
  given ``intent_key``. The sync engine pins ``bar_ts_ms`` and ``retry_seq``
  on this envelope and re-uses them on every modify, which is what gives the
  exchange a stable id to dedup against. Lose this map and a post-restart
  amend re-emits a *different* ``client_order_id`` — the exchange treats the
  modify as a brand-new order.
- ``_pending_verification`` — envelopes whose dispatch raised
  :class:`OrderDispositionUnknownError`. The next sync calls ``get_open_orders``
  and matches by ``client_order_id``; lose the parked envelope and the engine
  cannot tell which intent the exchange-side order belongs to.

The store is a single append-only JSONL file. Every mutation is one short line,
fsync'd by the OS in the usual way; a torn last line is silently dropped on
replay (the prior lines remain valid). The replayer reduces the event log into
the same pair of dicts above.

JSON schema (one object per line)::

    {"op": "envelope", "key": "Long",         "bar_ts_ms": 1700000000000, "retry_seq": 0}
    {"op": "park",     "key": "Long",         "coid": "abcd-...-e0"}
    {"op": "unpark",   "coid": "abcd-...-e0"}
    {"op": "complete", "key": "Long"}

``complete`` removes both the envelope and any park entry for that key — the
sync engine emits it whenever an intent is cancelled or the position closes.
The file is therefore self-compacting: replay only retains entries whose
``complete`` has not yet arrived.

The store is **not** a transaction log of every dispatch — only the envelope
*identity* (``bar_ts_ms``, ``retry_seq``) and the parked-verification queue.
The actual exchange-side order list is recovered on the next sync via
``get_open_orders`` matching, exactly as the in-process recovery path does.
"""
from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO

__all__ = [
    'StateStore',
    'EnvelopeRecord',
    'PendingRecord',
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvelopeRecord:
    """Replay output for a single live envelope.

    The sync engine reconstructs a :class:`DispatchEnvelope` from this by
    pairing it with the freshly-built intent (the intent itself is *not*
    persisted — it is always rebuilt from the Pine order book on the first
    sync after restart).
    """
    key: str
    bar_ts_ms: int
    retry_seq: int


@dataclass(frozen=True)
class PendingRecord:
    """Replay output for a single parked verification.

    ``key`` lets the sync engine route the recovered exchange order back into
    ``_order_mapping`` once :meth:`_verify_pending_dispatches` matches the
    ``client_order_id``.
    """
    key: str
    coid: str


class StateStore:
    """Append-only JSONL persistence for sync-engine envelopes.

    The store opens its file in line-buffered append mode on construction and
    keeps the handle for the lifetime of the engine. Each ``record_*`` method
    writes a single short JSON line. The OS buffer flush boundary is the line
    terminator — a process kill mid-write at worst loses the trailing line,
    and :meth:`replay` skips a malformed final entry rather than aborting.

    :param path: JSONL file path. The parent directory is created on demand.
    :param fsync_each_write: When ``True`` (the live-trading default), call
        ``os.fsync`` after every record so the kernel page cache cannot lose
        a write across a host crash. Off for unit tests where the cost is
        not justified — the line buffer alone is sufficient against process
        crashes (only host crashes need fsync).
    """

    def __init__(self, path: Path | str, *, fsync_each_write: bool = False) -> None:
        self._path = Path(path)
        self._fsync = fsync_each_write
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append. Opening in append mode is atomic on POSIX with
        # respect to concurrent appenders (a single process is the realistic
        # case here, but the guarantee removes a class of foot-guns).
        self._fp: IO[str] = open(self._path, 'a', buffering=1, encoding='utf-8')

    # === Lifecycle ===

    def close(self) -> None:
        """Flush and close the underlying file handle. Safe to call twice."""
        if self._fp is None:
            return
        try:
            self._fp.flush()
            try:
                os.fsync(self._fp.fileno())
            except (OSError, ValueError):  # pragma: no cover — best effort
                pass
        finally:
            self._fp.close()
            self._fp = None  # type: ignore[assignment]

    def __enter__(self) -> 'StateStore':
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # === Writers ===

    def record_envelope(self, key: str, bar_ts_ms: int, retry_seq: int) -> None:
        """Persist the first envelope for an ``intent_key``.

        Subsequent modifies do **not** call this — the sync engine pins the
        envelope on first build, so the persisted ``bar_ts_ms`` and
        ``retry_seq`` already match every later dispatch.
        """
        self._append({
            'op': 'envelope',
            'key': key,
            'bar_ts_ms': bar_ts_ms,
            'retry_seq': retry_seq,
        })

    def record_park(self, coid: str, key: str) -> None:
        """Persist a parked dispatch awaiting verification."""
        self._append({'op': 'park', 'coid': coid, 'key': key})

    def record_unpark(self, coid: str) -> None:
        """Persist that a parked dispatch was matched to an exchange order."""
        self._append({'op': 'unpark', 'coid': coid})

    def record_complete(self, key: str) -> None:
        """Persist that an envelope is no longer needed (cancelled / closed).

        Replay treats ``complete`` as the terminator for the whole ``key``:
        any earlier ``envelope`` and any still-open ``park`` for the key are
        dropped. This is what keeps the file self-compacting under steady-state
        churn.
        """
        self._append({'op': 'complete', 'key': key})

    def _append(self, payload: dict) -> None:
        line = json.dumps(payload, separators=(',', ':'))
        # The newline is what the line buffer flushes on; write+newline together
        # to keep the boundary atomic for the buffer.
        self._fp.write(line + '\n')
        if self._fsync:
            try:
                os.fsync(self._fp.fileno())
            except (OSError, ValueError):  # pragma: no cover — best effort
                pass

    # === Replay ===

    def replay(self) -> tuple[dict[str, EnvelopeRecord], dict[str, PendingRecord]]:
        """Read the JSONL file and reduce to the live state.

        :returns: ``(envelopes_by_key, pending_by_coid)``. ``envelopes_by_key``
            holds every key whose ``complete`` has not yet been recorded;
            ``pending_by_coid`` holds every parked dispatch whose ``unpark``
            (or whose key's ``complete``) has not yet been recorded.
        """
        envelopes: dict[str, EnvelopeRecord] = {}
        pending: dict[str, PendingRecord] = {}
        coid_to_key: dict[str, str] = {}
        if not self._path.exists():
            return envelopes, pending
        # Read from a fresh handle — the writer's buffer may not yet have hit
        # the disk, but that's fine: replay only runs at startup, before any
        # writes happen on this engine instance.
        with open(self._path, 'r', encoding='utf-8') as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.rstrip('\n')
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    # Torn final line — log and stop. Anything after a torn
                    # line is suspect, but the engine's behaviour with a
                    # truncated tail is the same as if the truncated events
                    # had never been recorded (they will simply replay as
                    # in-memory state on the new run).
                    _log.warning(
                        'broker state store: dropping malformed line %d in %s',
                        lineno, self._path,
                    )
                    break
                op = rec.get('op')
                if op == 'envelope':
                    key = rec['key']
                    envelopes[key] = EnvelopeRecord(
                        key=key,
                        bar_ts_ms=int(rec['bar_ts_ms']),
                        retry_seq=int(rec['retry_seq']),
                    )
                elif op == 'park':
                    coid = rec['coid']
                    key = rec['key']
                    pending[coid] = PendingRecord(key=key, coid=coid)
                    coid_to_key[coid] = key
                elif op == 'unpark':
                    coid = rec['coid']
                    pending.pop(coid, None)
                    coid_to_key.pop(coid, None)
                elif op == 'complete':
                    key = rec['key']
                    envelopes.pop(key, None)
                    # Drop any pending verifications still attached to this key.
                    stale = [c for c, k in coid_to_key.items() if k == key]
                    for c in stale:
                        pending.pop(c, None)
                        coid_to_key.pop(c, None)
                else:
                    _log.warning(
                        'broker state store: unknown op %r at line %d in %s',
                        op, lineno, self._path,
                    )
        return envelopes, pending

    # === Inspection helpers ===

    @property
    def path(self) -> Path:
        return self._path
