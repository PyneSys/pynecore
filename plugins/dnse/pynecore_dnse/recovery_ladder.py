"""Pure recovery verdict ladder (#71, Phase B1 — #60 riding).

Startup recovery POLICY over #36's journal rows. Nothing here does I/O — the
broker feeds evidence in, the ladder answers with verdicts. Two rules own
everything:

- **Still-unknown touches nothing** (#59 item-2 SC2): on a SOFTWARE-idempotency
  venue, doubt plus any venue write is how a double order is born. No verdict
  here ever authorises a write.
- **Report, never adopt, never cancel** what is not provably ours: on this
  shared netting account an unclaimed book row is BY DEFINITION the
  operator's.

**Why there is no matcher** (the #71 panel killed it three independent ways —
leave this epitaph in place):

1. ``foreign_live_exchange_order_ids`` is a *database* filter; the operator
   has no rows anywhere, so the "unclaimed" complement contains every
   operator order — a never-landed lost-reply row single-matches the
   operator's identical LO.
2. "No run owns it" is *anti*-evidence on a shared account.
3. A single-match verdict would ride provably incomplete reads (single-page
   history — #69; skip-on-unreadable books) — truncation turns multi-match
   into false single-match.

``match_candidates`` therefore returns nothing until real evidence exists
(persisted price — being collected now — and venue-time, which today is
overwritten by the local clock). Completeness is an INPUT: incomplete
evidence can only ever widen doubt, never resolve it.

Known blind spot (panel P2+P3, convergent): the core strand helper requires a
non-empty exchange id, so a SIBLING run's lost-reply strands (id-less DU
rows) are structurally invisible to the strand report — tracked as a
follow-up card, not silently claimed as covered.
"""
from dataclasses import dataclass


@dataclass
class RecoveryVerdict:
    kind: str                # 'still_unknown' | 'strand'
    subject: str             # coid (still_unknown) or venue id (strand)
    pine_id: "str | None"
    message: str             # the operator-facing WARNING line


def match_candidates(_du_row, _book_rows, _evidence_complete: bool) -> None:
    """Deliberately returns nothing — see the module docstring's epitaph.

    A future matcher may act ONLY on positive per-order evidence (persisted
    price + venue timestamps) AND only with ``_evidence_complete`` true.
    """
    return None


def classify_recovery(*, du_rows, strand_ids, evidence_complete: bool
                      ) -> "list[RecoveryVerdict]":
    """Verdicts for one startup pass.

    :param du_rows: this run's live ``disposition_unknown`` journal rows
        (each carries ``client_order_id``/``pine_entry_id`` and extras with
        the #67 transport phase).
    :param strand_ids: venue order ids owned by SIBLING runs on this account
        (core's ``foreign_live_exchange_order_ids`` — id-bearing rows only).
    :param evidence_complete: False when any evidence read was
        unreadable/unprovable — widens doubt, never narrows it.
    """
    verdicts: list[RecoveryVerdict] = []
    for row in du_rows:
        extras = row.extras or {}
        phase = extras.get("transport_phase", "unknown")
        completeness_note = ("" if evidence_complete else
                             " (evidence reads INCOMPLETE this pass)")
        verdicts.append(RecoveryVerdict(
            kind="still_unknown",
            subject=row.client_order_id,
            pine_id=row.pine_entry_id or None,
            message=(
                f"recovery: pine id {row.pine_entry_id!r} has an UNRESOLVED "
                f"lost-reply order (transport phase={phase}) — it MAY exist "
                f"at the venue{completeness_note}. Touching NOTHING (doubt "
                f"never writes). Operator: check `venue.py status` / "
                f"`venue.py history` before placing anything by hand."),
        ))
    for venue_id in strand_ids:
        verdicts.append(RecoveryVerdict(
            kind="strand",
            subject=str(venue_id),
            pine_id=None,
            message=(
                f"recovery: venue order {venue_id} is owned by ANOTHER run "
                f"label on this account (stranded by a different-label "
                f"relaunch, #60). Reported only — never adopted, never "
                f"cancelled. Operator: resume that label, or manage the "
                f"order via `venue.py order {venue_id}` / your app."),
        ))
    return verdicts
