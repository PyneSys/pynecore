"""Pure page-completeness decisions (#62/#57/#61).

A truncated page must never read as flat/complete: ``get_position`` returning
None arms the engine's external-flatten wipe, and a partial order-book union
false-cleans the L0 gate. Nothing here does I/O — the broker feeds envelope
metadata in, these functions answer "is this page set PROVEN complete?".

Guards pinned by ``tests/test_page_completeness.py``:

- **G1**: raise on PROVEN truncation only — an absent ``total`` (the field is
  documented derivatives-only, so STOCK never has it) is NOT evidence of
  truncation; inferring from missing metadata would halt healthy runs
  (round-2 item-6 correction, operator-merged).
- **G2b**: completeness is judged on the RAW row count, before any status
  filter — ``total`` counts CLOSED rows too.
- **G5'**: an over-cap or unparseable ``totalPages`` makes the BOOK
  unprovable (None) — the caller must treat it as unreadable, never return a
  partial drain as complete.
"""

#: Positions page size requested by ``get_position`` — far above any real
#: account's position count, so the single documented request knob does the
#: heavy lifting even where ``total`` is absent (STOCK).
POSITIONS_PAGE_SIZE = 500

#: Order-book drain cap. The venue's own ``totalPages`` drives the drain, but
#: a lying value must not authorize a request storm (measured budget: 100k/h
#: orders bucket vs the 0.5 s watch cadence) — over-cap means unprovable.
MAX_BOOK_PAGES = 10

#: Wall-clock deadline for one off-loop book/positions read. Under the
#: engine's ~30 s execute budget; the client socket alone may block 60 s.
BOOK_READ_DEADLINE_S = 25.0


def positions_complete(raw_row_count: int, total: object) -> bool:
    """G1/G2b: is the delivered RAW row set provably complete against
    ``total``? Absent or unparseable metadata is never proof of truncation."""
    if total is None:
        return True
    try:
        return raw_row_count >= int(total)
    except (TypeError, ValueError):
        return True


def is_exposure_row(row: dict) -> bool:
    """Only ``CLOSED`` is history. OPEN / PENDING_CLOSE / ODD_LOT all carry
    real exposure and must net (filtering ODD_LOT would re-import wrong-flat
    for STOCK accounts)."""
    return str(row.get("status") or "").upper() != "CLOSED"


def book_page_count(total_pages_field: object,
                    cap: int = MAX_BOOK_PAGES) -> int | None:
    """How many pages to drain for one book — or None when unprovable (G5').

    Absent field means the venue answered a single unpaged list (drain = 1);
    an unparseable or over-``cap`` value means completeness cannot be proven
    within budget, and the whole book must count as unreadable.
    """
    if total_pages_field is None:
        return 1
    try:
        pages = int(total_pages_field)
    except (TypeError, ValueError):
        return None
    if pages < 1:
        return 1
    if pages > cap:
        return None
    return pages
