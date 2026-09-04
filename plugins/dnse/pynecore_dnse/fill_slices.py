"""Pure fill-slice selection with the budget clamp (#56 / item 5).

The venue's ``/executions/{orderId}`` ``reports`` list is an UNORDERED
per-status-update lifecycle stream — the documented sample
(dnse-get-executions.md) carries qty-0 rows (PendingNew/New), a literally
duplicated eventNo, and arrives out of order; ``metadata`` is a JSON STRING
whose ``eventNo`` shows up as float or int. Nothing here does I/O.

The single dedup truth is the **booked-cumulative budget clamp** (#56 panel,
unanimous): only quantity that fits ``venue_cum - booked_cum`` is ever
emitted, every emission advances the watermark, an under-coverage residue
emits ONE average-priced remainder that ALSO advances it (so a late-arriving
execution row lands at-or-below the watermark and is discarded — the
remainder double-count P1 named), and over-coverage clamps the last slice.
Quantity is conserved on every path.
"""
import json
from dataclasses import dataclass


@dataclass
class FillSlice:
    cumulative: float        # per-report cumulative fillQuantity (the selector)
    qty: float               # this slice's lastQuantity
    price: float             # this slice's lastPrice
    event_no: "int | None"   # diagnostics only — repeats in the documented sample


def parse_reports(body: object) -> "list[FillSlice]":
    """Documented-hostile parse: qty-0 lifecycle rows filtered, literal
    duplicates dropped, sorted by per-report cumulative (monotonic by
    construction — eventNo is NOT, per the venue's own sample)."""
    if not isinstance(body, dict):
        return []
    reports = body.get("reports")
    if not isinstance(reports, list):
        reports = [body] if body.get("lastQuantity") else []
    slices: list[FillSlice] = []
    seen: set[tuple] = set()
    for report in reports:
        if not isinstance(report, dict):
            continue
        try:
            qty = float(report.get("lastQuantity") or 0)
            cumulative = float(report.get("fillQuantity") or 0)
            price = float(report.get("lastPrice") or 0)
        except (TypeError, ValueError):
            continue
        if qty <= 0 or price <= 0:
            continue                      # lifecycle row, not an execution
        event_no = None
        metadata = report.get("metadata")
        if isinstance(metadata, str) and metadata:
            try:
                raw_no = json.loads(metadata).get("eventNo")
                if raw_no is not None:
                    event_no = int(float(raw_no))
            except (ValueError, TypeError):
                pass                      # diagnostics only — never load-bearing
        key = (cumulative, qty, price, event_no)
        if key in seen:
            continue                      # the documented duplicate row
        seen.add(key)
        slices.append(FillSlice(cumulative=cumulative, qty=qty, price=price,
                                event_no=event_no))
    slices.sort(key=lambda s: s.cumulative)
    return slices


def select_events(slices: "list[FillSlice]", *, booked_cum: float,
                  venue_cum: float, average_price: float
                  ) -> "list[tuple[float, float]]":
    """Budget-clamped ``(qty, price)`` emissions, oldest slice first.

    Emits exactly ``venue_cum - booked_cum`` in total (quantity conservation):
    slices above the watermark fill the budget (last one clamped on
    over-coverage); any residue the slices did not cover emits once at
    ``average_price`` — the degraded-but-never-lost path.
    """
    budget = max(venue_cum - booked_cum, 0.0)
    if budget <= 0:
        return []
    events: list[tuple[float, float]] = []
    for fill_slice in slices:
        if budget <= 0:
            break
        if fill_slice.cumulative <= booked_cum:
            continue                      # already booked (late / re-served row)
        take = min(fill_slice.qty, budget)
        events.append((take, fill_slice.price))
        booked_cum += take
        budget -= take
    if budget > 0:
        events.append((budget, average_price))   # remainder: conserved, degraded
    return events
