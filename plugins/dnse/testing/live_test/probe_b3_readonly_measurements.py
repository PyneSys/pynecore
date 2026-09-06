"""Read-only B3 measurements (#74, out-of-session, no token, NO writes).

Three questions the B3 panel left unmeasured:
  M1 — does the shipped paginated drain (`_read_history_rows_sync`) behave
       against the REAL envelope (and does `page_index` paginate honestly
       at a small page_size)?
  M2 — do CONDITIONAL string ids ever appear in /orders/history? (Gates
       whether a conditional residue can ever conclude CANCELLED.)
  M3 — what do the day books return in a CLOSED session (readable-and-empty
       would mean the residue detector ages stamps off-hours)?

Masks 8+ digit runs; prints at most truncated id examples.
"""
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path("/home/mike/workspace/github/pynecore")
sys.path.insert(0, str(REPO / "plugins" / "dnse" / "tools"))
from pynecore.core.config import ensure_config                       # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig        # noqa: E402


def mask(value):
    return re.sub(r"\d{8,}", "<id>", str(value))


cfg = ensure_config(DNSEBrokerConfig,
                    REPO / "workdir/config/plugins/dnse_broker.toml")
b = DNSEBroker(symbol="VN30F1M", timeframe="1", config=cfg)
acct, mkt = b.account_id, b.market_type
ict = timezone(timedelta(hours=7))
now = datetime.now(ict)
print(f"probe @ {now:%Y-%m-%d %H:%M} ICT (weekday={now.weekday()}) market_type={mkt}")

# ---- M1a: the SHIPPED reader, verbatim (yesterday..today window) -----------
print("\n=== M1a: _read_history_rows_sync() — the shipped drain, verbatim ===")
rows, complete = b._read_history_rows_sync()
print(f"  rows={len(rows)} complete={complete}")

# ---- M1b: forced pagination at page_size=3 over a 30-day window ------------
print("\n=== M1b: manual drain, page_size=3, 30-day window (forces pages) ===")
frm = str(now.date() - timedelta(days=30))
to = str(now.date())
all_ids, page_index, total_seen = [], 0, None
while page_index < 20:
    st, body = b.client.get_order_history(acct, mkt, from_date=frm, to_date=to,
                                          page_size=3, page_index=page_index)
    if st != 200 or not isinstance(body, dict):
        print(f"  page {page_index}: http={st} -> STOP")
        break
    page = body.get("data") or []
    total_seen = body.get("total")
    ids = [str(r.get("id")) for r in page]
    dup = [i for i in ids if i in all_ids]
    all_ids.extend(ids)
    print(f"  page {page_index}: http={st} rows={len(page)} total={total_seen} "
          f"keys={sorted(body.keys())} dup_vs_prior={len(dup)}")
    if not page or (isinstance(total_seen, int) and len(all_ids) >= total_seen):
        break
    page_index += 1
unique = len(set(all_ids))
print(f"  DRAINED: {len(all_ids)} rows ({unique} unique) vs total={total_seen} "
      f"-> drain_complete_and_honest={unique == len(all_ids) and total_seen == len(all_ids)}")

# ---- M2: conditional string ids in history? --------------------------------
print("\n=== M2: id shapes in the 30-day history drain ===")
numeric, stringy, examples = 0, 0, []
statuses = {}
for raw_id in all_ids:
    suffix = raw_id.split("_")[-1]
    if suffix.isdigit():
        numeric += 1
    else:
        stringy += 1
        if len(examples) < 3:
            examples.append(mask(raw_id)[:28])
print(f"  numeric-suffix (NORMAL-book) ids: {numeric}")
print(f"  string-suffix (conditional-book) ids: {stringy} examples={examples}")
# status + field inventory on a sample row (masked), to see category markers
st, body = b.client.get_order_history(acct, mkt, from_date=frm, to_date=to,
                                      page_size=200, page_index=0)
sample_rows = (body.get("data") or []) if isinstance(body, dict) else []
for r in sample_rows:
    statuses[str(r.get("orderStatus"))] = statuses.get(str(r.get("orderStatus")), 0) + 1
print(f"  statuses_present={statuses}")
if sample_rows:
    print(f"  row_fields={sorted(sample_rows[0].keys())}")

# ---- M3: closed-session day books ------------------------------------------
print("\n=== M3: day books NOW (closed session / weekend) ===")
for category in ("NORMAL", "STOP", "OCO"):
    st, body = b.client.get_orders(acct, mkt, order_category=category,
                                   page_index=0, page_size=100)
    if isinstance(body, dict):
        n = len(body.get("orders") or [])
        meta = {k: body.get(k) for k in ("totalPages", "pageSize", "pageIndex")}
        print(f"  {category}: http={st} rows={n} meta={meta}")
    else:
        print(f"  {category}: http={st} body_type={type(body).__name__}")
print("\n  interpretation: http=200 with rows=0 on all books means an "
      "off-session watch cycle counts as 'all books readable, id absent' — "
      "the residue detector WOULD age stamps off-hours (grace+INCONCLUSIVE "
      "keep it safe; noise profile only).")
