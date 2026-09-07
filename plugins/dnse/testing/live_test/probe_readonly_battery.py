"""Read-only live battery (holiday, no token, no writes). Masks ids/accounts."""
import re, sys, urllib3
from datetime import datetime, timedelta, timezone
from pathlib import Path
REPO = Path("/home/mike/workspace/github/pynecore")
sys.path.insert(0, str(REPO / "plugins" / "dnse" / "tools"))
from pynecore.core.config import ensure_config
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig
from pynecore_dnse.client import DNSEClient
from pynecore_dnse import errors
from pynecore_dnse.transport_errors import to_sentinel

def mask(s):  # any 8+ digit run (account/custody/order ids) -> <id>
    return re.sub(r"\d{8,}", "<id>", str(s))

cfg = ensure_config(DNSEBrokerConfig, REPO / "workdir/config/plugins/dnse_broker.toml")
b = DNSEBroker(symbol="VN30F1M", timeframe="1", config=cfg)
acct, mkt = b.account_id, b.market_type
print(f"market_type={mkt}")

# ---------- PROBE 1: /positions envelope + paging fields + CLOSED rows ----------
print("\n=== PROBE 1: /positions envelope (page_size=500 then 1) ===")
for ps in (500, 1):
    st, body = b.client.get_positions(acct, mkt, ps)
    rows = (body.get("positions") or body.get("data") or []) if isinstance(body, dict) else []
    meta = {k: body.get(k) for k in ("total", "pageSize", "pageIndex", "pageNumber")} if isinstance(body, dict) else {}
    statuses = sorted({str(r.get("status")) for r in rows})
    print(f"  page_size={ps}: http={st} keys={sorted(body.keys()) if isinstance(body, dict) else type(body).__name__}")
    print(f"     meta={meta} rows_delivered={len(rows)} statuses_present={statuses}")

# ---------- PROBE 2: real pagination on /orders/history (page_size=1) ----------
print("\n=== PROBE 2: /orders/history real pagination (page_size=1, page 0 then 1) ===")
ict = timezone(timedelta(hours=7)); today = datetime.now(ict).date()
frm, to = str(today - timedelta(days=7)), str(today)
seen = []
for pi in (0, 1):
    st, body = b.client.get_order_history(acct, mkt, from_date=frm, to_date=to, page_size=1, page_index=pi)
    rows = (body.get("data") or body.get("orders") or []) if isinstance(body, dict) else []
    meta = {k: body.get(k) for k in ("totalPages", "total", "pageSize", "pageIndex", "pageNumber")} if isinstance(body, dict) else {}
    ids = [mask(r.get("id")) for r in rows]
    seen.append(rows[0].get("id") if rows else None)
    print(f"  page_index={pi}: http={st} keys={sorted(body.keys()) if isinstance(body, dict) else type(body).__name__}")
    print(f"     meta={meta} rows={len(rows)} ids={ids}")
print(f"  page0 != page1 row (drain yields DIFFERENT rows): {seen[0] is not None and seen[0] != seen[1]}")
st, body = b.client.get_orders(acct, mkt, order_category="NORMAL", page_index=0, page_size=1)
meta = {k: body.get(k) for k in ("totalPages", "pageSize", "pageIndex")} if isinstance(body, dict) else {}
print(f"  /orders today (holiday) NORMAL page_size=1: http={st} meta={meta} rows={len(body.get('orders') or []) if isinstance(body, dict) else '?'}")

# ---------- PROBE 3: REAL venue auth refusal shapes (#54/#67 AUTH ladder) ----------
print("\n=== PROBE 3: real 401 shapes -> errors.classify (wrong key; wrong secret) ===")
for label, key, sec in (("wrong-key", "not-a-real-key", cfg.api_secret), ("wrong-secret", cfg.api_key, "not-a-real-secret")):
    bad = DNSEClient(key, sec)
    st, body = bad.get_orders(acct, mkt, order_category="NORMAL", page_index=0, page_size=1)
    code = errors.code_of(body) if isinstance(body, dict) else None
    msg = mask(body.get("message"))[:80] if isinstance(body, dict) else mask(body)[:80]
    c = errors.classify(st, body, is_write=False)
    print(f"  {label}: http={st} code={code} msg={msg!r}")
    print(f"     classify -> {c.disposition.name if c else None}  (AUTH-class for the #54 halt: {c.disposition.name in ('AUTH','AUTH_TOKEN') if c else False})")

# ---------- PROBE 4: REAL transport exception shapes (#67) ----------
print("\n=== PROBE 4: real transport exception shapes -> to_sentinel ===")
def shape(fn, label):
    try:
        r = fn(); print(f"  {label}: NO EXCEPTION (returned http={r[0]})"); return
    except Exception as e:  # noqa: BLE001
        reason = getattr(e, "reason", None)
        print(f"  {label}: {type(e).__name__} | HTTPError={isinstance(e, urllib3.exceptions.HTTPError)} "
              f"| MaxRetryError={isinstance(e, urllib3.exceptions.MaxRetryError)} | reason={type(reason).__name__ if reason else None}")
        print(f"     to_sentinel -> {to_sentinel(e)}")
# read-timeout against the REAL venue (GET only — never a POST)
tmo = DNSEClient(cfg.api_key, cfg.api_secret)
tmo._sdk._http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=10.0, read=0.001), cert_reqs="CERT_REQUIRED")
shape(lambda: tmo.get_orders(acct, mkt, order_category="NORMAL", page_index=0, page_size=1), "read-timeout GET (real venue)")
# connect-phase against a non-routable host (never reaches any venue)
unr = DNSEClient(cfg.api_key, cfg.api_secret, base_url="https://10.255.255.1")
unr._sdk._http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=1.0, read=1.0), retries=urllib3.Retry(1))
shape(lambda: unr.get_orders(acct, mkt, order_category="NORMAL", page_index=0, page_size=1), "connect-fail GET (unroutable)")
