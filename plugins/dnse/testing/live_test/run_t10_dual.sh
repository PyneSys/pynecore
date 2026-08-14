#!/usr/bin/env bash
# T10 — dual-strategy isolation test, reusable runner.
#
# Two concurrent `pyne run --broker` engines on ONE DNSE account, aligned so both place
# on the SAME 1m bar: A cancels by id, B fires strategy.cancel_all() while A's order
# rests. PASS = each engine touched only its own ids, no external-cancel/quarantine
# lines, both orders Canceled at the venue, nothing left working.
#
# Needs: open continuous session (09:15-11:30 / 13:00-14:30 ICT), fresh trading token
# (plugins/dnse/tools/token_status.py -> GOOD). First measured PASS: 2026-08-14 13:55.
#
# Usage:  plugins/dnse/testing/live_test/run_t10_dual.sh [lead_minutes]
#   lead_minutes  window opens this many minutes after launch (default 5 — must be
#                 after both warmups; raise on a slow connection)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LT=plugins/dnse/testing/live_test
LEAD="${1:-5}"

echo "== T10 dual-strategy isolation =="
date '+launch %H:%M:%S'

# 1) mandatory L0 gate (user rule: before EVERY live run)
echo "-- L0 gate --"
.venv/bin/python $LT/level0_venue_semantics/l0_order_semantics.py >/dev/null 2>&1 \
  || { echo "L0 gate FAILED — aborting, nothing launched"; exit 1; }
echo "L0 gate: PASS"

# 2) identical windows for both instances, opening after warmup
S=$(python3 -c "import datetime as d;now=d.datetime.now(d.timezone(d.timedelta(hours=7)));print(int((now+d.timedelta(minutes=$LEAD)).replace(second=0,microsecond=0).timestamp()*1000))")
E=$((S + 20*60*1000))   # 20-minute window is ample: each instance needs 2 bars
for f in live_dual_a live_dual_b; do
  # regenerate the toml if missing (first run on a fresh checkout)
  [ -f "$LT/$f.toml" ] || .venv/bin/pyne run "$LT/$f.py" dnse:VN30F1M@1 >/dev/null 2>&1 || true
  python3 - "$S" "$E" "$LT/$f.toml" <<'PY'
import sys, pathlib, re
s, e, path = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path); t = p.read_text()
t = re.sub(r'(\[inputs\.winStart\][\s\S]*?)#?value =.*', rf'\1value = {s}', t, count=1)
t = re.sub(r'(\[inputs\.winEnd\][\s\S]*?)#?value =.*',   rf'\1value = {e}', t, count=1)
p.write_text(t)
PY
done
python3 -c "import datetime as d;print('window opens', d.datetime.fromtimestamp($S/1000).strftime('%H:%M'), '(both instances place on that bar)')"

# 3) launch both concurrently
mkdir -p $LT/logs
.venv/bin/pyne run $LT/live_dual_a.py dnse_broker:VN30F1M@1 --broker > $LT/logs/dual_a.log 2>&1 &
PA=$!
.venv/bin/pyne run $LT/live_dual_b.py dnse_broker:VN30F1M@1 --broker > $LT/logs/dual_b.log 2>&1 &
PB=$!
echo "launched A=$PA B=$PB — waiting for both CANCEL steps (~$((LEAD+3)) min)"
trap 'kill $PA $PB 2>/dev/null || true' EXIT

# 4) wait for completion (place + cancel in each log), then stop both
DEADLINE=$(( $(date +%s) + (LEAD+10)*60 ))
until grep -aq "CANCEL step done" $LT/logs/dual_a.log 2>/dev/null \
   && grep -aq "CANCEL step done" $LT/logs/dual_b.log 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT — check logs, orders may rest"; exit 1; }
  sleep 10
done
sleep 10   # let the post-cancel confirmations land
kill $PA $PB 2>/dev/null || true; trap - EXIT; sleep 2

# 5) grade: cross-contamination greps + venue ground truth
echo "-- grading --"
IDA=$(sed 's/\x1b\[[0-9;]*m//g' $LT/logs/dual_a.log | grep -aoE "event CREATED id=[0-9a-z]+" | head -1 | cut -d= -f2)
IDB=$(sed 's/\x1b\[[0-9;]*m//g' $LT/logs/dual_b.log | grep -aoE "event CREATED id=[0-9a-z]+" | head -1 | cut -d= -f2)
XA=$(sed 's/\x1b\[[0-9;]*m//g' $LT/logs/dual_a.log | grep -acE "$IDB|external cancel|unexpected|quarantin" || true)
XB=$(sed 's/\x1b\[[0-9;]*m//g' $LT/logs/dual_b.log | grep -acE "$IDA|external cancel|unexpected|quarantin" || true)
echo "A placed $IDA, B placed $IDB; cross-hits A=$XA B=$XB (must be 0/0)"

.venv/bin/python - "$IDA" "$IDB" <<'PY'
import asyncio, sys
sys.path.insert(0, "src")
from pathlib import Path
from pynecore.core.config import ensure_config
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig
ida, idb = sys.argv[1], sys.argv[2]
b = DNSEBroker(symbol="VN30F1M", timeframe="1", config=ensure_config(
    DNSEBrokerConfig, Path("workdir/config/plugins/dnse_broker.toml")))
bad = False
for oid, who in ((ida, "A"), (idb, "B")):
    s, body = b.client.get_order_detail(b.account_id, oid, b.market_type,
                                        order_category="NORMAL")
    st = body.get("orderStatus") if isinstance(body, dict) else f"HTTP {s}"
    print(f"venue: {who} {oid} -> {st}")
    if st not in ("Canceled", "Cancelled"):
        bad = True
        print(f"  !! {who}'s order is NOT cancelled — cancel it manually")
working = [str(o.id) for o in asyncio.run(b.get_open_orders("VN30F1M"))
           if str(o.id) in (ida, idb)]
if working:
    bad = True
    print(f"!! still working: {working}")
print("VERDICT:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
PY
RC=$?
[ "$XA" = "0" ] && [ "$XB" = "0" ] || { echo "VERDICT: FAIL (cross-contamination)"; exit 1; }
exit $RC
