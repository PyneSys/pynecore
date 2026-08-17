#!/usr/bin/env bash
# T16 — restart adoption test (card rubycell/pynecore#22, Group B).
#
# Every real deployment eventually restarts against a venue book that still holds its
# own resting orders. The quarantine/adoption path has only ever fired ACCIDENTALLY
# (the #19 measurement) — this makes it a designed, repeatable test.
#
# Phase 1: launch the param probe at startState=10 (T16a): places ONE far limit
#          (1 contract, -5%, cannot fill) and HOLDS. We then kill the engine.
# Phase 2: relaunch the SAME strategy+account at startState=11 (T16b): the broker's
#          startup reconciliation meets the pre-existing resting order. It must adopt
#          or quarantine it EXPLICITLY (logged), place NOTHING new, and cancel_all()
#          must reach it. PASS = venue-clean at the end.
#
# Kill modes: SIGTERM (default; teardown runs) or SIGKILL (hard death; no teardown).
# Run BOTH modes — they exercise different recovery paths.
#
# Needs: open continuous session, fresh trading token. Grade the final verdict from
# DNSE's own records (>=1 bar wait), per the suite README.
#
# Usage:  plugins/dnse/testing/live_test/run_t16_restart.sh [lead_minutes] [TERM|KILL]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LT=plugins/dnse/testing/live_test
LEAD="${1:-5}"
MODE="${2:-TERM}"
[ "$MODE" = "TERM" ] || [ "$MODE" = "KILL" ] || { echo "mode must be TERM or KILL"; exit 2; }

echo "== T16 restart adoption (kill mode: SIG$MODE) =="
date '+launch %H:%M:%S'

# 1) mandatory L0 gate (before EVERY live run)
echo "-- L0 gate --"
.venv/bin/python $LT/level0_venue_semantics/l0_order_semantics.py >/dev/null 2>&1 \
  || { echo "L0 gate FAILED — aborting, nothing launched"; exit 1; }
echo "L0 gate: PASS"

patch_toml () {  # $1 winStart-ms  $2 winEnd-ms  $3 startState
  # regenerate the toml if missing (first run on a fresh checkout)
  [ -f "$LT/live_staged_params.toml" ] || \
    .venv/bin/pyne run "$LT/live_staged_params.py" dnse:VN30F1M@1 >/dev/null 2>&1 || true
  python3 - "$1" "$2" "$3" "$LT/live_staged_params.toml" <<'PY'
import sys, pathlib, re
s, e, st, path = sys.argv[1:5]
p = pathlib.Path(path); t = p.read_text()
t = re.sub(r'(\[inputs\.winStart\][\s\S]*?)#?value =.*',   rf'\1value = {s}',  t, count=1)
t = re.sub(r'(\[inputs\.winEnd\][\s\S]*?)#?value =.*',     rf'\1value = {e}',  t, count=1)
t = re.sub(r'(\[inputs\.startState\][\s\S]*?)#?value =.*', rf'\1value = {st}', t, count=1)
p.write_text(t)
PY
}

now_ms () { python3 -c "import datetime as d;now=d.datetime.now(d.timezone(d.timedelta(hours=7)));print(int((now+d.timedelta(minutes=$1)).replace(second=0,microsecond=0).timestamp()*1000))"; }

mkdir -p $LT/logs

# 2) PHASE 1 — place and hold
S=$(now_ms "$LEAD"); E=$((S + 60*60*1000))
patch_toml "$S" "$E" 10
python3 -c "import datetime as d;print('phase 1: window opens', d.datetime.fromtimestamp($S/1000).strftime('%H:%M'), 'startState=10 (T16a place-and-hold)')"
.venv/bin/pyne run $LT/live_staged_params.py dnse_broker:VN30F1M@1 --broker > $LT/logs/t16_phase1.log 2>&1 &
P1=$!
trap 'kill $P1 2>/dev/null || true' EXIT
echo "launched phase 1 pid=$P1 — waiting for T16a HOLDING (order resting)"
DEADLINE=$(( $(date +%s) + (LEAD+10)*60 ))
until grep -aq "T16a HOLDING" $LT/logs/t16_phase1.log 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT waiting for placement — check log"; exit 1; }
  kill -0 $P1 2>/dev/null || { echo "phase-1 engine died early — check log"; exit 1; }
  sleep 5
done
ID1=$(sed 's/\x1b\[[0-9;]*m//g' $LT/logs/t16_phase1.log | grep -aoE "event CREATED id=[0-9A-Za-z_-]+" | head -1 | cut -d= -f2 || true)
echo "order resting (venue id: ${ID1:-unknown}) — killing engine with SIG$MODE"
sleep 5   # one extra sync so the resting order is in persisted broker state

# 3) the kill
kill "-$MODE" $P1 2>/dev/null || true
trap - EXIT
# give TERM a moment to run teardown; KILL dies instantly
sleep 8
kill -0 $P1 2>/dev/null && { kill -KILL $P1 2>/dev/null || true; sleep 2; }
echo "engine down. The T16 order should still be RESTING at the venue."

# 4) PHASE 2 — relaunch, observe adoption, sweep
S2=$(now_ms 1); E2=$((S2 + 60*60*1000))
patch_toml "$S2" "$E2" 11
echo "phase 2: relaunch startState=11 (T16b observe + cancel_all)"
.venv/bin/pyne run $LT/live_staged_params.py dnse_broker:VN30F1M@1 --broker > $LT/logs/t16_phase2.log 2>&1 &
P2=$!
trap 'kill $P2 2>/dev/null || true' EXIT
DEADLINE=$(( $(date +%s) + 15*60 ))
until grep -aq "T16b POST-RESTART cancel_all" $LT/logs/t16_phase2.log 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT in phase 2 — ORDER MAY STILL REST, cancel manually"; exit 1; }
  kill -0 $P2 2>/dev/null || { echo "phase-2 engine died — ORDER MAY STILL REST, check log"; exit 1; }
  sleep 5
done
sleep 15   # let the cancel confirmations land (2xx is an ACK, not a completion — #20)
kill $P2 2>/dev/null || true; trap - EXIT; sleep 2

# 5) grade from the logs; venue record is the final authority
echo "-- grading --"
L2=$LT/logs/t16_phase2.log
STRIP() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
ADOPT=$(STRIP $L2 | grep -acE "adopt|quarantin|reconcil|external order" || true)
DUP=$(STRIP $L2 | grep -acE "event CREATED id=" || true)
CANC=$(STRIP $L2 | grep -acE "cancel" || true)
echo "phase-2 adoption/quarantine/reconcile lines : $ADOPT  (0 = SILENT ownership -> FAIL)"
echo "phase-2 NEW placements (CREATED)            : $DUP  (>0 = duplicate placement -> FAIL)"
echo "phase-2 cancel activity lines               : $CANC  (0 = sweep never reached it -> FAIL)"
if [ "${ADOPT:-0}" -gt 0 ] && [ "${DUP:-0}" -eq 0 ] && [ "${CANC:-0}" -gt 0 ]; then
  echo "T16 (SIG$MODE): tentative PASS — NOW CONFIRM AT THE VENUE: order ${ID1:-?} terminal, nothing working"
  exit 0
else
  echo "T16 (SIG$MODE): FAIL or inconclusive — read $LT/logs/t16_phase{1,2}.log and check the venue NOW"
  exit 1
fi
