#!/usr/bin/env bash
# run_fill_case.sh — fill-tier runner (#91): one F-case, fully sequenced.
#
# Replaces ad-hoc live monitoring (2026-09-08: hand-monitoring produced two
# operator-caught mistakes — a wrong "flatten-on-sight" briefing, and a
# flatten at 3.5 bars that tripped the script's 5-bar HALT). Every step
# prints a [RUN] banner; a human or Claude watches THIS output only.
#
# Sequence: gates (L0, venue-flat, ATC headroom) -> window patch -> launch
# -> wait PLACE -> wait FILL (or clean no-fill window end) -> on the
# script's ">>> OPERATOR: CLOSE THIS POSITION NOW <<<" line: grace, then
# flatten_api (close-first + owned-protection sweep, #91) -> wait for the
# engine to observe the external close -> TERM the engine -> final report.
#
# Usage:  run_fill_case.sh <startState 0-7> [lead_minutes=2] [--check]
#   --check: validate gates and print the plan, launch nothing.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LT=plugins/dnse/testing/live_test
STATE="${1:?usage: run_fill_case.sh <startState 0-7> [lead_minutes] [--check]}"
LEAD="${2:-2}"
CHECK=0; [[ "${2:-}" == "--check" || "${3:-}" == "--check" ]] && CHECK=1
[[ "$STATE" =~ ^[0-7]$ ]] || { echo "[RUN] startState must be 0-7"; exit 2; }

say () { echo "[RUN] $(date '+%H:%M:%S') $*"; }

# --- gates -------------------------------------------------------------------
say "gate 1/3: ATC headroom (placements + flatten must finish by 14:20)"
NOW_M=$(( $(date '+%H' | sed 's/^0//') * 60 + $(date '+%M' | sed 's/^0//') ))
LATEST=$(( 14 * 60 + 20 - LEAD - 8 ))   # window + fill + flatten budget ~8m
if (( NOW_M > LATEST )); then
  say "REFUSED: too close to ATC (now +lead +8m past 14:20). Run next session."
  exit 2
fi
say "gate 2/3: L0 venue semantics (mandatory before every live run)"
if ! .venv/bin/python $LT/level0_venue_semantics/l0_order_semantics.py >/dev/null 2>&1; then
  say "L0 gate FAILED — nothing launched"; exit 1
fi
say "L0 PASS"
say "gate 3/3: venue flat"
if ! .venv/bin/python plugins/dnse/tools/venue.py flat >/dev/null 2>&1; then
  say "venue NOT flat/clean — resolve first (venue.py status)"; exit 1
fi
say "venue FLAT"
if (( CHECK )); then
  say "--check: all gates pass; would run F-state $STATE with lead ${LEAD}m"
  exit 0
fi

# --- window ------------------------------------------------------------------
S=$(python3 -c "import datetime as d;now=d.datetime.now(d.timezone(d.timedelta(hours=7)));print(int((now+d.timedelta(minutes=$LEAD)).replace(second=0,microsecond=0).timestamp()*1000))")
E=$((S + 5*60*1000))
python3 - "$S" "$E" "$STATE" "$LT/live_staged_fill.toml" <<'PY'
import sys, pathlib, re
s, e, st, path = sys.argv[1:5]
p = pathlib.Path(path); t = p.read_text()
t = re.sub(r'(\[inputs\.winStart\][\s\S]*?)#?value =.*',   rf'\1value = {s}',  t, count=1)
t = re.sub(r'(\[inputs\.winEnd\][\s\S]*?)#?value =.*',     rf'\1value = {e}',  t, count=1)
t = re.sub(r'(\[inputs\.startState\][\s\S]*?)#?value =.*', rf'\1value = {st}', t, count=1)
p.write_text(t)
PY
say "window: opens +${LEAD}m, 5m wide, startState=$STATE"

# --- launch ------------------------------------------------------------------
mkdir -p $LT/logs
LOG=$LT/logs/fill_case_${STATE}.log
.venv/bin/pyne run $LT/live_staged_fill.py dnse_broker:VN30F1M@1 --broker > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
say "launched pid=$PID log=$LOG — waiting for PLACE"
STRIP() { sed 's/\x1b\[[0-9;]*m//g' "$LOG"; }
DEADLINE=$(( $(date +%s) + (LEAD+7)*60 ))
until STRIP | grep -aq "\[F\] F[0-9]* PLACE"; do
  (( $(date +%s) > DEADLINE )) && { say "TIMEOUT waiting for PLACE — check $LOG"; exit 1; }
  kill -0 $PID 2>/dev/null || { say "engine died pre-PLACE — check $LOG"; exit 1; }
  sleep 5
done
say "PLACED: $(STRIP | grep -a '\[F\] F[0-9]* PLACE' | tail -1 | sed 's/.*\[F\] //')"

# --- fill or clean window end ------------------------------------------------
say "waiting for FILL (the CLOSE-NOW line) or clean window end"
DEADLINE=$(( $(date +%s) + 9*60 ))
FILLED=0
until (( FILLED )); do
  if STRIP | grep -aq "OPERATOR: CLOSE THIS POSITION NOW"; then FILLED=1; break; fi
  if STRIP | grep -aq "DONE — flat again\|ALL .* DONE"; then break; fi
  (( $(date +%s) > DEADLINE )) && { say "no fill inside budget — treating as no-fill end"; break; }
  kill -0 $PID 2>/dev/null || { say "engine DIED mid-case — flatten + sweep NOW"; break; }
  sleep 5
done

if (( FILLED )); then
  say "FILLED — grace 4s (protection race, measured F2), then flatten (#91: close-first + owned sweep)"
  sleep 4
  set +e
  .venv/bin/python $LT/flatten_api.py; RC=$?
  set -e
  say "flatten rc=$RC (0 flat+swept, 1 unresolved, 2 undetermined)"
  say "waiting for the engine to observe the external close (<=2 bars)"
  DEADLINE=$(( $(date +%s) + 150 ))
  until STRIP | grep -aq "external close detected"; do
    (( $(date +%s) > DEADLINE )) && { say "engine did not log the external close — note for grading"; break; }
    kill -0 $PID 2>/dev/null || break
    sleep 5
  done
fi

# --- teardown + final report -------------------------------------------------
say "stopping engine (TERM)"
kill $PID 2>/dev/null || true; trap - EXIT; sleep 5
kill -0 $PID 2>/dev/null && kill -KILL $PID 2>/dev/null || true
say "final venue check"
set +e
.venv/bin/python plugins/dnse/tools/venue.py flat; FLAT_RC=$?
set -e
say "venue.py flat rc=$FLAT_RC (0 = flat AND no live orders)"
say "grade from the VENUE record + $LOG (registry rules); evidence-strip before commit"
exit $FLAT_RC
