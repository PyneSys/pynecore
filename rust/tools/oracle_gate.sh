#!/bin/sh
# The emission oracle of the monorepo (work/ast-oracle/oracle.py) over this
# checkout, once per backend: the verify and native captures must equal the
# Python one. Usage: rust/tools/oracle_gate.sh [OUTDIR]  (python = $PYTHON)
set -e
checkout="$(cd "$(dirname "$0")/../.." && pwd)"
monorepo="$(cd "$checkout/../.." && pwd)"
oracle="$monorepo/work/ast-oracle/oracle.py"
python="${PYTHON:-$monorepo/venv/bin/python}"
out="${1:-${TMPDIR:-/tmp}/pyne-oracle-gate}"
if [ ! -f "$oracle" ]; then
    echo "no oracle at $oracle" >&2
    exit 2
fi
for backend in python verify native; do
    PYNE_AST_BACKEND=$backend "$python" "$oracle" capture --src "$checkout/src" \
        --out "$out/$backend" --jobs "${JOBS:-8}" | tail -1
done
status=0
for backend in verify native; do
    echo "python vs $backend:"
    "$python" "$oracle" compare "$out/python" "$out/$backend" | tail -1 || status=1
    if "$python" - "$out/python" "$out/$backend" <<'PY'
import gzip, json, sys
from pathlib import Path

def load(directory):
    records = {}
    for path in Path(directory).glob('*.json.gz'):
        with gzip.open(path, 'rt') as stream:
            record = json.load(stream)
        records[record['key']] = record
    return records

base, cand = load(sys.argv[1]), load(sys.argv[2])
bad = [key for key in base if base[key].get('status') != cand.get(key, {}).get('status')
       or base[key].get('code_sha') != cand.get(key, {}).get('code_sha')
       or base[key].get('error') != cand.get(key, {}).get('error')]
for key in bad[:10]:
    print('  differs:', key, cand.get(key, {}).get('error', '')[:200])
sys.exit(1 if bad else 0)
PY
    then echo "  identical: status, errors and code digests"; else status=1; fi
done
exit $status
