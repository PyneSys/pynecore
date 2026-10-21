"""
Maintain ``crates/pyne-transform/ported.json``: per native step, the Python module it
ports and the digest of that module's sources at the time of the port.

    python rust/tools/ported.py                    # list every step, current or stale
    python rust/tools/ported.py --check            # exit 1 if any step is stale
    python rust/tools/ported.py --update NAME ...  # record NAME as ported from today's source
    python rust/tools/ported.py --add NAME MODULE  # a new step, ported from transformers/MODULE.py

Update a step only after its port matches the Python step again (the verify
backend over the AST tests and the oracle). The manifest is compiled into the
extension, so rebuild it afterwards (``rust/tools/develop.sh``).
"""
import argparse
import json
import sys
from pathlib import Path

from pynecore.transformers.native import source_digest

MANIFEST = Path(__file__).resolve().parent.parent / 'crates' / 'pyne-transform' / 'ported.json'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--update', nargs='+', metavar='NAME')
    parser.add_argument('--add', nargs=2, metavar=('NAME', 'MODULE'))
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    if args.add:
        name, module = args.add
        manifest[name] = {'module': module, 'digest': source_digest(module)}
    for name in args.update or ():
        manifest[name]['digest'] = source_digest(manifest[name]['module'])
    if args.add or args.update:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    stale = 0
    for name, entry in manifest.items():
        current = source_digest(entry['module']) == entry['digest']
        stale += not current
        print(f"{'ok   ' if current else 'STALE'} {name} ({entry['module']})")
    return 1 if args.check and stale else 0


if __name__ == '__main__':
    sys.exit(main())
