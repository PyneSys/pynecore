"""
Maintain ``crates/pyne-transform/ported.json``: per native step, the Python module it
ports and the digest of that module's sources at the time of the port.

    python rust/tools/ported.py                    # list every step, current or stale
    python rust/tools/ported.py --check            # exit 1 if any step or the build is stale
    python rust/tools/ported.py --update NAME ...  # record NAME as ported from today's source
    python rust/tools/ported.py --add NAME MODULE  # a new step, ported from transformers/MODULE.py

Update a step only after its port matches the Python step again (the verify
backend over the AST tests and the oracle). The manifest is compiled into the
extension, so rebuild it afterwards (``rust/tools/develop.sh``).

The extension Python actually loads is checked against the manifest too: a build
from an older manifest, or a stale interpreter-specific ``_native.cpython-*.so``
that Python loads before the fresh ``_native.abi3.so``, runs the steps it is
behind on in Python, with nothing but a one-line warning to show for it.
"""
import argparse
import json
import sys
from pathlib import Path

from pynecore.transformers.native import source_digest

try:
    from pynecore.transformers import _native
except ImportError:
    _native = None

MANIFEST = Path(__file__).resolve().parent.parent / 'crates' / 'pyne-transform' / 'ported.json'


def check_build(manifest: dict) -> int:
    """Report whether the loaded extension was built from the manifest as it is now.

    :param manifest: The manifest, as written to disk
    :return: The number of steps the loaded build is behind on
    """
    if _native is None:
        print("no native extension loaded: every step runs in Python")
        return 0
    loaded = Path(_native.__file__)
    print(f"loaded extension: {loaded}")
    others = sorted(p.name for p in loaded.parent.glob('_native*.so') if p != loaded)
    if others:
        print(f"  not loaded beside it: {', '.join(others)}")
    built = json.loads(_native.ported_json())
    registered = set(_native.steps())
    behind = sorted(name for name, entry in manifest.items()
                    if name not in registered or built.get(name, {}).get('digest') != entry['digest'])
    for name in behind:
        print(f"BUILD {name}: the loaded extension predates its port, rebuild "
              f"(rust/tools/develop.sh)")
    return len(behind)


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
    stale += check_build(manifest)
    return 1 if args.check and stale else 0


if __name__ == '__main__':
    sys.exit(main())
