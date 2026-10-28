"""
Parse gate: the native parser against CPython's ``ast.parse``.

Every input is parsed both ways and compared with
``ast.dump(..., include_attributes=True)``, so locations count. A module the
native parser refuses (``Unsupported``) is counted apart: the host parses those
with CPython.

    python rust/tools/parse_gate.py [PATH ...]

Without arguments the inputs are this checkout's ``src`` and ``tests`` and the
Pyne-in-the-Wild corpus next to the monorepo checkout.
"""
import ast
import difflib
import sys
from collections import Counter
from pathlib import Path

from pynecore.transformers import _native

CHECKOUT = Path(__file__).resolve().parents[2]
MONOREPO = CHECKOUT.parents[1]


def default_inputs() -> list[Path]:
    roots = [CHECKOUT / 'src', CHECKOUT / 'tests',
             MONOREPO / 'pyne-in-the-wild' / 'results' / 'runs',
             MONOREPO / 'pyne-in-the-wild' / 'corpus' / 'libs']
    return [path for root in roots if root.exists() for path in sorted(root.rglob('*.py'))]


def main() -> int:
    # Dumping a deeply nested module recurses once per level
    sys.setrecursionlimit(200_000)
    paths = [Path(arg) for arg in sys.argv[1:]] or default_inputs()
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    shown = 0
    for path in paths:
        try:
            source = path.read_text(encoding='utf-8')
            expected = ast.dump(ast.parse(source), include_attributes=True)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            counts['cpython-rejects'] += 1
            continue
        try:
            actual = ast.dump(_native.parse(source), include_attributes=True)
        except _native.Unsupported as exc:
            counts['unsupported'] += 1
            reasons[str(exc).split(':')[0]] += 1
            continue
        if actual == expected:
            counts['equal'] += 1
            continue
        counts['DIFFERENT'] += 1
        if shown < 5:
            shown += 1
            diff = difflib.unified_diff(
                ast.dump(ast.parse(source), include_attributes=True, indent=1).splitlines(),
                ast.dump(_native.parse(source), include_attributes=True, indent=1).splitlines(),
                lineterm='', n=3)
            print(f'--- {path}')
            print('\n'.join(list(diff)[:40]))
    print(dict(counts))
    if reasons:
        print('unsupported:', dict(reasons.most_common()))
    return 1 if counts['DIFFERENT'] else 0


if __name__ == '__main__':
    sys.exit(main())
