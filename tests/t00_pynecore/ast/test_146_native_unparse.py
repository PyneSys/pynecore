"""The native ``ast.unparse`` (``rust/crates/pyne-transform/src/unparse.rs``) against CPython's.

The native steps spell expressions into messages and identifiers the way the Python
steps do, with the running interpreter's ``ast.unparse``: every expression of the
parity corpus of ``test_137_native_transform`` and a set of strings that exercise the
quote choice, the escapes and the printability table must unparse to the same text,
f-strings in the spelling of the running interpreter's version.
"""
import ast
from pathlib import Path

from .test_137_native_transform import __test_helper_inputs, __test_helper_require_native

#: Expressions whose spelling turns on what a corpus rarely holds
__test_helper_CASES = [
    "f'a{x}b'", "f\"{x!r:>{w}}\"", "f'{x=}'", "f'{ x = !s:10}'", "f'{{a}}{x}'",
    "f'''a\n{x}\nb'''", "f'{x}\\n'", "f\"it's {x}\"", "f'say \"{x}\"'", "f'''it's \"{x}\"'''",
    "f'{x:%Y-%m-%d}'", "f'{f\"{x}\"}'", "f'{x:{y}.{z}f}'", "f'\\u25b2 {x}'", "f'{x}\\t\\x00'",
    "f'{x[\"a\"]}'", "f'{a!r} {\"q\"} {b:{c!s}}'", "f'''{x}\n{y}'''", "f'{x:\\n}'",
    "'▲'", "'▼ ● ■'", "'\\u200b'", "'\\x7f\\x80\\xa0\\xad'", "'\\U0001f4ca'",
    "'a\\'b\"c'", "\"it's\"", "u'a'", "b''", "b'a\\'b'", "b\"it's\"", "b'\\x00\\xff\\\\'",
    "1j", "2.5j", "1e300j * 1e300", "-0.0", "1e309", "(1, 'a', None)", "await x",
    "(yield)", "(yield x)", "(yield from x)", "lambda: (yield)",
]


def __test_helper_compare(node: ast.expr) -> str | None:
    """The mismatch of one expression, None when the spellings agree or native defers."""
    from pynecore.transformers import _native

    try:
        want = ast.unparse(node)
    except ValueError:
        return None
    try:
        got = _native._unparse(node)
    except _native.Unsupported:
        return f'{want!r}: native defers'
    return None if got == want else f'{want!r} != native {got!r}'


def __test_cases_unparse_like_cpython__():
    """Quotes, escapes, printability and the rarer constants follow CPython"""
    __test_helper_require_native()
    mismatches = []
    for source in __test_helper_CASES:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.expr) and not isinstance(node, (ast.Starred, ast.Slice)):
                mismatch = __test_helper_compare(node)
                if mismatch is not None:
                    mismatches.append(mismatch)
    assert not mismatches, mismatches[:10]


def __test_corpus_unparses_like_cpython__():
    """Every expression of the parity corpus unparses to CPython's text"""
    __test_helper_require_native()
    mismatches = []
    for path in __test_helper_inputs():
        tree = ast.parse(Path(path).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.expr) and not isinstance(node, (ast.Starred, ast.Slice)):
                mismatch = __test_helper_compare(node)
                if mismatch is not None:
                    mismatches.append(f'{Path(path).name}: {mismatch}')
    assert not mismatches, mismatches[:10]
