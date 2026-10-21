"""The native transform (``rust/``) against the Python pipeline it ports.

The native wheel gate (``PYNE_REQUIRE_NATIVE_TRANSFORM=1``) requires the extension
and every port to be current. A source install without the extension skips the
native checks.
"""
import ast
import os
from pathlib import Path

import pytest

from pynecore.core import import_hook
from pynecore.transformers import native

#: A module exercising every branch of the ported steps
__test_helper_SOURCE = '''"""
@pyne
"""
from pynecore.lib import close


def helper(x):
    from pynecore.lib import ta
    from pynecore.lib import ta
    from pynecore.lib.math import abs as absolute

    def inner():
        from pynecore.lib import math
        return math.max(1, 2)

    return ta.sma(x, 2) + absolute(x) + inner()


async def ignored():
    from pynecore.lib import ta
    return ta


def main():
    from pynecore.lib import ta as indicators
    return helper(close) + indicators.ema(close, 3)
'''

_TESTS = Path(__file__).resolve().parents[2]
_LIB = Path(import_hook.__file__).resolve().parent.parent / 'lib'


def __test_helper_require_native():
    if not native.available():
        if os.environ.get('PYNE_REQUIRE_NATIVE_TRANSFORM'):
            pytest.fail('the native transform extension is not installed')
        pytest.skip('native transform extension not built')


def __test_helper_inputs() -> list[Path]:
    """Pyne modules of the test suite and the lib, for the parity checks."""
    paths = sorted((_TESTS / 't00_pynecore' / 'ast').glob('test_*.py'))
    paths += sorted(path for path in _LIB.rglob('*.py') if '@pyne' in path.read_text()[:400])
    return paths


def __test_native_ports_are_current__():
    """Every native step was ported from the Python source as it is now."""
    __test_helper_require_native()
    from pynecore.transformers import _native

    assert native.current_steps() == frozenset(_native.steps())


def __test_native_parse_matches_cpython__():
    """The native parser builds CPython's tree, locations included."""
    __test_helper_require_native()
    from pynecore.transformers import _native

    sources = [__test_helper_SOURCE] + [path.read_text() for path in __test_helper_inputs()]
    for source in sources:
        try:
            tree = _native.parse(source)
        except _native.Unsupported:
            continue
        assert (ast.dump(tree, include_attributes=True)
                == ast.dump(ast.parse(source), include_attributes=True))


def __test_native_dump_matches_cpython__():
    """The native ``ast.dump`` writes what CPython writes."""
    __test_helper_require_native()
    from pynecore.transformers import _native

    for path in __test_helper_inputs():
        tree = ast.parse(path.read_text())
        assert _native.dump(tree, True) == ast.dump(tree, include_attributes=True), path


@pytest.mark.parametrize('source', ['helper', 'suite'])
def __test_native_steps_match_python__(source, monkeypatch):
    """Under the verify backend the whole pipeline runs, every native step checked."""
    __test_helper_require_native()
    monkeypatch.setenv('PYNE_AST_BACKEND', 'verify')
    if source == 'helper':
        items = [(Path('native_parity.py'), __test_helper_SOURCE)]
    else:
        items = [(path, path.read_text()) for path in __test_helper_inputs()]
    for path, text in items:
        tree = ast.parse(text)
        user, mode = import_hook._module_mode(tree)
        if not user:
            continue
        analysed = import_hook._analyse_tree(tree, text, path.resolve(), mode)
        import_hook._lower_tree(analysed, path.resolve(), mode)


def __test_helper_emission(backend: str, monkeypatch) -> str:
    """The lowered emission of the helper module under a backend."""
    monkeypatch.setenv('PYNE_AST_BACKEND', backend)
    path = Path('native_parity.py').resolve()
    tree = ast.parse(__test_helper_SOURCE)
    mode = import_hook._module_mode(tree)[1]
    analysed = import_hook._analyse_tree(tree, __test_helper_SOURCE, path, mode)
    lowered, _ = import_hook._lower_tree(analysed, path, mode)
    return ast.dump(lowered, include_attributes=True)


def __test_native_backend_emits_what_python_emits__(monkeypatch):
    """The ``native`` backend's emission is the Python backend's, text and locations."""
    __test_helper_require_native()
    assert (__test_helper_emission('native', monkeypatch)
            == __test_helper_emission('python', monkeypatch))


def __test_stale_port_runs_in_python__(monkeypatch):
    """A port whose Python source changed is out of use, and the result does not change."""
    __test_helper_require_native()
    expected = __test_helper_emission('python', monkeypatch)
    monkeypatch.setattr(native, '_current', None)
    monkeypatch.setattr(native, 'source_digest', lambda module: 'edited')
    assert native.current_steps() == frozenset()
    assert __test_helper_emission('native', monkeypatch) == expected


@pytest.mark.parametrize('backend', ['native', 'verify'])
def __test_native_rejection_matches_python__(backend, monkeypatch):
    """A module the Python step rejects is rejected natively too, with Python's error."""
    __test_helper_require_native()
    monkeypatch.setenv('PYNE_AST_BACKEND', backend)
    source = '"""\n@pyne\n"""\nfrom pynecore.lib import plot, close\n\n\ndef main():\n' \
             '    if close > 1:\n        plot(close)\n'
    tree = ast.parse(source)
    mode = import_hook._module_mode(tree)[1]
    with pytest.raises(SyntaxError, match='plot'):
        import_hook._analyse_tree(tree, source, Path('rejected.py').resolve(), mode)
