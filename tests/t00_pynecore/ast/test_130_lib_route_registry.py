"""
The transform reads the lib's routing facts from the generated registry.

``FunctionIsolationTransformer`` routes a ``pynecore.lib`` callee by the
``routes`` section of ``lib_types.json``, and ``ImportNormalizerTransformer``
expands a lib ``import *`` by its ``star_exports`` section -- neither imports
and inspects the live lib. The registry is hashed into the pipeline digest and
the lib sources are not, so this is what makes a cached script's bytecode
follow a lib edit (through the regeneration ``test_104`` enforces).

The tests below change the LIVE lib object and check that the emission does not
move, then change the registry and check that it does.
"""
import ast
from pathlib import Path

import pynecore
import pynecore.lib
import pynecore.lib.ta
from pynecore.core.import_hook import source_starts_with_pyne
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
from pynecore.transformers.pine_type_infer import lib_routes, lib_star_exports
from pynecore.transformers.slot_layout import ModuleLayout

_ISOLATED = '''
from pynecore import lib

def main():
    a = lib.ta.mom(lib.close, 2)
    b = lib.ta.sma(lib.close, 3)
    c = lib.ta.ema(lib.close, 4)
    return a + b + c
'''

_STAR = '''
from pynecore.lib import *

def main():
    return hl2
'''


def _isolate(source: str) -> str:
    """Run the call-site isolation alone and unparse the result."""
    return ast.unparse(FunctionIsolationTransformer(ModuleLayout()).visit(ast.parse(source)))


def _normalize(source: str) -> str:
    """Run the import normalization alone and unparse the result."""
    return ast.unparse(ImportNormalizerTransformer().visit(ast.parse(source)))


def __test_lib_routes_ignore_the_live_object__(monkeypatch):
    """A lib callee keeps its registry route whatever the imported object is now"""
    def stateless(source, length):
        return source

    monkeypatch.setattr(pynecore.lib.ta, 'mom', stateless)
    code = _isolate(_ISOLATED)
    assert '__resolve_slot·__(__state__, 0, lib.ta.mom)' in code


def __test_lib_routes_follow_the_registry__(monkeypatch):
    """Changing the registry entry is what changes the emitted call"""
    monkeypatch.setitem(lib_routes(), 'lib.ta.sma', 'direct')
    monkeypatch.delitem(lib_routes(), 'lib.ta.ema')
    code = _isolate(_ISOLATED)
    assert 'b = lib.ta.sma(lib.close, 3)' in code
    assert '__bind_any·__(__state__, 1, lib.ta.ema)' in code


def __test_star_import_ignores_the_live_all__(monkeypatch):
    """A lib ``import *`` expands to the registry's ``__all__``, not the live one"""
    monkeypatch.setattr(pynecore.lib, '__all__',
                        [name for name in pynecore.lib.__all__ if name != 'hl2'])
    assert 'return lib.hl2' in _normalize(_STAR)

    monkeypatch.setitem(lib_star_exports(), 'pynecore.lib',
                        [name for name in lib_star_exports()['pynecore.lib'] if name != 'hl2'])
    assert 'return hl2' in _normalize(_STAR)


def __test_no_pyne_module_outside_the_lib__():
    """
    A transformed module outside ``pynecore/lib`` would need routes of its own.

    The registry covers ``pynecore.lib`` only; every other ``pynecore`` callee is
    still inspected at transform time. That is safe because only a ``@pyne``
    module yields a state-carrying or a provably stateless route, and they all
    live under ``lib``: anywhere else the route can only be skip or uniform, and
    both are correct whatever a cached script was compiled against.
    """
    root = Path(pynecore.__file__).parent
    pyne = [str(path.relative_to(root)) for path in sorted(root.rglob('*.py'))
            if 'lib' not in path.relative_to(root).parts[:1]
            and source_starts_with_pyne(path.read_bytes()[:4096])]
    assert not pyne, (f"@pyne modules outside pynecore/lib: {pyne} -- record their callees in "
                      "scripts/lib_type_collector.py and route them from the registry")
