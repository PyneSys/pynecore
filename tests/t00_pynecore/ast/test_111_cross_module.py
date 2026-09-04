"""
Calls into another module, and what the type pass may know about them.

MEASURED: TradingView compiles a library on its own, so a caller's argument
types never reach into it. An imported function is therefore typed from what it
DECLARES -- no per-call-site contexts across modules -- while an imported
overload GROUP is still pinned, because a pin selects among signatures and needs
no body to do it.

The other half of the mechanism is the bookkeeping: every interface a module's
types were derived from lands in ``table.deps``, the loader bakes those records
into the bytecode, and a dependency whose signatures moved invalidates exactly
the modules that could care.
"""
import ast
import importlib
import sys
from pathlib import Path

import pytest

from pynecore.core.import_hook import PyneLoader, _baked_deps, analyse_source
from pynecore.core.instance_state import _make_state  # noqa: internal API
from pynecore.transformers import pine_type_artifact
from pynecore.transformers.pine_type_artifact import interface_digest, registered
from pynecore.transformers.pine_type_rules import (
    FLOAT, INT, OBJECT, UNKNOWN, get_pin, get_ty, tuple_of,
)
from pynecore.transformers.pine_type_table import PineTypeTable

#: Name of the folded constant the loader bakes the dependency records into.
DEPS_CONST = '__pyne_type_deps__'


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-wide interface registry from leaking between tests."""
    pine_type_artifact._registry.clear()
    pine_type_artifact._analysing.clear()
    yield
    pine_type_artifact._registry.clear()
    pine_type_artifact._analysing.clear()


@pytest.fixture(autouse=True)
def _clean_modules():
    """Drop the modules a test imported, so a later one starts from the source."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith('xm_'):
            del sys.modules[name]


def _write(tmp_path: Path, name: str, source: str) -> Path:
    """Write a module under ``tmp_path`` and hand back its path."""
    path = tmp_path / f'{name.replace(".", "/")}.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
    return path


def _analysed(path: Path) -> tuple[ast.Module, PineTypeTable]:
    """Run the analysing half of the pipeline, cross-module resolution included."""
    analysed = analyse_source(str(path))
    assert analysed is not None, 'the module was not recognized as Pyne code'
    return analysed[0], analysed[1]


def _call(tree: ast.Module, callee: str) -> ast.Call:
    """The last call to one callee, as the analysed tree spells it."""
    found = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == callee]
    assert found, f'no call to {callee} in the analysed tree'
    return found[-1]


def _compile(path: Path):
    """Run the real pipeline over a module, the way an import would."""
    return PyneLoader(path.stem, str(path)).source_to_code(path.read_bytes(), str(path))


def _diag(table: PineTypeTable, reason: str):
    """The one diagnostic of a given reason, asserting there is exactly one."""
    found = [diag for diag in table.diags
             if diag.origin is not None and diag.origin.reason == reason]
    assert len(found) == 1, f'expected one {reason} diagnostic, got {[d.render() for d in found]}'
    return found[0]


# --- a plain annotated export ---------------------------------------------


PLAIN_LIB = '''"""
@pyne
"""


def helper(x: int) -> int:
    return x * 2


def loose(x) -> int:
    return x
'''


def __test_an_annotated_export_types_the_call__(tmp_path, monkeypatch):
    """A call into an imported module reads the export's declared return"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'xm_plain_lib', PLAIN_LIB)
    app = _write(tmp_path, 'xm_plain_app', '''"""
@pyne
"""
from xm_plain_lib import helper

value = helper(3)
''')

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper')) == INT
    interface = registered(str(lib))
    assert interface is not None, 'the dependency published no interface'
    record = table.deps[str(lib.resolve())]
    assert record.digest == interface_digest(interface)


def __test_the_dependency_records_are_baked_into_the_bytecode__(tmp_path, monkeypatch):
    """The loader folds the records into one constant, readable without importing"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'xm_baked_lib', PLAIN_LIB)
    app = _write(tmp_path, 'xm_baked_app', '''"""
@pyne
"""
from xm_baked_lib import helper

value = helper(3)
''')

    code = _compile(app)

    assert any(isinstance(const, tuple) and const and const[0] == DEPS_CONST
               for const in code.co_consts), 'no dependency constant was baked in'
    assert [record.path for record in _baked_deps(code)] == [str(lib.resolve())]


def __test_the_dotted_import_forms_resolve__(tmp_path, monkeypatch):
    """``import p.m as x`` and ``import p.m`` reach the same export"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'xm_alias_pkg.mylib', PLAIN_LIB)
    aliased = _write(tmp_path, 'xm_alias_app', '''"""
@pyne
"""
import xm_alias_pkg.mylib as m

value = m.helper(3)
''')
    full = _write(tmp_path, 'xm_full_app', '''"""
@pyne
"""
import xm_alias_pkg.mylib

value = xm_alias_pkg.mylib.helper(3)
''')

    alias_tree, alias_table = _analysed(aliased)
    full_tree, full_table = _analysed(full)

    assert get_ty(_call(alias_tree, 'm.helper')) == INT
    assert get_ty(_call(full_tree, 'xm_alias_pkg.mylib.helper')) == INT
    assert list(alias_table.deps) == [str(lib.resolve())]
    assert list(full_table.deps) == [str(lib.resolve())]


def __test_an_unannotated_export_is_unknown__(tmp_path, monkeypatch):
    """Without annotations the module was analysed without this call's types"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'xm_loose_lib', PLAIN_LIB)
    app = _write(tmp_path, 'xm_loose_app', '''"""
@pyne
"""
from xm_loose_lib import loose

value = loose(3)
''')

    tree, table = _analysed(app)

    node = _call(tree, 'loose')
    assert get_ty(node) == UNKNOWN
    assert get_pin(node) is None
    diag = _diag(table, 'unannotated-import')
    where = f'{lib.resolve()}:10'
    assert "'loose'" in diag.message and where in diag.message
    assert diag.fix == f"annotate the parameters of 'loose' in {where}"


def __test_a_deeper_attribute_path_is_unknown__(tmp_path, monkeypatch):
    """An attribute of an export is not an export, and is not followed"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_deep_lib', PLAIN_LIB)
    app = _write(tmp_path, 'xm_deep_app', '''"""
@pyne
"""
from xm_deep_lib import helper

value = helper.method(3)
''')

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper.method')) == UNKNOWN
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == ['unknown-call']


# --- the shadowed namespace ------------------------------------------------


def __test_a_shadowed_namespace_answers_member_by_member__(tmp_path, monkeypatch):
    """The library's exports win; every other member falls back to the builtin"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'xm_shadow_lib', '''"""
@pyne
"""
__all__ = ['doubled']


def doubled(x: int) -> int:
    return x * 2
''')
    app = _write(tmp_path, 'xm_shadow_app', '''"""
@pyne
"""
from pynecore.core.pine_import import shadowed_namespace
from pynecore.lib import ta
import xm_shadow_lib as library

merged = shadowed_namespace(library, ta)


def main():
    return merged.doubled(3), merged.sma(1.0, 3)
''')

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'merged.doubled')) == INT
    assert get_ty(_call(tree, 'merged.sma')) == FLOAT
    assert [call.callee for call in table.calls] == ['merged.doubled', 'ta.sma']
    assert list(table.deps) == [str(lib.resolve())]


# --- the compiled-library shape --------------------------------------------


PROXY_LIB = '''"""
@pyne
"""
from pynecore.core.pine_export import Exported, export

__all__ = ['scaled']

scaled = Exported()


def setup():
    @export
    def scaled(x: int, factor: int):
        return x * factor
'''


def __test_an_exported_proxy_publishes_its_nested_definition__(tmp_path, monkeypatch):
    """``X = Exported()`` takes its shape from the ``@export`` def nested in the body"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_proxy_lib', PROXY_LIB)
    app = _write(tmp_path, 'xm_proxy_app', '''"""
@pyne
"""
from xm_proxy_lib import scaled

value = scaled(3, 4)
''')

    tree, table = _analysed(app)

    # The nested def annotates both parameters and returns int * int
    assert get_ty(_call(tree, 'scaled')) == INT
    assert table.diags == []


# --- overload groups across modules ----------------------------------------


EXPORTED_GROUP_LIB = '''"""
@pyne
"""
from pynecore.core.overload import overload
from pynecore.core.pine_export import Exported, export

__all__ = ['pick']

pick = Exported()


def setup():
    @export
    @overload
    def pick(x: int) -> float:
        return 1.0

    @export
    @overload
    def pick(x: float) -> float:
        return 2.0
'''

GROUP_APP = '''"""
@pyne
"""
from {lib} import pick


def main(r: int):
    return pick(r / 8), pick(r * 1.0), pick(r + 1)
'''


def __test_an_exported_overload_group_is_pinned__(tmp_path, monkeypatch):
    """A group published by another module pins from the argument types"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_group_lib', EXPORTED_GROUP_LIB)
    app = _write(tmp_path, 'xm_group_app', GROUP_APP.format(lib='xm_group_lib'))

    tree, table = _analysed(app)

    pins = [call.pin for call in table.calls if call.callee == 'pick']
    # ``r / 8`` is int-TYPED while its value is fractional -- the whole point
    assert pins == [INT, None, INT]
    assert [call.ty for call in table.calls if call.callee == 'pick'] == [FLOAT] * 3
    assert get_pin(_call(tree, 'pick')) == INT


def __test_the_pin_reaches_the_binder_across_modules__(tmp_path, monkeypatch, capsys):
    """The uniform route carries the pin to the imported dispatcher"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_emit_lib', EXPORTED_GROUP_LIB)
    app = _write(tmp_path, 'xm_emit_app', GROUP_APP.format(lib='xm_emit_lib'))
    monkeypatch.setenv('PYNE_AST_DEBUG_RAW', str(app))

    _compile(app)

    dump = capsys.readouterr().out
    assert "__bind_any·__(__state__, 0, pick, 'i')" in dump
    assert '__bind_any·__(__state__, 1, pick)' in dump
    assert "__bind_any·__(__state__, 2, pick, 'i')" in dump


def __test_an_exported_group_dispatches_on_the_type_at_runtime__(tmp_path, monkeypatch):
    """The imported group runs the int implementation for an int-TYPED argument"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_run_lib', EXPORTED_GROUP_LIB)
    _write(tmp_path, 'xm_run_app', GROUP_APP.format(lib='xm_run_lib'))

    library = importlib.import_module('xm_run_lib')
    module = importlib.import_module('xm_run_app')
    library.setup()

    state = _make_state(module.__pyne_slot_layout__['main'])
    assert module.main(state, 14) == (1.0, 2.0, 1.0)


def __test_the_cross_module_pin_can_be_switched_off__(tmp_path, monkeypatch):
    """``PYNE_NO_TYPE_PIN=1`` dispatches from the values alone again"""
    monkeypatch.setenv('PYNE_NO_TYPE_PIN', '1')
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_nopin_lib', EXPORTED_GROUP_LIB)
    _write(tmp_path, 'xm_nopin_app', GROUP_APP.format(lib='xm_nopin_lib'))

    library = importlib.import_module('xm_nopin_lib')
    module = importlib.import_module('xm_nopin_app')
    library.setup()

    state = _make_state(module.__pyne_slot_layout__['main'])
    assert module.main(state, 14) == (2.0, 2.0, 1.0)


def __test_a_module_level_group_is_pinned_the_same_way__(tmp_path, monkeypatch):
    """A hand-written library's ``@overload`` group needs no proxy to be pinned"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_bare_lib', '''"""
@pyne
"""
from pynecore.core.overload import overload


@overload
def pick(x: int) -> int:
    return 1


@overload
def pick(x: float) -> float:
    return 2.0
''')
    app = _write(tmp_path, 'xm_bare_app', '''"""
@pyne
"""
from xm_bare_lib import pick


def main(r: int):
    return pick(r / 8), pick(r * 1.0)
''')

    tree, table = _analysed(app)

    pinned = [call for call in table.calls if call.callee == 'pick']
    assert [call.pin for call in pinned] == [INT, None]
    # The implementations disagree on their return, so only the pinned site
    # can say which one runs
    assert [call.ty for call in pinned] == [INT, UNKNOWN]
    assert get_pin(_call(tree, 'pick')) is None


# --- what cannot be resolved -----------------------------------------------


def __test_an_import_cycle_terminates_and_is_reported__(tmp_path, monkeypatch):
    """A -> B -> A ends in a diagnostic, not in a recursion"""
    monkeypatch.syspath_prepend(tmp_path)
    first = _write(tmp_path, 'xm_cycle_a', '''"""
@pyne
"""
from xm_cycle_b import beta


def alpha(x: int) -> int:
    return beta(x) + 1
''')
    second = _write(tmp_path, 'xm_cycle_b', '''"""
@pyne
"""
from xm_cycle_a import alpha


def beta(x: int) -> int:
    return alpha(x) + 1
''')

    # The inner analysis is where the cycle is noticed, and it is the module's
    # INTERFACE that comes back from it -- so the table is captured on the way
    tables: dict[str, PineTypeTable] = {}
    original = analyse_source

    def capture(path: str):
        result = original(path)
        if result is not None:
            tables[str(Path(path).resolve())] = result[1]
        return result

    monkeypatch.setattr('pynecore.core.import_hook.analyse_source', capture)

    tree, table = _analysed(first)

    # The direction that got there first resolves; the other one is the cycle
    assert get_ty(_call(tree, 'beta')) == INT
    assert list(table.deps) == [str(second.resolve())]
    inner = tables[str(second.resolve())]
    diag = _diag(inner, 'import-cycle')
    assert diag.fix == f'break the import cycle between {second.resolve()} and {first.resolve()}'
    assert not [call for call in inner.calls if call.callee == 'alpha']

    # Both modules still go through the whole pipeline
    assert _compile(first) is not None
    assert _compile(second) is not None


def __test_a_reassigned_import_is_unknown__(tmp_path, monkeypatch):
    """An imported name the module also assigns to calls something else"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_rebind_lib', PLAIN_LIB)
    app = _write(tmp_path, 'xm_rebind_app', '''"""
@pyne
"""
from xm_rebind_lib import helper

value = helper(3)
helper = loose
''')

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper')) == UNKNOWN
    diag = _diag(table, 'rebound-name')
    assert "'helper'" in diag.message
    assert diag.fix == "call 'helper' under a name nothing assigns to"


def __test_a_twice_imported_name_is_unknown__(tmp_path, monkeypatch):
    """One name, two imports: which module it reaches is not a static question"""
    # The import map holds ONE entry per bound name, so the second statement
    # simply overwrites the first -- and picking either would type the call
    # against a module the run may never reach.
    monkeypatch.syspath_prepend(tmp_path)
    first = _write(tmp_path, 'xm_twice_a', PLAIN_LIB)
    second = _write(tmp_path, 'xm_twice_b', PLAIN_LIB.replace('-> int', '-> float'))
    app = _write(tmp_path, 'xm_twice_app', """\"\"\"
@pyne
\"\"\"
from xm_twice_a import helper
from xm_twice_b import helper

value = helper(3)
""")

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper')) == UNKNOWN
    diag = _diag(table, 'rebound-name')
    assert diag.message == ("'helper' is imported more than once, so what it calls is "
                            "unknown")
    assert diag.fix == "import 'helper' once, under a name nothing else binds"
    # Nothing was consulted, so nothing is depended on -- neither module's
    # signatures decided anything here
    assert table.deps == {}
    assert first.exists() and second.exists()


def __test_two_imports_in_exclusive_branches_are_unknown_too__(tmp_path, monkeypatch):
    """A guarded pair of imports is two bindings, whichever one runs"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_guard_a', PLAIN_LIB)
    _write(tmp_path, 'xm_guard_b', PLAIN_LIB.replace('-> int', '-> float'))
    app = _write(tmp_path, 'xm_guard_app', """\"\"\"
@pyne
\"\"\"
try:
    from xm_guard_a import helper
except ImportError:
    from xm_guard_b import helper

value = helper(3)
""")

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper')) == UNKNOWN
    assert _diag(table, 'rebound-name').fix == \
        "import 'helper' once, under a name nothing else binds"
    assert table.deps == {}


def __test_one_import_inside_a_branch_still_types_the_call__(tmp_path, monkeypatch):
    """An ``if`` at module level opens no scope, and one import is still one import"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'xm_branch_lib', PLAIN_LIB)
    app = _write(tmp_path, 'xm_branch_app', """\"\"\"
@pyne
\"\"\"
if True:
    from xm_branch_lib import helper

value = helper(3)
""")

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper')) == INT
    assert table.diags == []
    assert list(table.deps) == [str(lib.resolve())]


def __test_a_function_level_import_is_opaque__(tmp_path, monkeypatch):
    """A local import binds a local, and says nothing about the module"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_local_lib', PLAIN_LIB)
    app = _write(tmp_path, 'xm_local_app', '''"""
@pyne
"""


def main():
    from xm_local_lib import helper
    return helper(3)
''')

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper')) == UNKNOWN
    # Unresolved is unresolved: the call is reported as one nothing types
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == ['unknown-call']
    assert table.deps == {}


def __test_a_nearer_binding_wins_over_the_import__(tmp_path, monkeypatch):
    """A parameter of the enclosing scope holds whatever was passed to it"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_param_lib', PLAIN_LIB)
    app = _write(tmp_path, 'xm_param_app', '''"""
@pyne
"""
from xm_param_lib import helper


def main(helper):
    return helper(3)
''')

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper')) == UNKNOWN
    # Unresolved is unresolved: the call is reported as one nothing types
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == ['unknown-call']


def __test_a_relative_import_is_not_resolved__(tmp_path, monkeypatch):
    """A relative import names a package this pass has no anchor for"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_rel_pkg.sibling', PLAIN_LIB)
    app = _write(tmp_path, 'xm_rel_pkg.consumer', '''"""
@pyne
"""
from .sibling import helper

value = helper(3)
''')

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper')) == UNKNOWN
    assert table.deps == {}
    # Unresolved is unresolved: the call is reported as one nothing types
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == ['unknown-call']


def __test_a_stdlib_import_records_no_dependency__(tmp_path, monkeypatch):
    """The standard library publishes no Pine interface, so it is left alone"""
    monkeypatch.syspath_prepend(tmp_path)
    app = _write(tmp_path, 'xm_std_app', '''"""
@pyne
"""
import math

value = math.floor(1.5)
''')

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'math.floor')) == UNKNOWN
    assert table.deps == {}
    # Unresolved is unresolved: the call is reported as one nothing types
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == ['unknown-call']


def __test_a_non_pyne_dependency_is_left_unknown__(tmp_path, monkeypatch):
    """A plain Python module has no interface, and nothing is guessed for it"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_foreign_lib', '''def helper(x: int) -> int:
    return x * 2
''')
    app = _write(tmp_path, 'xm_foreign_app', '''"""
@pyne
"""
from xm_foreign_lib import helper

value = helper(3)
''')

    tree, table = _analysed(app)

    assert get_ty(_call(tree, 'helper')) == UNKNOWN
    assert table.deps == {}
    # Unresolved is unresolved: the call is reported as one nothing types
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == ['unknown-call']


# --- the object-returning shapes -------------------------------------------


def __test_an_object_return_comes_back_as_the_export_says__(tmp_path, monkeypatch):
    """Whatever the export returns is what the call site gets, tuples included"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_tuple_lib', '''"""
@pyne
"""


def pair(x: int):
    return x, x + 1
''')
    app = _write(tmp_path, 'xm_tuple_app', '''"""
@pyne
"""
from xm_tuple_lib import pair

value = pair(3)
''')

    tree, table = _analysed(app)

    # The return is INFERRED from the body -- the module publishes what its
    # own analysis found, and a tuple travels like any other shape
    assert get_ty(_call(tree, 'pair')) == tuple_of([INT, INT])
    assert table.diags == []


SHAPED_LIB = '''"""
@pyne
"""


def add(a: int, b: int = 0) -> int:
    return a + b
'''

SHAPED_APP = '''"""
@pyne
"""
from xm_shaped_lib import add


def main():
    x = add(1)
    y = add(1, 2, 3)
    z = add(1, nosuch=2)
    w = add(1, 2.5)
    v = add(1, b=2)
    u = add(1, a=2)
    return x + y + z + w + v + u
'''


def __test_a_call_into_another_module_is_held_to_its_signature__(tmp_path, monkeypatch):
    """The published shape -- arity, names, types -- is what the call has to meet"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_shaped_lib', SHAPED_LIB)
    app = _write(tmp_path, 'xm_shaped_app', SHAPED_APP)

    _, table = _analysed(app)

    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['x'] == INT and types['v'] == INT
    assert all(types[name] == UNKNOWN for name in ('y', 'z', 'w', 'u'))
    reasons = [diag.origin.reason for diag in table.diags if diag.origin is not None]
    assert reasons == ['bad-call'] * 4
    messages = [diag.message for diag in table.diags]
    assert "'add' does not take 3 argument(s)" in messages[0]
    assert "'add' has no parameter 'nosuch'" in messages[1]
    assert "'add' takes int for 'b', float passed" in messages[2]
    assert "'a' is passed to 'add' twice" in messages[3]


METHOD_LIB = '''"""
@pyne
"""
from pynecore.core.pine_method import method
from pynecore.core.pine_udt import udt

__all__ = ['Cfg', 'bump']


@udt
class Cfg:
    level: int
    tag: str = "x"


@method
def bump(this: Cfg, by: int) -> int:
    return this.level + by
'''

METHOD_APP = '''"""
@pyne
"""
from pynecore.core.pine_method import method_call
import xm_method_lib as ordlib


def main():
    c = ordlib.Cfg.new(1)
    a = method_call('bump', c)
    b = method_call('bump', c, 1, 2)
    d = method_call('bump', c, "x")
    e = method_call('bump', c, 1)
    f = method_call(ordlib.bump, c, 2.5)
    return a
'''


def __test_a_method_of_an_imported_class_is_held_to_its_signature__(tmp_path, monkeypatch):
    """The receiver is the first parameter; the rest has to fit what the module publishes"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_method_lib', METHOD_LIB)
    app = _write(tmp_path, 'xm_method_app', METHOD_APP)

    _, table = _analysed(app)

    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['e'] == INT
    assert all(types[name] == UNKNOWN for name in ('a', 'b', 'd', 'f'))
    reasons = [diag.origin.reason for diag in table.diags if diag.origin is not None]
    assert reasons == ['bad-call'] * 4
    messages = [diag.message for diag in table.diags]
    assert "'bump' does not take 1 argument(s)" in messages[0]
    assert "'bump' does not take 3 argument(s)" in messages[1]
    assert "'bump' takes int for 'by', string passed" in messages[2]
    assert "takes int for 'by', float passed" in messages[3]


BARE_CLASS_APP = '''"""
@pyne
"""
from xm_method_lib import Cfg


def main():
    s = Cfg.new(1)
    t = Cfg(1, "a", 3)
    return s
'''


def __test_a_class_imported_by_name_is_a_type__(tmp_path, monkeypatch):
    """``from lib.m import Cfg`` binds the class itself, constructor and all"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'xm_method_lib', METHOD_LIB)
    app = _write(tmp_path, 'xm_bare_class_app', BARE_CLASS_APP)

    _, table = _analysed(app)

    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['s'].startswith('o:') and types['s'].endswith('#Cfg')
    assert types['t'] == UNKNOWN
    reasons = [diag.origin.reason for diag in table.diags if diag.origin is not None]
    assert reasons == ['bad-call']
    assert "'Cfg' has 2 field(s), 3 argument(s) passed" in table.diags[0].message
