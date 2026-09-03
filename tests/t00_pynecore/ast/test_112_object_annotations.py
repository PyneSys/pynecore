"""
A user class is a type, so an annotation naming one is an annotation.

A UDT is what a Pine script declares and a library publishes, and a parameter
typed by one is fully annotated. Reading such a name as UNKNOWN made the
parameter behave like an unannotated one: the body lost the type, the export
was published as ``annotated=False``, and every caller in another module was
told the return was unknowable -- for a signature that spells its types out.

Which names ARE classes cannot be read off the annotation, so the set is
collected first: every class the module declares, in any scope, plus the ones
its imports bring in -- resolved through the same published interface the
calls go through, which is why the interface carries them and the digest
covers them.
"""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

from pynecore.core.import_hook import PIPELINE_DIGEST, analyse_source
from pynecore.transformers import pine_type_artifact
from pynecore.transformers.pine_type_artifact import (
    build_interface, interface_digest, registered, table_json, _interface_from_json,
)
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import (
    INT, OBJECT, UNKNOWN, annotation_type, class_id, get_ty, object_ty,
)
from pynecore.transformers.pine_type_table import ModuleInterface, PineTypeTable


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
        if name.startswith('ob_'):
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


def _interface(tmp_path: Path, name: str, source: str) -> ModuleInterface:
    """Analyse a source and build the interface it publishes."""
    tree, table = _analysed(_write(tmp_path, name, source))
    return build_interface(tree, table, str((tmp_path / f'{name}.py').resolve()))


def _call(tree: ast.Module, callee: str) -> ast.Call:
    """The last call to one callee, as the analysed tree spells it."""
    found = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == callee]
    assert found, f'no call to {callee} in the analysed tree'
    return found[-1]


def _param_type(source: str, scope: str = 'take', name: str = 'thing') -> str:
    """Infer a snippet and read back the type one parameter was bound with."""
    table = infer_module(ast.parse(source), 'test')
    binding = table.binding(scope, name)
    assert binding is not None, f"'{name}' was never bound in scope '{scope}'"
    return binding.ty


# --- the classes a module declares itself ---------------------------------


PREAMBLE = '''from pynecore.types import NA, Persistent, Series


class Pivot:
    price: float = 0.0
'''

#: What a ``Pivot`` of a module analysed under the path ``'test'`` is typed as.
#: The id is (module, name), so the module key is part of every expectation
#: here -- two modules' same-named classes are two different types.
PIVOT = object_ty(class_id('test', 'Pivot'))


@pytest.mark.parametrize('annotation,expected', [
    # The bare name, and the wrappers that change the storage and not the type
    ('Pivot', PIVOT),
    ('Series[Pivot]', PIVOT),
    ('Persistent[Pivot]', PIVOT),
    ('NA[Pivot]', PIVOT),
    # A stringized forward reference is resolved the same way
    ("'Pivot'", PIVOT),
    ("'Series[Pivot]'", PIVOT),
    # ``NA`` is an absence marker, so the union keeps the class
    ('Pivot | NA', PIVOT),
    ('Pivot | None', PIVOT),
    # A container carries the class as its ELEMENT
    ('list[Pivot]', f'a:{PIVOT}'),
    ('Series[list[Pivot]]', f'a:{PIVOT}'),
    # A name that is not a class is still nothing this pass can read
    ('Missing', UNKNOWN),
    ('Series[Missing]', UNKNOWN),
    # ... and the scalars are untouched
    ('int', INT),
])
def __test_a_class_annotation_is_an_object__(annotation: str, expected: str):
    """A parameter typed by a UDT is annotated, not unannotated"""
    assert _param_type(f'{PREAMBLE}\n\ndef take(thing: {annotation}):\n    return thing\n') \
        == expected


def __test_a_class_declared_below_its_use_still_types_it__():
    """The classes are collected before anything reads an annotation"""
    source = '''def take(thing: Later):
    return thing


class Later:
    pass
'''

    assert _param_type(source) == object_ty(class_id('test', 'Later'))


def __test_a_nested_class_is_nameable_from_the_body_it_lives_in__():
    """A class declared inside a function is a class where it stands"""
    source = '''def outer():
    class Inner:
        pass

    def take(thing: Inner):
        return thing

    return take
'''

    assert _param_type(source, scope='outer·take') == object_ty(class_id('test', 'Inner'))


def __test_a_rebound_class_name_is_no_class__():
    """What stands under the name at import time is the assignment, not the class"""
    # ``Amount`` is a class statement AND a module-level store, so an
    # annotation naming it describes whatever the store put there -- reading
    # it as an object would type against a class nothing holds.
    source = """class Amount:
    pass


Amount = int


def take(thing: Amount):
    return thing
"""
    table = infer_module(ast.parse(source), 'test')

    assert 'Amount' not in table.classes
    assert _param_type(source) == UNKNOWN


def __test_a_class_the_module_only_declares_stays_a_class__():
    """The comparison is by position: the class statement is a binding too"""
    source = """class Amount:
    pass


def take(thing: Amount):
    return thing
"""
    assert _param_type(source) == object_ty(class_id('test', 'Amount'))


def __test_an_annotated_class_variable_is_an_object__():
    """An explicit annotation is a declaration, whatever it declares"""
    table = infer_module(ast.parse(f'{PREAMBLE}\n\nlast: Pivot = None\n'), 'test')
    binding = table.binding('', 'last')

    assert binding is not None and binding.ty == PIVOT


def __test_the_module_records_the_classes_it_can_name__():
    """The set the annotations are read against travels on the table"""
    table = infer_module(ast.parse(f'{PREAMBLE}\n\ndef take(thing: Pivot):\n    return thing\n'),
                         'test')

    assert 'Pivot' in table.classes


def __test_the_rules_answer_unknown_without_a_class_set__():
    """The names are an input, not a guess: nothing is a class by its spelling"""
    annotation = ast.parse('Pivot', mode='eval').body

    assert annotation_type(annotation) == UNKNOWN
    assert annotation_type(annotation, {'Pivot': 'm#Pivot'}) == 'o:m#Pivot'


# --- the classes an import brings in --------------------------------------


UDT_LIB = '''"""
@pyne
"""
from pynecore.types import Series

__all__ = ['Settings', 'newInstance', 'depth']


class Settings:
    depth: int = 10


def newInstance(settings: Series[Settings]) -> Settings:
    return settings


def depth(settings: Series[Settings]) -> int:
    return settings.depth
'''


def __test_a_from_import_brings_the_class_in__(tmp_path, monkeypatch):
    """``from m import C`` makes ``C`` a class name in the importing module"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'ob_from_lib', UDT_LIB)
    app = _write(tmp_path, 'ob_from_app', '''"""
@pyne
"""
from ob_from_lib import Settings


def take(thing: Settings):
    return thing
''')

    _, table = _analysed(app)

    # The identity travels with the class: the importing module types the
    # parameter as the DEPENDENCY's ``Settings``, not as one of its own.
    imported = object_ty(class_id(str(lib.resolve()), 'Settings'))
    assert table.classes['Settings'] == class_id(str(lib.resolve()), 'Settings')
    binding = table.binding('take', 'thing')
    assert binding is not None and binding.ty == imported


def __test_a_module_import_brings_its_classes_in__(tmp_path, monkeypatch):
    """``import m`` then ``m.C``: the attribute tail is resolved the same way"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'ob_dotted_lib', UDT_LIB)
    app = _write(tmp_path, 'ob_dotted_app', '''"""
@pyne
"""
import ob_dotted_lib as m
from pynecore.types import Series


def take(thing: Series[m.Settings]):
    return thing
''')

    _, table = _analysed(app)

    binding = table.binding('take', 'thing')
    assert binding is not None
    assert binding.ty == object_ty(class_id(str(lib.resolve()), 'Settings'))


def __test_an_imported_name_that_is_no_class_stays_unknown__(tmp_path, monkeypatch):
    """Only what the dependency PUBLISHES as a class answers for one"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'ob_nocls_lib', UDT_LIB)
    app = _write(tmp_path, 'ob_nocls_app', '''"""
@pyne
"""
from ob_nocls_lib import depth


def take(thing: depth):
    return thing
''')

    _, table = _analysed(app)

    assert 'depth' not in table.classes
    binding = table.binding('take', 'thing')
    assert binding is not None and binding.ty == UNKNOWN


def __test_a_class_annotated_export_types_the_call__(tmp_path, monkeypatch):
    """The whole point: a ``Series[Settings]`` parameter no longer hides a return"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'ob_call_lib', UDT_LIB)
    app = _write(tmp_path, 'ob_call_app', '''"""
@pyne
"""
from ob_call_lib import Settings, newInstance


def build(settings: Settings):
    return newInstance(settings)
''')

    tree, table = _analysed(app)

    interface = registered(str(lib))
    assert interface is not None, 'the dependency published no interface'
    assert interface.exports['newInstance'].annotated is True
    assert get_ty(_call(tree, 'newInstance')) == object_ty(
        class_id(str(lib.resolve()), 'Settings'))
    assert [diag.render() for diag in table.diags
            if diag.origin is not None and diag.origin.reason == 'unannotated-import'] == []


# --- what the interface publishes -----------------------------------------


CLASS_BASE = '''"""
@pyne
"""
__all__ = ['Settings', 'area']


class Settings:
    depth: int = 10


def area(width: int, height: int) -> int:
    return width * height
'''


def __test_the_interface_publishes_the_classes__(tmp_path):
    """Module level and ``__all__``-filtered, the way the exports are"""
    interface = _interface(tmp_path, 'ob_pub_mod', CLASS_BASE)

    assert list(interface.classes) == ['Settings']
    # A published class is its fields too -- a dependent reads ``s.depth`` as
    # INT only because the interface carries the field's declared type.
    assert interface.classes['Settings'].fields == {'depth': INT}


def __test_an_unlisted_class_is_not_published__(tmp_path):
    """A namespace import reads the module through ``__all__``"""
    interface = _interface(tmp_path, 'ob_hidden_mod',
                           CLASS_BASE + '\n\nclass Hidden:\n    pass\n')

    assert list(interface.classes) == ['Settings']


def __test_adding_a_class_moves_the_digest__(tmp_path):
    """A dependent's annotations resolve against these, so they are in the contract"""
    base = _interface(tmp_path, 'ob_digest_base', CLASS_BASE)
    added = _interface(tmp_path, 'ob_digest_added', CLASS_BASE.replace(
        "__all__ = ['Settings', 'area']", "__all__ = ['Settings', 'Extra', 'area']")
        + '\n\nclass Extra:\n    pass\n')

    assert base.digest != added.digest


def __test_the_artifact_round_trips_the_classes__(tmp_path):
    """What the JSON carries is what a later process reads back"""
    path = _write(tmp_path, 'ob_json_mod', CLASS_BASE)
    tree, table = _analysed(path)
    interface = build_interface(tree, table, str(path.resolve()))

    data = json.loads(json.dumps(
        table_json(tree, table, interface, path.read_bytes(), PIPELINE_DIGEST)))

    assert list(data['interface']['classes']) == ['Settings']
    assert data['interface']['classes']['Settings']['fields'] == {'depth': INT}
    stat = os.stat(path)
    restored = _interface_from_json(interface.path, data, (stat.st_mtime_ns, stat.st_size))
    assert list(restored.classes) == ['Settings']
    assert restored.classes['Settings'] == interface.classes['Settings']
    assert restored.digest == interface.digest
