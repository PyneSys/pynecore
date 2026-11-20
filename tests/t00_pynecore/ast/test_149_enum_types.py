"""
A Pine ``enum`` is a type, and a member of it is a value of that type.

PyneComp writes an enum as ``class Speed(Enum)`` over ``from enum import
StrEnum as Enum``, each member holding its title. The class is a type like a
UDT: an annotation naming it is an object of it, and so is a member read
through the class (``Speed.fast``), the default an ``input.enum`` is given and
what that input holds. A library publishes its enums with its other classes.
"""
import ast
import json
import os
import sys
import zlib
from pathlib import Path

import pytest

from ..pine_analysis import analyse_module
from pynecore.errorcodes import TYPE_NO_FIELD, TYPE_UNKNOWN_LIB_CALL
from pynecore.transformers import module_interface
from pynecore.transformers.module_interface import (
    build_interface, interface_from_payload, interface_payload,
)
from pynecore.transformers.pine_qualifier import CONST
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import (
    BOOL, FLOAT, INT, OBJECT, UNKNOWN, array_of, class_id, get_ty, key_of, map_of, object_ty,
    render_ty, value_of,
)
from pynecore.transformers.pine_type_table import PineTypeTable


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-wide interface registry from leaking between tests."""
    module_interface._registry.clear()
    module_interface._analysing.clear()
    yield
    module_interface._registry.clear()
    module_interface._analysing.clear()


@pytest.fixture(autouse=True)
def _clean_modules():
    """Drop the modules a test imported, so a later one starts from the source."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith('en_'):
            del sys.modules[name]


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f'{name}.py'
    path.write_text(source, encoding='utf-8')
    return path


def _analysed(path: Path) -> tuple[ast.Module, PineTypeTable]:
    analysed = analyse_module(str(path))
    assert analysed is not None, 'the module was not recognized as Pyne code'
    return analysed[0], analysed[1]


def _binding(table: PineTypeTable, scope: str, name: str) -> str:
    binding = table.binding(scope, name)
    assert binding is not None, f"'{name}' was never bound in scope '{scope}'"
    return binding.ty


ENUM = '''from enum import StrEnum as Enum
from pynecore import lib


class Speed(Enum):
    fast = 'Fast'
    slow = 'Slow'
'''

#: What a ``Speed`` of a module analysed under the path ``'test'`` is typed as.
SPEED = object_ty(class_id('test', 'Speed'))


# --- the enum a module declares itself -------------------------------------


def __test_an_enum_is_a_class_whose_fields_are_its_members__():
    table = infer_module(ast.parse(ENUM), 'test')

    sig = table.class_sigs[class_id('test', 'Speed')]
    assert sig.enum
    assert sig.fields == {'fast': SPEED, 'slow': SPEED}
    assert sig.required == 0


def __test_a_class_without_the_enum_base_is_no_enum__():
    table = infer_module(ast.parse('class Pivot:\n    price: float = 0.0\n'), 'test')

    assert not table.class_sigs[class_id('test', 'Pivot')].enum


@pytest.mark.parametrize('body, name, expected', [
    # A member read through the class is a value of the enum
    ('def main():\n    x = Speed.fast\n    return x\n', 'x', SPEED),
    # ... which an annotation naming the enum agrees with
    ('def main():\n    x: Speed = Speed.slow\n    return x\n', 'x', SPEED),
    # Two values of it compare like any others
    ('def main():\n    x = Speed.fast == Speed.slow\n    return x\n', 'x', BOOL),
    # An enum input holds a member of its default's enum, however it is spelled
    ('def main():\n    x = lib.input.enum(Speed.fast, "Speed")\n    return x\n', 'x', SPEED),
    ('def main():\n    x = lib.input.enum(title="Speed", defval=Speed.fast)\n    return x\n', 'x', SPEED),
])
def __test_a_member_is_a_value_of_its_enum__(body: str, name: str, expected: str):
    table = infer_module(ast.parse(ENUM + body), 'test')

    assert _binding(table, 'main', name) == expected


def __test_a_member_is_a_constant__():
    tree = ast.parse(ENUM + 'def main():\n    x = Speed.fast\n    return x\n')
    table = infer_module(tree, 'test')

    binding = table.binding('main', 'x')
    assert binding is not None and binding.qual == CONST


def __test_a_member_the_enum_does_not_declare_is_reported__():
    tree = ast.parse(ENUM + 'def main():\n    x = Speed.medium\n    return x\n')
    table = infer_module(tree, 'test')

    assert _binding(table, 'main', 'x') == UNKNOWN
    assert [diag.code for diag in table.diags] == [TYPE_NO_FIELD.code]
    assert table.diags[0].message == TYPE_NO_FIELD.render('Speed', 'medium')


def __test_a_value_bound_under_the_enums_name_is_no_enum__():
    """A parameter named like the enum shadows it: the read is a field of the value"""
    tree = ast.parse(ENUM + 'def main(Speed: int):\n    x = Speed.fast\n    return x\n')
    table = infer_module(tree, 'test')

    assert _binding(table, 'main', 'x') == UNKNOWN


# --- an enum as a map key --------------------------------------------------


#: Every map call over a map keyed by ``Speed``, and a ``for [k, v] in m`` loop
ENUM_MAP = ENUM + '''

def main():
    levels: dict[Speed, float] = lib.map.new()
    put = lib.map.put(levels, Speed.fast, 1.5)
    got = lib.map.get(levels, Speed.slow)
    has = lib.map.contains(levels, Speed.fast)
    gone = lib.map.remove(levels, Speed.fast)
    keys = lib.map.keys(levels)
    values = lib.map.values(levels)
    size = lib.map.size(levels)
    copied = lib.map.copy(levels)
    lib.map.put_all(copied, levels)
    lib.map.clear(levels)
    for key, value in levels.items():
        k = key
        v = value
    return got
'''


def __test_an_enum_keys_a_map__():
    """Pine takes an enum as a map key: the map, its keys and its values stay typed"""
    table = infer_module(ast.parse(ENUM_MAP), 'test')

    levels = map_of(SPEED, FLOAT, frozenset({class_id('test', 'Speed')}))
    assert key_of(levels) == SPEED and value_of(levels) == FLOAT
    assert render_ty(levels) == 'map<Speed, float>'
    expected = {
        'levels': levels, 'put': FLOAT, 'got': FLOAT, 'has': BOOL, 'gone': FLOAT,
        'keys': array_of(SPEED), 'values': array_of(FLOAT), 'size': INT, 'copied': levels,
        'k': SPEED, 'v': FLOAT,
    }
    assert {name: _binding(table, 'main', name) for name in expected} == expected
    assert table.enums == {class_id('test', 'Speed')}
    assert not table.diags, [diag.message for diag in table.diags]


def __test_a_key_of_another_enum_does_not_fit__():
    """A map keyed by one enum takes no member of another as its key"""
    source = ENUM + '''

class Side(Enum):
    left = 'Left'


def main():
    levels: dict[Speed, float] = lib.map.new()
    lib.map.put(levels, Side.left, 1.5)
    return levels
'''
    table = infer_module(ast.parse(source), 'test')

    assert [diag.origin.reason for diag in table.diags if diag.origin is not None] \
        == ['shape-mismatch']


@pytest.mark.parametrize('key', ['Pivot', 'lib.Line', 'list[Speed]'])
def __test_a_class_that_is_no_enum_keys_no_map__(key: str):
    """A UDT, a drawing or a container is no Pine map key: the map is a bare object"""
    source = ENUM + f'''
from pynecore.core.pine_udt import udt


@udt
class Pivot:
    price: float = 0.0


def main():
    levels: dict[{key}, float] = lib.map.new()
    got = lib.map.get(levels, Speed.fast)
    return got
'''
    table = infer_module(ast.parse(source), 'test')

    assert _binding(table, 'main', 'levels') == OBJECT
    assert _binding(table, 'main', 'got') == UNKNOWN
    assert [diag.code for diag in table.diags] == [TYPE_UNKNOWN_LIB_CALL.code]
    assert table.diags[0].message == TYPE_UNKNOWN_LIB_CALL.render('map.get')


# --- the enum a library publishes ------------------------------------------


ENUM_LIB = '''"""
@pyne
"""
from enum import StrEnum as Enum

__all__ = ['Zone']


class Zone(Enum):
    exchange = 'Exchange'
    utc = 'UTC'
'''


def __test_a_library_enum_is_read_through_its_namespace__(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'en_ns_lib', ENUM_LIB)
    app = _write(tmp_path, 'en_ns_app', '''"""
@pyne
"""
import en_ns_lib as chrono
from pynecore.lib import input


def main():
    x = chrono.Zone.exchange
    y = input.enum(defval=chrono.Zone.utc, title='Zone')
    z: chrono.Zone = y
    return x == z
''')

    tree, table = _analysed(app)

    zone = object_ty(class_id(str(lib.resolve()), 'Zone'))
    assert _binding(table, 'main', 'x') == zone
    assert _binding(table, 'main', 'y') == zone
    assert _binding(table, 'main', 'z') == zone
    assert not table.diags
    returned = next(node for node in ast.walk(tree) if isinstance(node, ast.Return))
    assert returned.value is not None and get_ty(returned.value) == BOOL


def __test_a_qualified_enum_member_named_new_is_no_constructor__(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'en_new_lib', ENUM_LIB.replace(
        "    utc = 'UTC'\n", "    utc = 'UTC'\n    new = 'New'\n"))
    app = _write(tmp_path, 'en_new_app', '''"""
@pyne
"""
import en_new_lib as chrono
from pynecore.lib import input


def main():
    x = chrono.Zone.new
    y = input.enum(defval=chrono.Zone.new, title='Zone')
    return x == y
''')

    _, table = _analysed(app)

    zone = object_ty(class_id(str(lib.resolve()), 'Zone'))
    assert _binding(table, 'main', 'x') == zone
    assert _binding(table, 'main', 'y') == zone
    assert not table.diags


def __test_a_qualified_enum_without_a_new_member_reports_it__(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'en_nonew_lib', ENUM_LIB)
    app = _write(tmp_path, 'en_nonew_app', '''"""
@pyne
"""
import en_nonew_lib as chrono


def main():
    x = chrono.Zone.new
    return x
''')

    _, table = _analysed(app)

    assert _binding(table, 'main', 'x') == UNKNOWN
    assert [diag.code for diag in table.diags] == [TYPE_NO_FIELD.code]


def __test_a_library_enum_is_read_through_a_from_import__(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'en_from_lib', ENUM_LIB)
    app = _write(tmp_path, 'en_from_app', '''"""
@pyne
"""
from en_from_lib import Zone


def main():
    x = Zone.utc
    return x
''')

    _, table = _analysed(app)

    assert _binding(table, 'main', 'x') == object_ty(class_id(str(lib.resolve()), 'Zone'))
    assert not table.diags


def __test_the_payload_round_trips_an_enum__(tmp_path):
    """What the .pyc carries is what a later process reads back"""
    path = _write(tmp_path, 'en_json_lib', ENUM_LIB)
    tree, table = _analysed(path)
    interface = build_interface(tree, table, str(path.resolve()))
    zone = object_ty(class_id(str(path.resolve()), 'Zone'))

    assert interface.classes['Zone'].enum
    payload = interface_payload(interface)
    data = json.loads(zlib.decompress(payload))
    assert data['classes']['Zone']['enum'] is True
    assert data['classes']['Zone']['fields'] == {'exchange': zone, 'utc': zone}
    stat = os.stat(path)
    restored = interface_from_payload(interface.path, payload, interface.deps,
                                      (stat.st_mtime_ns, stat.st_size))
    assert restored is not None
    assert restored.classes['Zone'] == interface.classes['Zone']


def __test_turning_a_class_into_an_enum_moves_the_digest__(tmp_path):
    """A dependent reads a member through the class only when it is an enum"""
    enum = _write(tmp_path, 'en_digest_enum', ENUM_LIB)
    plain = _write(tmp_path, 'en_digest_plain', ENUM_LIB.replace('class Zone(Enum):', 'class Zone:'))

    digests = []
    for path in (enum, plain):
        tree, table = _analysed(path)
        digests.append(build_interface(tree, table, str(path.resolve())).digest)
    assert digests[0] != digests[1]


ENUM_MAP_LIB = '''"""
@pyne
"""
from enum import StrEnum as Enum

from pynecore import lib
from pynecore.core.pine_export import Exported, export

__all__ = ['Zone', 'total']
total = Exported()


class Zone(Enum):
    exchange = 'Exchange'
    utc = 'UTC'


@export
def total(levels: dict[Zone, float]) -> float:
    return lib.map.get(levels, Zone.utc)
'''


@pytest.mark.parametrize('spelling, imported', [
    ('chrono.Zone', 'import en_mapkey_lib as chrono'),
    ('Zone', 'from en_mapkey_lib import Zone'),
])
def __test_a_library_enum_keys_a_map__(tmp_path, monkeypatch, spelling: str, imported: str):
    """An imported enum keys a map like an own one, and an export publishes such a map"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'en_mapkey_lib', ENUM_MAP_LIB)
    app = _write(tmp_path, 'en_mapkey_app', f'''"""
@pyne
"""
{imported}
from pynecore import lib


def main():
    levels: dict[{spelling}, float] = lib.map.new()
    lib.map.put(levels, {spelling}.utc, 1.5)
    got = lib.map.get(levels, {spelling}.exchange)
    keys = lib.map.keys(levels)
    return got
''')

    _, table = _analysed(app)

    cid = class_id(str(lib.resolve()), 'Zone')
    levels = map_of(object_ty(cid), FLOAT, frozenset({cid}))
    assert _binding(table, 'main', 'levels') == levels
    assert _binding(table, 'main', 'got') == FLOAT
    assert _binding(table, 'main', 'keys') == array_of(object_ty(cid))
    assert not table.diags, [diag.message for diag in table.diags]

    tree, lib_table = _analysed(lib)
    interface = build_interface(tree, lib_table, str(lib.resolve()))
    assert interface.exports['total'].params == (levels,)
    assert interface.exports['total'].ret == FLOAT
