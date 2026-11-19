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
from pynecore.errorcodes import TYPE_NO_FIELD
from pynecore.transformers import module_interface
from pynecore.transformers.module_interface import (
    build_interface, interface_from_payload, interface_payload,
)
from pynecore.transformers.pine_qualifier import CONST
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import BOOL, UNKNOWN, class_id, get_ty, object_ty
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
