"""Export capture validation applies to handwritten Pyne as well as compiled Pine."""
import ast
import os
import sys

import pytest

from pynecore.core.import_hook import PyneLoader
from pynecore.core import script as script_mod
from pynecore.core.script_runner import ScriptRunner
from pynecore.transformers.export_capture import ExportCaptureTransformer
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
from pynecore.types.ohlcv import OHLCV


HEAD = '''
from pynecore.lib import script, label, close, color
from pynecore.core.pine_export import export, Exported
from pynecore.types import Persistent
getObject = Exported()
'''


def check(source):
    tree = ImportNormalizerTransformer().visit(ast.parse(HEAD + source))
    return ExportCaptureTransformer().visit(tree)


@pytest.mark.parametrize('value', ['label.new(0, 1)', 'label.all', 'close', '[1, 2]', 'color.new(color.red, 20)'])
@pytest.mark.parametrize('indirect', [False, True])
def __test_export_capture__(value, indirect):
    helper = '    def capture():\n        return obj\n' if indirect else ''
    result = 'capture()' if indirect else 'obj'
    source = f'''
@script.library("Probe")
def main():
    obj = {value}
{helper}    @export
    def getObject():
        return {result}
'''
    if value.startswith('color.'):
        check(source)
    else:
        with pytest.raises(SyntaxError, match='cannot capture non-constant library global "obj"'):
            check(source)


@pytest.mark.parametrize('source', [
    '''@script.indicator("Probe")
def main():
    obj = label.new(0, 1)
    def getObject():
        return obj
''',
    '''@script.library("Probe")
def main():
    @export
    def getObject():
        obj: Persistent = label.new(0, 1)
        return obj
''',
    '''@script.library("Probe")
def main():
    n = 2
    factor = n * 3
    def helper():
        return factor
    @export
    def getObject():
        return helper()
''',
    '''@script.library("Probe")
def main():
    obj = label.new(0, 1)
    @export
    def getObject(obj):
        return obj
''',
])
def __test_valid_closures__(source):
    check(source)


def __test_module_capture__():
    with pytest.raises(SyntaxError, match='global "obj"'):
        check('''obj = label.new(0, 1)
@export
def getObject():
    return obj
@script.library("Probe")
def main():
    pass
''')


@pytest.mark.parametrize('mode', ['', ' edge'])
def __test_loader_rejects_before_execution__(tmp_path, mode):
    source = f'"""\n@pyne{mode}\n"""\n' + HEAD + '''
@script.library("Probe")
def main():
    obj = label.new(0, 1)
    @export
    def getObject():
        return obj
'''
    path = tmp_path / 'capture.py'
    path.write_text(source)
    with pytest.raises(SyntaxError, match='global "obj"'):
        PyneLoader('capture', str(path)).get_code('capture')


@pytest.mark.parametrize('body', [
    '    @export\n    def getObject(obj=obj):\n        return obj\n',
    '    def helper():\n        return obj\n    @export\n    def getObject():\n        return helper()\n',
    '    capture = lambda: obj\n    @export\n    def getObject():\n        return capture()\n',
])
def __test_capture_paths__(body):
    with pytest.raises(SyntaxError, match='cannot capture non-constant library global'):
        check('@script.library("Probe")\ndef main():\n    obj = label.new(0, 1)\n' + body)


def __test_nonlocal_write_invalidates_literal__():
    with pytest.raises(SyntaxError, match='global "obj"'):
        check('''@script.library("Probe")
def main():
    obj = 1
    @export
    def getObject():
        nonlocal obj
        obj += 1
        return obj
''')


def __test_compiler_constant_storage__():
    check('''@script.library("Probe")
def main():
    factor: Persistent[float] = 3.0
    @export
    def getObject():
        return factor
''')


@pytest.mark.parametrize('initializer', ['auto()', '-1', '1 << 2'])
def __test_local_enum_capture_compatibility__(initializer):
    check(f'''from enum import Enum, auto
class Mode(Enum):
    fixed = {initializer}
obj = Mode.fixed
@script.library("Probe")
def main():
    @export
    def getObject():
        return obj
''')


@pytest.fixture
def __test_helper_enum_package(tmp_path, monkeypatch):
    package = tmp_path / 'capture_enums'
    package.mkdir()
    (package / '__init__.py').write_text('raise RuntimeError("Package must not execute")\n')
    (package / 'definitions.py').write_text('''from enum import StrEnum as Enum
class NumericSystem(Enum):
    decimal = 'DEC'
    hexadecimal = 'HEX'
''')
    (package / 'bridge.py').write_text('from .definitions import NumericSystem as NumberBase\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    return package


@pytest.mark.parametrize('import_line, value', [
    ('import capture_enums.definitions as sd', 'sd.NumericSystem.decimal'),
    ('import capture_enums.definitions', 'capture_enums.definitions.NumericSystem.decimal'),
    ('from capture_enums.definitions import NumericSystem as NS', 'NS.decimal'),
    ('from capture_enums.bridge import NumberBase', 'NumberBase.hexadecimal'),
])
@pytest.mark.parametrize('module_constant', [False, True])
def __test_imported_enum_capture__(__test_helper_enum_package, import_line, value, module_constant):
    declaration = f'obj = {value}\n'
    source = import_line + '\n'
    if module_constant:
        source += declaration
    source += '@script.library("Probe")\ndef main():\n'
    if not module_constant:
        source += '    ' + declaration
    source += '''    def helper():
        return obj
    @export
    def getObject():
        return helper()
'''
    check(source)
    assert 'capture_enums' not in sys.modules
    assert 'capture_enums.definitions' not in sys.modules


@pytest.mark.parametrize('value', [
    'sd.NumericSystem.missing',
    'sd.NumericSystem.decimal.value',
    'sd.NumericSystem.__members__',
    'sd.mutable',
    'sd.close',
])
def __test_imported_non_enum_capture_rejected__(__test_helper_enum_package, value):
    with pytest.raises(SyntaxError, match='global "obj"'):
        check(f'''import capture_enums.definitions as sd
obj = {value}
@script.library("Probe")
def main():
    @export
    def getObject():
        return obj
''')


@pytest.mark.parametrize('extra', [
    'sd = object()\n',
    'sd = 1\n',
])
def __test_import_alias_rebinding_rejected__(__test_helper_enum_package, extra):
    with pytest.raises(SyntaxError, match='global "obj"'):
        check('import capture_enums.definitions as sd\n' + extra + '''
obj = sd.NumericSystem.decimal
@script.library("Probe")
def main():
    @export
    def getObject():
        return obj
''')


@pytest.mark.parametrize('definition', [
    "class Enum:\n    pass\nclass NumericSystem(Enum):\n    decimal = 'DEC'\n",
    "from enum import Enum\nclass NumericSystem(Enum):\n    decimal = []\n",
    "from enum import Enum\nclass NumericSystem(Enum):\n    decimal = 'DEC'\nNumericSystem = object()\n",
])
def __test_unproven_imported_enum_rejected__(__test_helper_enum_package, definition):
    (__test_helper_enum_package / 'definitions.py').write_text(definition)
    with pytest.raises(SyntaxError, match='global "obj"'):
        check('''import capture_enums.definitions as sd
obj = sd.NumericSystem.decimal
@script.library("Probe")
def main():
    @export
    def getObject():
        return obj
''')


def __test_import_cycle_cannot_prove_enum__(__test_helper_enum_package):
    (__test_helper_enum_package / 'definitions.py').write_text('from .bridge import NumberBase as NumericSystem\n')
    with pytest.raises(SyntaxError, match='global "obj"'):
        check('''from capture_enums.bridge import NumberBase
obj = NumberBase.decimal
@script.library("Probe")
def main():
    @export
    def getObject():
        return obj
''')


def __test_unloaded_nested_namespace_enum__(tmp_path, monkeypatch):
    package = tmp_path / 'capture_namespace' / 'author' / 'numeric'
    package.mkdir(parents=True)
    (package / 'v1.py').write_text("from enum import StrEnum\nclass Mode(StrEnum):\n    fixed = 'FIXED'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    check('''import capture_namespace.author.numeric.v1 as ns
obj = ns.Mode.fixed
@script.library("Probe")
def main():
    @export
    def getObject():
        return obj
''')
    assert 'capture_namespace' not in sys.modules


@pytest.mark.parametrize('mode', ['', ' edge'])
def __test_enum_source_change_invalidates_capture_cache__(tmp_path, monkeypatch, mode):
    monkeypatch.syspath_prepend(str(tmp_path))
    enum_path = tmp_path / 'capture_enum.py'
    enum_path.write_text("from enum import StrEnum\nclass Mode(StrEnum):\n    fixed = 'FIXED'\n")
    path = tmp_path / 'capture.py'
    path.write_text(f'"""@pyne{mode}"""\n' + HEAD + '''import capture_enum as modes
obj = modes.Mode.fixed
@script.library("Probe")
def main():
    @export
    def getObject():
        return obj
''')
    loader = PyneLoader('capture', str(path))
    first = loader.get_code('capture')
    assert any(isinstance(item, tuple) and item and item[0] == '__pyne_capture_deps__'
               for item in first.co_consts)
    assert loader.get_code('capture') is not None
    before = enum_path.stat()
    enum_path.write_text("class Mode:\n    fixed = 'FIXED'\n")
    os.utime(enum_path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    with pytest.raises(SyntaxError, match='global "obj"'):
        loader.get_code('capture')


@pytest.mark.parametrize('mode', ['', ' edge'])
def __test_imported_enum_export_executes_across_bars__(tmp_path, monkeypatch, syminfo, mode):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(script_mod, '_registered_libraries', [])
    (tmp_path / 'capture_runtime_enum.py').write_text(
        "from enum import StrEnum\nclass Mode(StrEnum):\n    fixed = 'FIXED'\n")
    (tmp_path / 'capture_runtime_library.py').write_text(f'"""@pyne{mode}"""\n' + HEAD + '''
from capture_runtime_enum import Mode
obj = Mode.fixed
@script.library("Probe")
def main():
    def helper():
        return obj
    @export
    def getObject():
        return helper()
''')
    path = tmp_path / 'capture_runtime_script.py'
    path.write_text(f'"""@pyne{mode}"""\n' + '''
import capture_runtime_library as captured
from pynecore.lib import script, plot
@script.indicator("Probe")
def main():
    result = 0
    if captured.getObject() == captured.Mode.fixed:
        result = 1
    plot(result, "enum")
''')
    bars = [OHLCV(timestamp=1_735_689_600_000 + bar * 300_000,
                  open=10, high=12, low=9, close=11, volume=1) for bar in range(3)]
    try:
        rows = list(ScriptRunner(path, bars, syminfo).run_iter())
        assert [row[1]['enum'] for row in rows] == [1, 1, 1]
    finally:
        sys.modules.pop('capture_runtime_enum', None)
        sys.modules.pop('capture_runtime_library', None)
