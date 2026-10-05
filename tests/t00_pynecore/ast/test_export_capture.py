"""Export capture validation applies to handwritten Pyne as well as compiled Pine."""
import ast

import pytest

from pynecore.core.import_hook import PyneLoader
from pynecore.transformers.export_capture import ExportCaptureTransformer
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer


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
