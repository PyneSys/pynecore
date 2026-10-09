"""
The plain double-underscore names the transform emits are reserved in Pyne code.

A generated temporary that a script also binds would silently collide with it: the
truthiness walrus ``(__bool1__ := ...)`` rebinding a module variable of that name
changes what the script computes. The loader therefore rejects user code that
spells one of them, and every name the transformers emit must be in the list.
"""
import ast
import re
from pathlib import Path

import pytest

from pynecore.core.import_hook import PyneLoader
from pynecore.transformers.pipeline_names import is_pipeline_name

__test_helper_script = '''"""@pyne"""
from pynecore.lib import script, close, plot

{module}

@script.indicator("pipeline names")
def main():
{body}
    plot(close)
'''


def __test_helper_compile(tmp_path: Path, module: str = '', body: str = '    pass') -> None:
    path = tmp_path / 'pipeline_names_script.py'
    source = __test_helper_script.format(module=module, body=body)
    path.write_text(source, encoding='utf-8')
    PyneLoader(path.stem, str(path)).source_to_code(source, str(path))


def __test_a_generated_temporary_is_rejected__(tmp_path):
    """ A module variable named like a truthiness temporary is a compile error """
    with pytest.raises(SyntaxError, match="'__bool1__' is a name PyneCore's transform generates"):
        __test_helper_compile(tmp_path, module='__bool1__ = 7.0')


@pytest.mark.parametrize('name', ['__state__', '__pyne_layout__', '__sec_read__',
                                  '__dyn_default__', '__cmp12__', '__hist_3__'])
def __test_every_family_is_rejected__(tmp_path, name):
    """ Plumbing, protocol and numbered temporaries alike """
    with pytest.raises(SyntaxError, match=re.escape(f"'{name}'")):
        __test_helper_compile(tmp_path, body=f'    {name} = 1')


def __test_the_normalized_spelling_is_rejected__(tmp_path):
    """ Python binds the NFKC form, so a fullwidth spelling is the same name """
    with pytest.raises(SyntaxError, match="'__state__'"):
        __test_helper_compile(tmp_path, body='    __ｓｔａｔｅ__ = 1')


def __test_strings_and_comments_may_mention_them__(tmp_path):
    """ Only identifiers are reserved """
    __test_helper_compile(tmp_path, module='# __bool1__\nNOTE = "__state__ and __cmp1__"')


def __test_compiler_spelled_names_are_not_reserved__(tmp_path):
    """ The names PyneComp writes into the source belong to the script """
    __test_helper_compile(tmp_path, body='    __block_result__ = 1\n    __switch_2__ = 2\n'
                                         '    __loop_3__ = 3\n    __hist_close__ = 4\n'
                                         '    __input_5__ = 5')


def __test_test_functions_may_use_them__(tmp_path):
    """ ``__test_*__`` functions are removed before the check, as before any step """
    __test_helper_compile(
        tmp_path, module='def __test_reads_the_runtime__():\n    __state__ = 1\n    return __state__')


#: Double-underscore names the transformers spell that are not emitted as names of
#: their own: Python's own, PyneComp's suffix markers, and runtime protocol
#: attributes read off lib objects outside the ``__pyne_*__`` namespace
__test_helper_not_emitted = frozenset({
    '__all__', '__call__', '__class__', '__defaults__', '__file__', '__future__',
    '__init__', '__kwdefaults__', '__main__', '__module__', '__name__', '__self__',
    '__setitem__', '__ren__', '__global__', '__module_property__',
})


def __test_every_emitted_name_is_reserved__():
    """ Every plain ``__x__`` name literal in the transform is in the reserved list """
    import pynecore

    root = Path(pynecore.__file__).parent
    files = sorted((root / 'transformers').glob('*.py')) + [root / 'core' / 'import_hook.py']
    found = set()
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.JoinedStr):
                # A formatted part is the reserved separator or a counter / index
                text = ''.join(part.value if isinstance(part, ast.Constant)
                               else '·' if ast.unparse(part.value) == 'PYNE_RESERVED_NAME_CHAR'
                               else '1'
                               for part in node.values)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            else:
                continue
            if re.fullmatch(r'__\w+__', text) and '·' not in text:
                found.add(text)
    missing = sorted(name for name in found
                     if not is_pipeline_name(name) and name not in __test_helper_not_emitted
                     and not name.startswith('__test_'))
    assert not missing, f"emitted names missing from pipeline_names.PIPELINE_NAME: {missing}"
