"""
Freshness guard for ``python_names.json``.

The call-site isolation routes a call by what its callee name is -- a builtin,
a keyword it cannot rebuild from its spelling, a standard library import --
and reads those name sets from the generated file, not from the interpreter it
runs under (the native transform has none). The file holds the union over every
supported interpreter, so the one running the tests must find all of its names
there: ``scripts/python_names_collector.py`` adds them.
"""
import builtins
import importlib.util
import json
import keyword
import sys
from pathlib import Path

from pynecore.transformers.python_names import is_builtin, is_keyword, is_stdlib

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JSON_PATH = _REPO_ROOT / 'src' / 'pynecore' / 'transformers' / 'python_names.json'

_REGENERATE = ("python_names.json lacks this interpreter's names -- run "
               "`python scripts/python_names_collector.py` with it from the checkout root, "
               "and commit the result")


def __test_helper_collector():
    path = _REPO_ROOT / 'scripts' / 'python_names_collector.py'
    spec = importlib.util.spec_from_file_location('python_names_collector', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def __test_the_running_interpreter_is_collected__():
    """Every builtin, keyword and stdlib module of this interpreter is in the file"""
    committed = json.loads(_JSON_PATH.read_text())
    collector = __test_helper_collector()
    assert collector.merged(collector.collect(), committed) == committed, _REGENERATE


def __test_the_file_is_sorted_union_data__():
    """The sections are sorted and free of duplicates, as the collector writes them"""
    committed = json.loads(_JSON_PATH.read_text())
    assert sorted(committed) == ['builtins', 'keywords', 'stdlib']
    for names in committed.values():
        assert names == sorted(set(names))


def __test_the_lookups_answer_like_the_interpreter__():
    """The registry answers what the interpreter answers for its own names"""
    assert all(is_builtin(name) for name in vars(builtins))
    assert all(is_keyword(name) for name in keyword.kwlist)
    assert not is_keyword('match') and not is_keyword('print')
    assert all(is_stdlib(module) for module in sys.stdlib_module_names)
    assert is_stdlib('os.path') and is_stdlib('collections.abc')
    assert not is_stdlib('pynecore') and not is_stdlib('numpy') and not is_builtin('ta')
