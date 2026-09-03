"""
The ``@pyne edge`` gate: a module that promises to be Pine is held to it.

The promise has two halves. The STRUCTURAL half is the Pyne Edge profile --
the syntax a Pine program can be written with -- defined once in the
PyneIDE's spec, extracted into ``edge_rules.json`` and applied by
``pine_edge_gate`` on the tree as written. The TYPE half is that every value
has a known Pine type, which the inference reports once per cause through
``pine_type_report``. Both halves are diagnostics; in hand-written code they
are a coverage meter and stop nothing, in an edge module under
``PYNE_EDGE_STRICT=1`` the first one is a ``PineTypeError`` with a real caret.
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from pynecore.core.import_hook import PyneLoader, analyse_source
from pynecore.transformers import pine_edge_gate, pine_type_artifact
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
from pynecore.transformers.pine_edge_gate import (
    EDGE_RULES_VERSION, DIAG_ENV, STRICT_ENV, edge_rules, gate_module, gated, render_diags,
)
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_table import Diag, PineTypeError, PineTypeTable
from pynecore.transformers.pine_type_report import unknown_diags

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JSON_PATH = _REPO_ROOT / 'src' / 'pynecore' / 'transformers' / 'edge_rules.json'


@pytest.fixture(autouse=True)
def _clean_registry():
    pine_type_artifact._registry.clear()
    pine_type_artifact._analysing.clear()
    yield
    pine_type_artifact._registry.clear()
    pine_type_artifact._analysing.clear()


def _gate(source: str) -> list[Diag]:
    return gate_module(ast.parse(source))


def _reasons(diags: list[Diag]) -> list[str]:
    return [diag.origin.reason for diag in diags if diag.origin is not None]


def _infer(source: str) -> tuple[ast.Module, PineTypeTable]:
    tree = ImportNormalizerTransformer().visit(ast.parse(source))
    return tree, infer_module(tree, 'probe.py')


def _type_diags(source: str) -> list[Diag]:
    _, table = _infer(source)
    return table.diags


HEAD = '''
from pynecore import lib
from pynecore.lib import script, close, math, ta, array
from pynecore.core.pine_udt import udt
from pynecore.core.pine_method import method
from pynecore.core.overload import overload
from dataclasses import field
'''


# --- 1. the rules are extracted, not written ------------------------------


def _load_collector():
    path = _REPO_ROOT / 'scripts' / 'edge_rules_collector.py'
    spec = importlib.util.spec_from_file_location('edge_rules_collector', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def __test_edge_rules_json_is_current__():
    """The committed rules are what the collector extracts from the spec today"""
    collector = _load_collector()
    spec_path = collector.default_spec_path()
    if not spec_path.exists():
        pytest.skip(f'the Pyne Edge spec is not checked out at {spec_path}')
    generated = collector.collect(collector.load_spec(spec_path))
    assert generated == json.loads(_JSON_PATH.read_text()), \
        'edge_rules.json is stale -- rerun scripts/edge_rules_collector.py'


def __test_the_gate_pins_the_profile_revision__():
    """A spec revision that was not re-extracted is an error, not a stale gate"""
    rules = edge_rules()
    assert rules['rules_version'] == EDGE_RULES_VERSION
    assert rules['v'] == _load_collector().SCHEMA_VERSION
    pine_edge_gate._rules = None
    saved = pine_edge_gate.EDGE_RULES_VERSION
    try:
        pine_edge_gate.EDGE_RULES_VERSION = '1999.01.1'
        with pytest.raises(RuntimeError, match='edge_rules_collector'):
            edge_rules()
    finally:
        pine_edge_gate.EDGE_RULES_VERSION = saved
        pine_edge_gate._rules = None


def __test_only_an_edge_module_is_gated__():
    assert gated('edge') and not gated('lib') and not gated(None)


# --- 2. the structural half --------------------------------------------------


@pytest.mark.parametrize("snippet,reason,words", [
    ('try:\n    x = 1\nexcept TypeError:\n    x = 2\n', 'edge-syntax', "'try'"),
    ('with open("f") as h:\n    pass\n', 'edge-syntax', "'with'"),
    ('match close:\n    case 1:\n        pass\n', 'edge-syntax', "'match'"),
    ('def f():\n    global g\n', 'edge-syntax', "'global'"),
    ('del close\n', 'edge-syntax', "'del'"),
    ('assert close > 0\n', 'edge-syntax', "'assert'"),
    ('x = {1: 2}\n', 'edge-syntax', 'dict literal'),
    ('x = {1, 2}\n', 'edge-syntax', 'set literal'),
    ('x = [i for i in (1, 2)]\n', 'edge-syntax', 'comprehension'),
    ('x = f"{close}"\n', 'edge-syntax', 'f-string'),
    ('x = (y := 1)\n', 'edge-syntax', "':='"),
    ('x = close[1:2]\n', 'edge-syntax', 'slice'),
    ('x = 1 | 2\n', 'edge-syntax', "'|'"),
    ('x = 1 << 2\n', 'edge-syntax', "'<<'"),
    ('x = close is None\n', 'edge-syntax', "'is'"),
    ('x = 1 in (1, 2)\n', 'edge-syntax', "'in'"),
    ('def f(*a):\n    pass\n', 'edge-syntax', '*args'),
    ('def f(a, *, b):\n    pass\n', 'edge-syntax', 'keyword-only'),
    ('x = lambda: 1\n', 'edge-lambda', 'lambda'),
    ('x = [1, 2]\n', 'edge-syntax', 'list literal'),
    ('import os\n', 'edge-import', "'os'"),
    ('from typing import List\n', 'edge-import', 'typing.List'),
    ('from . import sibling\n', 'edge-import', 'relative'),
    ('def f():\n    pass\nf.attr = 1\n', 'edge-attr-store', 'attribute'),
    ('x = print\nx(1)\n', 'edge-call', "'x'"),
    ('exec("1")\n', 'edge-call', "'exec'"),
])
def __test_a_construct_outside_the_profile_is_reported__(snippet: str, reason: str, words: str):
    diags = _gate(HEAD + snippet)
    assert diags, snippet
    assert _reasons(diags)[0] == reason, [diag.message for diag in diags]
    assert words in diags[0].message, diags[0].message
    assert 'Pine' in diags[0].message and diags[0].fix


def __test_one_construct_is_one_finding__():
    """The walk does not descend into a rejected construct"""
    diags = _gate(HEAD + 'x = [i | 1 for i in {1, 2}]\n')
    assert len(diags) == 1 and 'comprehension' in diags[0].message


@pytest.mark.parametrize("snippet", [
    'x = close + 1\ny = x // 2\nz = -x ** 2\nw = not (x > 1 and y < 2 or z == 3)\n',
    'x = (1, 2)\na, b = x\n',
    'import lib.TradingView.ta.v8\nfrom lib.mine import helper\n',
    '@script.indicator("x")\ndef main():\n    pass\n',
    '@lib.script.strategy("x")\ndef main():\n    pass\n',
    '@udt\nclass Point:\n    """Two floats"""\n    x: float\n    y: float = 0.0\n',
    '@udt\nclass Bag:\n    items: list = field(default_factory=lambda: array.new_float(0))\n',
    '@method\ndef area(self: Bag) -> float:\n    return 1.0\n',
    '@overload\ndef f(x: int) -> int:\n    return x\n@overload\ndef f(x: float) -> float:\n    return x\n',
    'def f(a, b=1):\n    return a + b\nx = f(1)\ny = len("ab") + abs(-1) + max(1, 2) + int(1.5)\n',
    '__all__ = ["f"]\nfrom typing import Protocol, Any\nclass _ProtocolF(Protocol):\n    def __call__(self, x: int) -> Any: ...\n',
    'for i in range(3):\n    pass\nwhile close > 1:\n    break\n',
    'x = 1 if close > 1 else 2\nx += 1\n',
])
def __test_pine_is_let_through__(snippet: str):
    diags = _gate(HEAD + snippet)
    assert not diags, [diag.message for diag in diags]


def __test_a_class_is_a_field_list__():
    diags = _gate(HEAD + 'class A(object):\n    x: int\n')
    assert 'inheritance' in diags[0].message
    diags = _gate(HEAD + 'class B:\n    x: int\n')
    assert 'decorator' in diags[0].message
    diags = _gate(HEAD + '@udt\nclass C:\n    x: int\n    def m(self):\n        pass\n')
    assert 'annotated fields' in diags[0].message


def __test_a_rejected_import_does_not_cascade_into_its_uses__():
    diags = _gate(HEAD + 'import os\nx = os.getcwd()\n')
    assert len(diags) == 1 and _reasons(diags) == ['edge-import']


# --- 3. the type half: one cause, one report -------------------------------


def __test_a_cascade_is_reported_once_at_its_root__():
    diags = _type_diags(HEAD + '''
def helper(x):
    return x.foo


def main(length: int):
    a = helper(3)
    b = a + 1
    c = math.max(b, length)
    d = c * 2
    return d
''')
    assert [diag.line for diag in diags] == [10, 14], render_diags(diags, 'probe.py')
    assert _reasons(diags) == ['unknown-field', 'unknown-return']
    assert diags[1].fix == "annotate the return of 'helper'"
    assert "'.foo'" in diags[0].message


def __test_each_root_kind_names_its_remedy__():
    diags = _type_diags(HEAD + '''
def main(length: int):
    a = getattr(close, "x")
    b = lib.nosuch
    c = close + 1
    return a + b + c
''')
    assert _reasons(diags) == ['unknown-call', 'unknown-lib-name']
    assert "'getattr'" in diags[0].message and 'import it' in diags[0].fix
    assert "'nosuch'" in diags[1].message


def __test_an_unannotated_parameter_is_the_cause__():
    diags = _type_diags(HEAD + '''
def helper(x):
    return x + 1


def main(length: int):
    return helper(length) + helper(1.5)
''')
    assert not diags, render_diags(diags, 'probe.py')


def __test_a_typed_module_has_nothing_to_report__():
    diags = _type_diags(HEAD + '''
def main(length: int):
    s = ta.sma(close, length)
    a = array.new_float(2, s)
    first = array.get(a, 0)
    return first + s
''')
    assert not diags, render_diags(diags, 'probe.py')


def __test_plumbing_and_statement_values_are_not_reported__():
    tree, table = _infer(HEAD + '''
def main(length: int):
    getattr(close, "x")
    a = __pyne_bind__(close)
    b = a + 1
    return length
''')
    assert not unknown_diags(tree, table)


# --- 4. the error ------------------------------------------------------------


def __test_the_error_carries_the_caret_the_origin_and_the_fix__():
    diag = Diag(message="'a' has no known type", line=3, col=4,
                origin=__import__('pynecore.transformers.pine_type_table',
                                  fromlist=['Unknown']).Unknown('unannotated-param', 1, 8, 'x'),
                fix="annotate 'x'")
    error = PineTypeError.from_diag(diag, 'probe.py', '    a = x')
    assert error.filename == 'probe.py' and error.lineno == 3 and error.offset == 5
    assert error.text == '    a = x'
    assert 'type lost at line 1' in str(error) and "annotate 'x'" in str(error)


def __test_the_dump_prints_one_line_per_diagnostic__():
    diags = _type_diags(HEAD + '''
def main(length: int):
    a = getattr(close, "x")
    return a
''')
    dumped = render_diags(diags, '/p/probe.py')
    assert dumped.startswith('/p/probe.py:10:8: ')
    assert '[unknown-call@10:8 getattr]' in dumped and 'fix: ' in dumped


# --- 5. through the import hook ---------------------------------------------


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f'{name}.py'
    path.write_text(source, encoding='utf-8')
    return path


EDGE_MODULE = '''"""
@pyne edge
"""
from pynecore.lib import script, close


@script.indicator("x")
def main():
    a = getattr(close, "x")
    return a
'''


def _compile(path: Path) -> None:
    loader = PyneLoader(path.stem, str(path))
    loader.source_to_code(path.read_bytes(), str(path))


def __test_strict_makes_the_first_diagnostic_an_error__(tmp_path, monkeypatch):
    monkeypatch.setenv(STRICT_ENV, '1')
    path = _write(tmp_path, 'edge_strict', EDGE_MODULE)
    with pytest.raises(PineTypeError) as caught:
        _compile(path)
    assert caught.value.lineno == 9 and caught.value.filename == str(path)
    assert "'getattr'" in str(caught.value)
    assert caught.value.text is not None and 'getattr' in caught.value.text


def __test_without_strict_the_module_compiles_and_keeps_the_list__(tmp_path, monkeypatch):
    monkeypatch.setenv(STRICT_ENV, '0')
    path = _write(tmp_path, 'edge_lenient', EDGE_MODULE)
    _compile(path)
    analysed = analyse_source(str(path))
    assert analysed is not None
    # The structural half names the construct; the typed half of the same
    # node is not repeated behind it
    assert _reasons(analysed[1].diags) == ['edge-call']


def __test_strict_is_off_when_unset__(tmp_path, monkeypatch):
    monkeypatch.delenv(STRICT_ENV, raising=False)
    _compile(_write(tmp_path, 'edge_default', EDGE_MODULE))


def __test_a_hand_written_script_is_never_gated__(tmp_path, monkeypatch):
    monkeypatch.setenv(STRICT_ENV, '1')
    path = _write(tmp_path, 'hand_written', EDGE_MODULE.replace('@pyne edge', '@pyne')
                  .replace('    a = getattr', '    try:\n        pass\n    except TypeError:\n'
                                              '        pass\n    a = getattr'))
    _compile(path)
    analysed = analyse_source(str(path))
    assert analysed is not None
    # The type half is still measured -- the structural half is not applied
    assert _reasons(analysed[1].diags) == ['unknown-call']


def __test_the_structural_half_comes_first__(tmp_path, monkeypatch):
    monkeypatch.setenv(STRICT_ENV, '1')
    path = _write(tmp_path, 'edge_syntax', EDGE_MODULE.replace(
        '    a = getattr(close, "x")', '    b = [1, 2]\n    a = getattr(close, "x")'))
    with pytest.raises(PineTypeError) as caught:
        _compile(path)
    assert 'list literal' in str(caught.value) and caught.value.lineno == 9


def __test_the_dump_goes_to_stderr_in_any_mode__(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(DIAG_ENV, '1')
    monkeypatch.delenv(STRICT_ENV, raising=False)
    path = _write(tmp_path, 'dumped', EDGE_MODULE.replace('@pyne edge', '@pyne'))
    _compile(path)
    err = capsys.readouterr().err
    assert f'{path}:9:8: ' in err and '[unknown-call@9:8 getattr]' in err


def __test_a_computed_callee_and_a_reserved_name_are_rejected__():
    reasons = _reasons(_gate(HEAD + 'x = __builtins__["eval"]("40+2")\n'))
    assert 'edge-call' in reasons and 'edge-name' in reasons
    assert _reasons(_gate(HEAD + 'y = __builtins__.eval("40+2")\n')) == ['edge-name']
    assert _reasons(_gate(HEAD + 'z = close.__class__\n')) == ['edge-name']
    # What the compiler spells is let through
    assert not _gate(HEAD + 'if __name__ == "__main__":\n    pass\n__block_result_3__ = 1\n'
                            '__all__ = ["f"]\ndef f(__input_0__):\n    return __input_0__\n')


def __test_a_shim_signature_still_runs__():
    shim = ('from typing import Protocol, Any\nclass _ProtocolF(Protocol):\n'
            '    @print("executed")\n    def __call__(self, x: int) -> Any: ...\n')
    assert _gate(HEAD + shim)
    shim = ('from typing import Protocol, Any\nclass _ProtocolF(Protocol):\n'
            '    def __call__(self, x: int = __import__("os")) -> Any: ...\n')
    assert 'edge-call' in _reasons(_gate(HEAD + shim))


def __test_an_unresolved_callee_is_reported_where_its_value_is_discarded__():
    diags = _type_diags(HEAD + 'def main():\n    lib.nosuch()\n    nothing()\n')
    assert _reasons(diags) == ['unknown-lib', 'unknown-call']


def __test_a_namespace_is_not_callable__():
    diags = _type_diags(HEAD + 'def main():\n    math()\n    lib.math()\n')
    assert _reasons(diags) == ['unknown-lib', 'unknown-lib']
    assert all('namespace' in diag.message for diag in diags)


def __test_a_rebound_protocol_is_no_shim__():
    shim = ('from typing import Protocol, Any\nProtocol = 1\n'
            'class _ProtocolF(Protocol):\n    def __call__(self, x: int) -> Any: ...\n')
    assert _gate(HEAD + shim)
    shim = ('from typing import Protocol as P, Any\nP = 1\n'
            'class _ProtocolF(P):\n    def __call__(self, x: int) -> Any: ...\n')
    assert _gate(HEAD + shim)


def __test_a_script_decorator_is_always_called__():
    assert _reasons(_gate(HEAD + '@script.indicator\ndef main():\n    pass\n')) == ['edge-decorator']
    assert _reasons(_gate(HEAD + '@lib.script.strategy\ndef main():\n    pass\n')) \
        == ['edge-decorator']


def __test_a_store_on_a_shadowing_value_is_a_field_write__():
    box = '@udt\nclass Box:\n    x: int\n'
    assert not _gate(HEAD + box + 'def f(math: Box):\n    math.x = 1\n')
    assert not _gate(HEAD + box + 'def g():\n    math = Box(1)\n    math.x = 1\n')
    assert not _gate(HEAD + box + 'math = Box(1)\nmath.x = 1\n')
    assert _reasons(_gate(HEAD + box + 'math.x = 1\nmath = Box(1)\n')) == ['edge-attr-store']
    assert _reasons(_gate(HEAD + 'math.x = 1\n')) == ['edge-attr-store']


def __test_a_field_factory_belongs_to_a_class_body__():
    stray = 'def main():\n    x = field(default_factory=lambda: 1)\n    return x\n'
    assert _reasons(_gate(HEAD + stray)) == ['edge-lambda']
    assert 'unknown-call' in _reasons(_type_diags(HEAD + stray))
    housed = '@udt\nclass Bag:\n    items: list = field(default_factory=lambda: array.new_float(0))\n'
    assert not _gate(HEAD + housed)
    assert not _type_diags(HEAD + housed)


def __test_the_call_shape_is_pine_too__():
    """Keyword unpacking, a loop's else and a chained comparison are not Pine"""
    assert _reasons(_gate(HEAD + 'def f(x):\n    return x\ny = f(**close)\n')) == ['edge-syntax']
    assert _reasons(_gate(HEAD + 'for i in range(3):\n    pass\nelse:\n    pass\n')) \
        == ['edge-syntax']
    assert _reasons(_gate(HEAD + 'while close > 1:\n    break\nelse:\n    pass\n')) \
        == ['edge-syntax']
    assert _reasons(_gate(HEAD + 'x = 1 < 2 < 3\n')) == ['edge-syntax']
    assert not _gate(HEAD + 'x = 1 < 2 and 2 < 3\n')


def __test_one_node_is_one_finding__():
    """A reserved-name callee is reported as the call, not twice"""
    assert _reasons(_gate(HEAD + 'x = __import__("os")\n')) == ['edge-call']


def __test_every_identifier_position_is_checked__():
    """Keywords, class names and import aliases are identifiers too"""
    assert _reasons(_gate(HEAD + 'def f(x):\n    return x\ny = f(__x__=1)\n')) == ['edge-name']
    assert _reasons(_gate(HEAD + '@udt\nclass __A__:\n    x: int\n')) == ['edge-name']
    assert _reasons(_gate(HEAD + 'from pynecore.lib import high as __h__\n')) == ['edge-name']
    assert _reasons(_gate(HEAD + 'import pynecore.lib as __l__\n')) == ['edge-name']


def __test_a_star_import_is_rejected__():
    assert _reasons(_gate(HEAD + 'from pynecore.lib import *\n')) == ['edge-import']


def __test_a_function_scope_store_is_order_aware__():
    """A name the function binds LATER is not yet a value at the store"""
    assert _reasons(_gate(HEAD + 'def f():\n    math.x = 1\n    math = 1\n')) \
        == ['edge-attr-store']
    assert not _gate(HEAD + 'def g():\n    math = 1\n    math.x = 1\n')


def __test_the_merge_keeps_an_unrelated_typed_finding__(tmp_path, monkeypatch):
    """A structural finding hides only the typed repeat of ITS construct"""
    monkeypatch.setenv(STRICT_ENV, '0')
    path = _write(tmp_path, 'edge_merge', EDGE_MODULE.replace(
        '    a = getattr(close, "x")', '    a = nosuchname | 1'))
    _compile(path)
    analysed = analyse_source(str(path))
    assert analysed is not None
    assert _reasons(analysed[1].diags) == ['edge-syntax', 'unknown-name']


def __test_the_merged_list_is_in_source_order__(tmp_path, monkeypatch):
    monkeypatch.setenv(STRICT_ENV, '0')
    path = _write(tmp_path, 'edge_order', EDGE_MODULE.replace(
        '    a = getattr(close, "x")', '    a = nosuch\n    b = [1]\n    c = getattr(close, "x")'))
    _compile(path)
    analysed = analyse_source(str(path))
    assert analysed is not None
    positions = [(diag.line, diag.col) for diag in analysed[1].diags]
    assert positions == sorted(positions) and len(positions) == 3


def __test_every_unknown_operand_is_a_root__():
    """``foo + bar`` has two causes and reports both"""
    diags = _type_diags(HEAD + 'def main():\n    return foo + bar\n')
    assert _reasons(diags) == ['unknown-name', 'unknown-name']
    assert [diag.col for diag in diags] == [11, 17]


def __test_every_compiler_dunder_is_let_through__():
    """The names the compiler emits for blocks, switches and loops are Pine's"""
    assert not _gate(HEAD + '__switch_1__ = 1\n__loop_2__ = 1\n__block_keep_3__ = 1\n'
                            '__switch__ = 1\n__block_result__ = 1\n')
    assert _reasons(_gate(HEAD + '__switch_x__ = 1\n')) == ['edge-name']


def __test_a_rejected_expression_covers_its_parts__(tmp_path, monkeypatch):
    """The typed half says nothing about what is written inside a rejected construct"""
    monkeypatch.setenv(STRICT_ENV, '0')
    path = _write(tmp_path, 'edge_span', EDGE_MODULE.replace(
        '    a = getattr(close, "x")', '    a = f"bar {close} and {close}"\n    b = nosuch | 1'))
    _compile(path)
    analysed = analyse_source(str(path))
    assert analysed is not None
    assert _reasons(analysed[1].diags) == ['edge-syntax', 'edge-syntax', 'unknown-name']
