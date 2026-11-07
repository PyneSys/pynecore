"""
An input call is valid only as a parameter default of module-level ``main()``.

The InputTransformer gives such a call its parameter's name as ``_id``; the runner
reads the settings file and the overrides by that id. A call anywhere else would
run with ``_id=""``: it would never reach the settings file, no override could
reach it, and it would silently run with its default. The transform rejects it.
"""
import ast
from pathlib import Path

import pytest

from pynecore.core import import_hook

#: The header every case module starts with, ``{mode}`` being the ``@pyne`` mode word
__test_helper_HEADER = ('"""\n@pyne{mode}\n"""\n'
                        'from pynecore import lib\n'
                        'from pynecore.lib import script, input, close, plot, ta, request, syminfo\n'
                        '\n')

#: Cases that must be rejected: the module body and the 1-based line of the
#: offending call within it
__test_helper_REJECTED = {
    'module level': ('x = input.int(9, "Mod")\n'
                     '@script.indicator("p")\n'
                     'def main():\n'
                     '    plot(close + x)\n', 1),
    'main body': ('@script.indicator("p")\n'
                  'def main():\n'
                  '    y = input.float(1.5, "Body")\n'
                  '    plot(close * y)\n', 3),
    'generic input in body': ('@script.indicator("p")\n'
                              'def main():\n'
                              '    y = lib.input(1.5, "Body")\n'
                              '    plot(close * y)\n', 3),
    'conditional': ('@script.indicator("p")\n'
                    'def main():\n'
                    '    if close > 0:\n'
                    '        z = input.bool(True, "Z")\n'
                    '    plot(close)\n', 4),
    'reassigned parameter': ('@script.indicator("p")\n'
                             'def main(n=input.int(3, "N")):\n'
                             '    n = input.int(20, "N")\n'
                             '    plot(close + n)\n', 3),
    'helper parameter': ('def helper(n=input.int(3, "H")):\n'
                         '    return n\n'
                         '@script.indicator("p")\n'
                         'def main():\n'
                         '    plot(close + helper())\n', 1),
    'nested in a main default': ('@script.indicator("p")\n'
                                 'def main(n=max(input.int(3, "H"), 1)):\n'
                                 '    plot(close + n)\n', 2),
    'nested main': ('@script.indicator("p")\n'
                    'def main():\n'
                    '    def main(n=input.int(3, "H")):\n'
                    '        return n\n'
                    '    plot(close + main())\n', 3),
    'lambda default in main': ('@script.indicator("p")\n'
                               'def main():\n'
                               '    f = lambda q=input.int(4, "Q"): q\n'
                               '    plot(close + f())\n', 3),
}

#: Valid declarations: plain, keyword-only and positional-only parameters, the
#: ``lib.``-qualified spellings, and inputs feeding ``request.security``, whose
#: generated per-context clones of main() copy its parameters
__test_helper_ACCEPTED = (
    '@script.indicator("p")\n'
    'def main(a=input.int(1, "A"), /, length=input.int(14, "L"), src=input.source(close, "S"),\n'
    '         tf=lib.input.timeframe("60", "TF"), *, k=lib.input.float(1.5, "K"), x=input(2.0)):\n'
    '    v = request.security(syminfo.tickerid, tf, ta.sma(src, length))\n'
    '    plot(v * k + a + x)\n'
)


def __test_helper_lower(tmp_path: Path, body: str, mode: str) -> ast.Module:
    """Transform a case module as the loader does, from a file so errors carry the line text.

    :param tmp_path: Directory for the case file
    :param body: The module body after the header
    :param mode: The ``@pyne`` mode word, '' for a hand-written script
    :return: The lowered module
    """
    source = __test_helper_HEADER.format(mode=f' {mode}' if mode else '') + body
    path = tmp_path / 'case.py'
    path.write_text(source, encoding='utf-8')
    tree = ast.parse(source)
    pyne_mode = import_hook._module_mode(tree)[1]
    analysed = import_hook._analyse_tree(tree, source, path, pyne_mode)
    return import_hook._lower_tree(analysed, path, pyne_mode)[0]


@pytest.mark.parametrize('backend', ['python', 'verify'])
@pytest.mark.parametrize('mode', ['', 'edge'])
@pytest.mark.parametrize('case', list(__test_helper_REJECTED))
def __test_input_outside_main_defaults_is_rejected__(case, mode, backend, tmp_path, monkeypatch):
    """Every input call that is not a main() parameter default fails at transform time,
    located at the call and naming the correct form."""
    monkeypatch.setenv('PYNE_AST_BACKEND', backend)
    body, line = __test_helper_REJECTED[case]
    with pytest.raises(SyntaxError) as info:
        __test_helper_lower(tmp_path, body, mode)
    error = info.value
    assert 'inputs must be declared as main() parameter defaults: ' \
           'def main(length=input.int(14, ...))' in error.msg
    assert error.lineno == __test_helper_HEADER.count('\n') + line
    assert error.text is not None and 'input' in error.text
    assert error.filename == str((tmp_path / 'case.py').resolve())


@pytest.mark.parametrize('backend', ['python', 'verify'])
@pytest.mark.parametrize('mode', ['', 'edge'])
def __test_main_parameter_inputs_get_their_ids__(mode, backend, tmp_path, monkeypatch):
    """main() parameter inputs of every parameter kind carry their parameter's name as _id."""
    monkeypatch.setenv('PYNE_AST_BACKEND', backend)
    lowered = __test_helper_lower(tmp_path, __test_helper_ACCEPTED, mode)
    main = next(stmt for stmt in lowered.body
                if isinstance(stmt, ast.FunctionDef) and stmt.name == 'main')
    spec = main.args
    positional = [*spec.posonlyargs, *spec.args]
    pairs = [*zip(positional[len(positional) - len(spec.defaults):], spec.defaults),
             *zip(spec.kwonlyargs, spec.kw_defaults)]
    ids = {}
    for param, default in pairs:
        if isinstance(default, ast.Call):
            ids[param.arg] = next(kw.value.value for kw in default.keywords if kw.arg == '_id')
    assert ids == {name: name for name in ('a', 'length', 'src', 'tf', 'k', 'x')}


@pytest.mark.parametrize('backend', ['python', 'verify'])
def __test_own_input_function_is_not_an_input__(backend, tmp_path, monkeypatch):
    """A function the script names ``input`` itself is no PyneCore input: its calls are
    neither rejected nor given an ``_id``."""
    monkeypatch.setenv('PYNE_AST_BACKEND', backend)
    body = ('def input(x, _id=None):\n'
            '    assert _id is None\n'
            '    return x * 2\n'
            '@script.indicator("p")\n'
            'def main(n=input(3)):\n'
            '    plot(close + n + input(4))\n')
    source = ('"""\n@pyne\n"""\n'
              'from pynecore.lib import script, close, plot\n'
              '\n') + body
    path = tmp_path / 'case.py'
    path.write_text(source, encoding='utf-8')
    tree = ast.parse(source)
    pyne_mode = import_hook._module_mode(tree)[1]
    analysed = import_hook._analyse_tree(tree, source, path, pyne_mode)
    lowered = import_hook._lower_tree(analysed, path, pyne_mode)[0]
    calls = [node for node in ast.walk(lowered)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == 'input']
    assert len(calls) == 2
    assert all(kw.arg != '_id' for call in calls for kw in call.keywords)


@pytest.mark.parametrize('backend', ['python', 'verify'])
def __test_source_rebinding_stays_in_its_main__(backend, tmp_path, monkeypatch):
    """A source input's rebinding goes into the main() that declares it, not into a
    nested function of the same name."""
    monkeypatch.setenv('PYNE_AST_BACKEND', backend)
    lowered = __test_helper_lower(tmp_path, '@script.indicator("p")\n'
                                            'def main(src=input.source(close, "S")):\n'
                                            '    def main():\n'
                                            '        return 1\n'
                                            '    plot(src + main())\n', '')
    outer = next(stmt for stmt in lowered.body
                 if isinstance(stmt, ast.FunctionDef) and stmt.name == 'main')
    inner = next(stmt for stmt in ast.walk(outer)
                 if isinstance(stmt, ast.FunctionDef) and stmt is not outer)

    def rebinds(function: ast.FunctionDef) -> bool:
        return any(isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                   and isinstance(stmt.value.func, ast.Name)
                   and stmt.value.func.id == '__pyne_getattr__' for stmt in function.body)

    assert rebinds(outer)
    assert not rebinds(inner)
